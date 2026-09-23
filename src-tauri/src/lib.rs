use anyhow::{anyhow, Result};
use axum::{
    extract::{Json as AxumJson, Path as AxumPath, State as AxumState},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
        Response,
    },
    routing::post,
    Router,
};
use base64::Engine;
use futures_util::StreamExt;
use rand::{prelude::IndexedRandom, seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, sync::Arc, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use tauri::{AppHandle, Manager};
use tauri::ipc::Channel;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_stream::{wrappers::UnboundedReceiverStream, Stream};
use tower_http::services::ServeDir;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: String, name: String, #[serde(rename = "type")] kind: String, path: String, size: u64,
    #[serde(default)] base_model: Option<String>,
    #[serde(default)] tags: Vec<String>,
    #[serde(default)] activation_tags: Vec<String>,
    #[serde(default)] character: bool,
    #[serde(default)] thumbnail: Option<String>,
    #[serde(default)] source: String,
    #[serde(default)] cache_name: Option<String>,
    #[serde(default)] cache_description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LibrarySnapshot { checkpoints: Vec<ModelInfo>, loras: Vec<ModelInfo>, source_roots: Vec<String>, warnings: Vec<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScanRequest { comfy_root: String, raphael_root: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PickResult { path: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmSettings {
    provider: String,
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
    #[serde(default = "default_context_tokens")]
    context_tokens: u32,
}

fn default_context_tokens() -> u32 { 16384 }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelectedLora {
    id: String, name: String, path: String, weight: f32,
    activation_tags: Vec<String>, tags: Vec<String>, description: Option<String>,
    character: bool, base_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SceneSelection {
    setting: String, pose: String, expression: String, character: String, dress: String, composition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreparedGeneration {
    checkpoint: ModelInfo, loras: Vec<SelectedLora>, scene: SceneSelection, compatibility_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PrepareRequest {
    checkpoint: ModelInfo, loras: Vec<ModelInfo>,
    #[serde(default)] selected_lora_ids: Vec<String>,
    setting: String, pose: String, expression: String, character: String,
    dress: String, composition: String, additional: String,
    random_lora_min: u32, random_lora_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptPair { positive_prompt: String, negative_prompt: String, rationale: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmRequest { settings: LlmSettings, system_prompt: String, user_prompt: String }

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LlmDelta { text: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkflowRequest {
    checkpoint: ModelInfo, loras: Vec<SelectedLora>, width: u32, height: u32,
    steps: u32, cfg: f32, sampler: String, seed: i64,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InjectRequest {
    workflow: Value,
    #[serde(default)]
    positivePrompt: Option<String>,
    #[serde(default)]
    positive_prompt: Option<String>,
    #[serde(default)]
    negativePrompt: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FinalizePromptRequest {
    prompt_pair: PromptPair,
    loras: Vec<SelectedLora>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubmitRequest { comfy_url: String, workflow: Value }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MonitorComfyRequest { comfy_url: String, prompt_id: String }

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComfyProgress {
    percent: f32,
    current: u32,
    total: u32,
    node: Option<String>,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComfyGenerationResult {
    image_data_url: Option<String>,
    filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryRecord { id: String, timestamp: String, payload: Value }

struct WebHostRuntime {
    port: u16,
    lan_url: String,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct AppState {
    active_stream: Arc<Mutex<bool>>,
    web_host: Arc<Mutex<Option<WebHostRuntime>>>,
}

#[derive(Clone)]
struct WebApiState {
    app: AppHandle,
}

fn norm(s: &str) -> String { s.trim().to_lowercase().replace([' ', '_', '-', '.', '/'], "") }
fn base_url(s: &str) -> String { s.trim().trim_end_matches('/').to_string() }

fn val_str(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_str()).map(str::to_string))
}

fn val_vec(v: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_array()).map(|a|
        a.iter().filter_map(|x| x.as_str().map(str::to_string)).filter(|x| !x.is_empty()).collect::<Vec<_>>()
    )).unwrap_or_default()
}

fn is_character(tags: &[String]) -> bool { tags.iter().any(|x| norm(x) == "character") }

fn thumbnail(v: &Value, root: &Path) -> Option<String> {
    for key in ["thumbnail", "thumbnailPath", "cover", "coverImage", "preview", "localThumbnail"] {
        if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
            let q = PathBuf::from(s);
            let q = if q.is_absolute() { q } else { root.join(q) };
            if q.exists() { return Some(q.to_string_lossy().to_string()); }
        }
    }
    None
}

fn cache_objects(v: &Value, root: &Path, out: &mut Vec<ModelInfo>, depth: usize) {
    if depth > 7 { return; }
    if let Some(name) = val_str(v, &["name","modelName","title","displayName"]) {
        let kind_raw = val_str(v, &["type","modelType","model_type"]).unwrap_or_default().to_lowercase();
        let path = val_str(v, &["path","filePath","file_path","modelPath","location","relativePath"]).unwrap_or_default();
        let base = val_str(v, &["baseModel","base_model","base","trainedModel"]);
        let tags = val_vec(v, &["tags","modelTags","model_tags","baseTags"]);
        let activation = val_vec(v, &["activationTags","activation_tags","triggerWords","trainedWords","trained_words"]);
        let desc = val_str(v, &["description","desc"]);
        let kind = if kind_raw.contains("checkpoint") { "checkpoint" } else if kind_raw.contains("lora") || base.is_some() { "lora" } else { "" };
        if !kind.is_empty() {
            let p = if path.is_empty() { String::new() } else {
                let q = PathBuf::from(&path);
                if q.is_absolute() { q.to_string_lossy().to_string() } else { root.join(q).to_string_lossy().to_string() }
            };
            out.push(ModelInfo {
                id: "cache-".to_string()+&norm(&name), name, kind: kind.to_string(), path: p, size: 0,
                base_model: base, tags: tags.clone(),
                activation_tags: if activation.is_empty() { tags.clone() } else { activation },
                character: is_character(&tags), thumbnail: thumbnail(v,root), source: "raphael-cache".into(),
                cache_name: None, cache_description: desc,
            });
        }
    }
    match v {
        Value::Object(m) => for x in m.values() { cache_objects(x,root,out,depth+1); },
        Value::Array(a) => for x in a.iter().take(1200) { cache_objects(x,root,out,depth+1); },
        _ => {}
    }
}

fn scan_cache(root: &Path) -> Vec<ModelInfo> {
    if !root.exists() { return vec![]; }
    let mut out = Vec::new();
    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() || entry.path().extension().and_then(|x|x.to_str()) != Some("json") { continue; }
        if entry.metadata().map(|m|m.len()>4_000_000).unwrap_or(true) { continue; }
        let bytes = match fs::read(entry.path()) { Ok(x)=>x, Err(_)=>continue };
        let value = match serde_json::from_slice::<Value>(&bytes) { Ok(x)=>x, Err(_)=>continue };
        cache_objects(&value,root,&mut out,0);
    }
    out
}


fn is_path_under(root: &Path, candidate: &Path) -> bool {
    let root_abs = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let candidate_abs = candidate.canonicalize().unwrap_or_else(|_| candidate.to_path_buf());
    if candidate_abs.starts_with(&root_abs) {
        return true;
    }
    let root_s = root_abs.to_string_lossy().replace('\\', "/").to_lowercase();
    let candidate_s = candidate_abs.to_string_lossy().replace('\\', "/").to_lowercase();
    candidate_s == root_s || candidate_s.starts_with(&(root_s + "/"))
}

fn manager_db_candidates(extra_root: Option<&str>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(value) = extra_root.map(str::trim).filter(|x| !x.is_empty()) {
        let p = PathBuf::from(value);
        if p.is_file() && p.file_name().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("raphael.db")).unwrap_or(false) {
            out.push(p);
        } else if p.is_dir() {
            out.push(p.join("raphael.db"));
        }
    }
    for env_name in ["APPDATA", "LOCALAPPDATA"] {
        if let Ok(base) = std::env::var(env_name) {
            let base = PathBuf::from(base);
            for rel in [
                PathBuf::from("com.raphael.modelmanager").join("raphael.db"),
                PathBuf::from("Raphael Model Manager").join("raphael.db"),
                PathBuf::from("Raphael-Model-Manager").join("raphael.db"),
            ] {
                out.push(base.join(rel));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn manager_model_type(raw: &str) -> Option<&'static str> {
    match raw {
        "Checkpoint" | "checkpoint" => Some("checkpoint"),
        "LoRA" | "lora" => Some("lora"),
        _ => None,
    }
}

fn load_manager_models(db_path: &Path, comfy_root: &Path) -> Result<(Vec<ModelInfo>, Vec<ModelInfo>), String> {
    if !db_path.is_file() {
        return Ok((vec![], vec![]));
    }
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Could not open Raphael Model Manager database {}: {}", db_path.display(), e))?;

    let mut stmt = conn.prepare(
        "SELECT id,path,filename,model_type,size_bytes,base_model,description,tags_json,activation_json,thumbnail_path,cover_path,civitai_name,version_name
         FROM models"
    ).map_err(|e| format!("Raphael database schema error: {}", e))?;

    let mut checkpoints = Vec::new();
    let mut loras = Vec::new();
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let path: String = row.get(1)?;
        let filename: String = row.get(2)?;
        let model_type: String = row.get(3)?;
        let size_bytes: i64 = row.get(4)?;
        let base_model: Option<String> = row.get(5)?;
        let description: Option<String> = row.get(6)?;
        let tags_json: String = row.get(7)?;
        let activation_json: String = row.get(8)?;
        let thumbnail_path: Option<String> = row.get(9)?;
        let cover_path: Option<String> = row.get(10)?;
        let civitai_name: Option<String> = row.get(11)?;
        let version_name: Option<String> = row.get(12)?;
        Ok((id,path,filename,model_type,size_bytes,base_model,description,tags_json,activation_json,thumbnail_path,cover_path,civitai_name,version_name))
    }).map_err(|e| format!("Could not read Raphael model records: {}", e))?;

    for row in rows {
        let (id,path,filename,model_type,size_bytes,base_model,description,tags_json,activation_json,thumbnail_path,cover_path,civitai_name,version_name) =
            row.map_err(|e| format!("Could not decode Raphael model record: {}", e))?;
        let Some(kind) = manager_model_type(&model_type) else { continue };
        let model_path = PathBuf::from(&path);
        if !model_path.exists() || !is_path_under(comfy_root, &model_path) {
            continue;
        }
        let mut tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let activation_tags: Vec<String> = serde_json::from_str(&activation_json).unwrap_or_default();
        if tags.is_empty() && kind == "lora" && !activation_tags.is_empty() {
            tags = activation_tags.clone();
        }
        let thumbnail = cover_path
            .or(thumbnail_path)
            .map(PathBuf::from)
            .filter(|p| p.is_file())
            .map(|p| p.to_string_lossy().to_string());

        let character = is_character(&tags);
        let model = ModelInfo {
            id: format!("raphael-{}", id),
            name: civitai_name.or(version_name).unwrap_or(filename),
            kind: kind.to_string(),
            path: model_path.to_string_lossy().to_string(),
            size: size_bytes.max(0) as u64,
            base_model,
            tags,
            activation_tags,
            character,
            thumbnail,
            source: "raphael-model-manager".into(),
            cache_name: None,
            cache_description: description,
        };
        if kind == "checkpoint" {
            checkpoints.push(model);
        } else {
            loras.push(model);
        }
    }

    Ok((checkpoints, loras))
}

fn file_type_from_path(path: &Path, root: &Path) -> &'static str {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let first = rel.components().next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match first.as_str() {
        "checkpoints" | "checkpoint" | "diffusion_models" | "unet" | "unets" => "checkpoint",
        "loras" | "lora" | "lycoris" => "lora",
        _ => "",
    }
}

fn is_model_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase().as_str(),
        "safetensors" | "ckpt" | "pt" | "pth" | "bin" | "gguf" | "onnx"
    )
}

fn scan_disk(root: &Path) -> (Vec<ModelInfo>,Vec<ModelInfo>) {
    let mut cps = Vec::new();
    let mut ls = Vec::new();
    if !root.exists() { return (cps, ls); }

    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() || !is_model_file(entry.path()) { continue; }
        let kind = file_type_from_path(entry.path(), root);
        if kind.is_empty() { continue; }

        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let name = entry.path().file_stem().and_then(|x| x.to_str()).unwrap_or("model").to_string();
        let model = ModelInfo {
            id: "disk-".to_string() + &norm(&entry.path().to_string_lossy()),
            name,
            kind: kind.into(),
            path: entry.path().to_string_lossy().to_string(),
            size,
            base_model: None,
            tags: vec![],
            activation_tags: vec![],
            character: false,
            thumbnail: None,
            source: "comfyui".into(),
            cache_name: None,
            cache_description: None,
        };
        if kind == "checkpoint" { cps.push(model); } else { ls.push(model); }
    }
    (cps, ls)
}

fn merge_one(d:&mut ModelInfo,c:&ModelInfo){
    if d.base_model.is_none(){d.base_model=c.base_model.clone();}
    if d.tags.is_empty(){d.tags=c.tags.clone();}
    if d.activation_tags.is_empty(){d.activation_tags=c.activation_tags.clone();}
    if d.thumbnail.is_none(){d.thumbnail=c.thumbnail.clone();}
    d.character|=c.character; d.cache_name=Some(c.name.clone());
    if d.cache_description.is_none(){d.cache_description=c.cache_description.clone();}
    d.source="merged".into();
}

fn merge(list:&mut Vec<ModelInfo>,cache:&[ModelInfo]){
    for d in list.iter_mut(){
        let dn=norm(&d.name); let dp=norm(&d.path);
        if let Some(c)=cache.iter().find(|c|norm(&c.name)==dn||(!c.path.is_empty()&&norm(&c.path).ends_with(&dp))){merge_one(d,c);}
    }
}

#[tauri::command]
fn pick_folder()->Result<PickResult,String>{
    Ok(PickResult{path:rfd::FileDialog::new().pick_folder().map(|x|x.to_string_lossy().to_string())})
}

#[derive(Debug, Serialize)]
struct RaphaelConfig {
    models_root: Option<String>,
    db_path: Option<String>,
}

#[tauri::command]
fn discover_raphael_config() -> RaphaelConfig {
    for db in manager_db_candidates(None) {
        if !db.is_file() { continue; }
        if let Ok(conn) = Connection::open_with_flags(&db, OpenFlags::SQLITE_OPEN_READ_ONLY) {
            if let Ok(models_root) = conn.query_row(
                "SELECT value FROM settings WHERE key='models_root'",
                [],
                |row| row.get::<_, String>(0)
            ).optional() {
                return RaphaelConfig {
                    models_root,
                    db_path: Some(db.to_string_lossy().to_string()),
                };
            }
        }
    }
    RaphaelConfig { models_root: None, db_path: None }
}

#[tauri::command]
fn discover_raphael_roots() -> Vec<String> {
    manager_db_candidates(None)
        .into_iter()
        .filter_map(|p| p.parent().map(|x| x.to_string_lossy().to_string()))
        .collect()
}

#[tauri::command]
fn scan_library(req:ScanRequest)->Result<LibrarySnapshot,String>{
    let root = Path::new(&req.comfy_root);
    if !root.is_dir() {
        return Err(format!("ComfyUI models root is not a folder: {}", req.comfy_root));
    }

    let (mut cps, mut ls) = scan_disk(root);
    let mut roots = vec![req.comfy_root.clone()];
    let mut manager_loaded = false;

    for db in manager_db_candidates(req.raphael_root.as_deref()) {
        match load_manager_models(&db, root) {
            Ok((manager_cps, manager_loras)) if !manager_cps.is_empty() || !manager_loras.is_empty() => {
                cps = manager_cps;
                ls = manager_loras;
                roots.push(db.to_string_lossy().to_string());
                manager_loaded = true;
                break;
            }
            Ok(_) => {}
            Err(_) => {}
        }
    }

    if !manager_loaded {
        if let Some(rr)=req.raphael_root.as_deref().filter(|x|!x.trim().is_empty()) {
            let cache=scan_cache(Path::new(rr));
            roots.push(rr.to_string());
            let cp:Vec<_>=cache.iter().filter(|x|x.kind=="checkpoint").cloned().collect();
            let lr:Vec<_>=cache.iter().filter(|x|x.kind=="lora").cloned().collect();
            merge(&mut cps,&cp);
            merge(&mut ls,&lr);
        }
    }

    cps.sort_by_key(|x| x.name.to_lowercase());
    ls.sort_by_key(|x| x.name.to_lowercase());

    let mut warnings=Vec::new();
    if !manager_loaded {
        warnings.push("Raphael Model Manager database was not found; using ComfyUI filesystem scan. Start/scan Raphael Model Manager or configure its raphael.db path for cached thumbnails, base metadata, tags and activation prompts.".into());
    }
    if cps.is_empty(){warnings.push("No checkpoints found under the selected ComfyUI models root.".into());}
    if ls.is_empty(){warnings.push("No LoRAs found under the selected ComfyUI models root.".into());}
    Ok(LibrarySnapshot{checkpoints:cps,loras:ls,source_roots:roots,warnings})
}

#[tauri::command]
async fn list_provider_models(settings:LlmSettings)->Result<Vec<String>,String>{
    let client=reqwest::Client::new(); let base=base_url(&settings.base_url);
    let (url,need_auth)=if settings.provider=="ollama"{(format!("{}/api/tags",base),false)}else{(if base.ends_with("/v1"){format!("{}/models",base)}else{format!("{}/v1/models",base)},true)};
    let mut request=client.get(url); if need_auth&&!settings.api_key.is_empty(){request=request.bearer_auth(settings.api_key);}
    let response=request.send().await.map_err(|e|e.to_string())?;
    if !response.status().is_success(){return Err(format!("Provider returned {}",response.status()));}
    let v:Value=response.json().await.map_err(|e|e.to_string())?;
    let data:Vec<Value>=if settings.provider=="ollama"{v.get("models").and_then(|x|x.as_array()).cloned().unwrap_or_default()}else{v.get("data").and_then(|x|x.as_array()).cloned().unwrap_or_default()};
    Ok(data.iter().filter_map(|m|m.get("name").or_else(||m.get("id")).and_then(|x|x.as_str()).map(str::to_string)).collect())
}

fn random_one(values:&[&str])->String{values.choose(&mut rand::rng()).unwrap_or(&"").to_string()}

#[tauri::command]
fn prepare_generation(req:PrepareRequest)->Result<PreparedGeneration,String>{
    let keys:Vec<String>=req.checkpoint.tags.iter().chain(req.checkpoint.base_model.iter()).chain(std::iter::once(&req.checkpoint.name)).map(|x|norm(x)).filter(|x|x.len()>2).collect();
    let compatible:Vec<&ModelInfo>=req.loras.iter().filter(|l|{
        let explicit=l.base_model.as_ref().map(|b|{let b=norm(b);keys.iter().any(|k|k==&b||k.contains(&b)||b.contains(k))}).unwrap_or(false);
        let tagged=l.tags.iter().map(|t|norm(t)).any(|t|keys.iter().any(|k|k==&t||k.contains(&t)||t.contains(k)));
        explicit||tagged
    }).collect();
    if compatible.is_empty(){return Err("No compatible LoRAs were found for the selected checkpoint.".into());}

    let manual:Vec<&ModelInfo>=if req.selected_lora_ids.is_empty(){
        Vec::new()
    }else{
        let picked:Vec<&ModelInfo>=req.selected_lora_ids.iter()
            .filter_map(|id|compatible.iter().copied().find(|l|&l.id==id))
            .collect();
        if picked.len()!=req.selected_lora_ids.len(){
            return Err("One or more manually selected LoRAs are no longer compatible with this checkpoint.".into());
        }
        picked
    };

    let chars:Vec<&ModelInfo>=compatible.iter().copied().filter(|l|l.character||l.tags.iter().any(|t|norm(t)=="character")).collect();
    if chars.is_empty(){return Err("No compatible character LoRA was found. Character identity is restricted to LoRAs marked with the character tag.".into());}

    let wanted=req.character.trim().to_lowercase();
    let manual_character=manual.iter().copied().find(|l|l.character||l.tags.iter().any(|t|norm(t)=="character"));
    if !wanted.is_empty() && manual_character.is_some() &&
        !manual_character.unwrap().name.to_lowercase().contains(&wanted) &&
        !manual_character.unwrap().tags.iter().any(|t|t.to_lowercase().contains(&wanted)) {
        return Err("The selected character LoRA does not match the requested character.".into());
    }

    let character=if !wanted.is_empty(){
        *chars.iter().find(|l|l.name.to_lowercase().contains(&wanted)||l.tags.iter().any(|t|t.to_lowercase().contains(&wanted))).ok_or("Requested character does not match a compatible character LoRA.")?
    }else if let Some(l)=manual_character{
        l
    }else{
        *chars.choose(&mut rand::rng()).unwrap()
    };

    let chosen:Vec<&ModelInfo>=if !manual.is_empty(){
        let mut picked=manual.clone();
        if !picked.iter().any(|l|l.id==character.id){picked.insert(0,character);}
        picked
    }else{
        let count=rand::rng().random_range(req.random_lora_min.max(1)..=req.random_lora_max.max(req.random_lora_min.max(1))) as usize;
        let mut pool:Vec<&ModelInfo>=compatible.into_iter().filter(|x|x.id!=character.id).collect();
        pool.shuffle(&mut rand::rng());
        let mut picked=vec![character];
        picked.extend(pool.into_iter().take(count.saturating_sub(1)));
        picked
    };

    let loras=chosen.into_iter().map(|l|SelectedLora{
        id:l.id.clone(),name:l.name.clone(),path:l.path.clone(),
        weight:rand::rng().random_range(0.65..=1.0),
        activation_tags:l.activation_tags.clone(),
        tags:l.tags.clone(),
        description:l.cache_description.clone(),
        character:l.character||l.tags.iter().any(|t|norm(t)=="character"),
        base_model:l.base_model.clone()
    }).collect();

    let scene=SceneSelection{
        setting:if req.setting.trim().is_empty(){random_one(&["rooftop at blue hour","rainy neon alley","quiet shrine at dawn","sunlit train platform","moonlit forest clearing","coastal city street after rain"])}else{req.setting.trim().into()},
        pose:if req.pose.trim().is_empty(){random_one(&["standing naturally","walking forward","sitting with one knee raised","looking over the shoulder","dynamic three-quarter pose","leaning against a wall"])}else{req.pose.trim().into()},
        expression:if req.expression.trim().is_empty(){random_one(&["soft smile","confident","curious","slightly mischievous","calm","surprised"])}else{req.expression.trim().into()},
        character:character.name.clone(),
        dress:if req.dress.trim().is_empty(){random_one(&["modern casual outfit","layered streetwear","school uniform","elegant dress","light summer clothes","fantasy-inspired outfit"])}else{req.dress.trim().into()},
        composition:if req.composition.trim().is_empty(){random_one(&["full body","three-quarter shot","medium shot","cinematic wide shot","portrait crop"])}else{req.composition.trim().into()},
    };
    Ok(PreparedGeneration{checkpoint:req.checkpoint,loras,scene,compatibility_keys:keys})
}

#[tauri::command]
async fn stream_llm(
    state:tauri::State<'_,AppState>,
    req:LlmRequest,
    on_event:Channel<LlmDelta>,
)->Result<(),String>{
    {
        let mut busy=state.active_stream.lock().await;
        if *busy{return Err("An LLM stream is already active.".into())}
        *busy=true;
    }
    let result=stream_llm_inner(req, |delta| {
        on_event.send(delta).map_err(|e|format!("LLM stream channel closed: {}",e))
    }).await;
    *state.active_stream.lock().await=false;
    result
}

async fn stream_llm_inner<F>(req:LlmRequest, mut emit:F)->Result<(),String>
where F:FnMut(LlmDelta)->Result<(),String> + Send
{
    let client=reqwest::Client::new();
    let base=base_url(&req.settings.base_url);
    let (url,mut body,ollama)=if req.settings.provider=="ollama"{
        (format!("{}/api/chat",base),json!({
            "model":req.settings.model,
            "stream":true,
            "format":"json",
            "think":false,
            "messages":[
                {"role":"system","content":req.system_prompt},
                {"role":"user","content":req.user_prompt}
            ],
            "options":{
                "temperature":req.settings.temperature,
                "num_predict":req.settings.max_tokens,
                "num_ctx":req.settings.context_tokens
            }
        }),true)
    }else{
        (if base.ends_with("/v1"){format!("{}/chat/completions",base)}else{format!("{}/v1/chat/completions",base)},json!({
            "model":req.settings.model,
            "stream":true,
            "temperature":req.settings.temperature,
            "max_tokens":req.settings.max_tokens,
            "response_format":{"type":"json_object"},
            "messages":[
                {"role":"system","content":req.system_prompt},
                {"role":"user","content":req.user_prompt}
            ]
        }),false)
    };

    let auth_key=req.settings.api_key.clone();
    let mut request=client.post(&url).json(&body);
    if !ollama&&!auth_key.trim().is_empty(){request=request.bearer_auth(auth_key.clone());}
    let mut response=request.send().await.map_err(|e|e.to_string())?;

    if !ollama && response.status()==reqwest::StatusCode::BAD_REQUEST {
        if let Some(obj)=body.as_object_mut(){obj.remove("response_format");}
        let mut retry=client.post(&url).json(&body);
        if !auth_key.trim().is_empty(){retry=retry.bearer_auth(auth_key);}
        response=retry.send().await.map_err(|e|e.to_string())?;
    }

    if !response.status().is_success(){
        let status=response.status();
        let detail=response.text().await.unwrap_or_default();
        return Err(format!("LLM returned HTTP {}: {}",status,detail));
    }

    let mut stream=response.bytes_stream();
    let mut buffer=String::new();

    while let Some(chunk)=stream.next().await{
        buffer.push_str(&String::from_utf8_lossy(&chunk.map_err(|e|e.to_string())?));
        while let Some(pos)=buffer.find('\n'){
            let line=buffer[..pos].trim_end_matches('\r').to_string();
            buffer=buffer[pos+1..].to_string();
            if line.trim().is_empty(){continue}
            let data=if ollama{line.trim()}else{line.strip_prefix("data: ").unwrap_or("").trim()};
            if data.is_empty()||data=="[DONE]"{continue}
            let v:Value=match serde_json::from_str(data){Ok(x)=>x,Err(_)=>continue};
            let token=if ollama{
                v.get("message").and_then(|x|x.get("content")).and_then(|x|x.as_str()).unwrap_or("")
            }else{
                v.get("choices").and_then(|x|x.get(0)).and_then(|x|x.get("delta")).and_then(|x|x.get("content")).and_then(|x|x.as_str()).unwrap_or("")
            };
            if !token.is_empty(){emit(LlmDelta{text:token.to_string()})?;}
        }
    }

    let tail=buffer.trim();
    if !tail.is_empty() && tail!="[DONE]"{
        let data=if ollama{tail}else{tail.strip_prefix("data: ").unwrap_or(tail).trim()};
        if !data.is_empty(){
            if let Ok(v)=serde_json::from_str::<Value>(data){
                let token=if ollama{
                    v.get("message").and_then(|x|x.get("content")).and_then(|x|x.as_str()).unwrap_or("")
                }else{
                    v.get("choices").and_then(|x|x.get(0)).and_then(|x|x.get("delta")).and_then(|x|x.as_str()).unwrap_or("")
                };
                if !token.is_empty(){emit(LlmDelta{text:token.to_string()})?;}
            }
        }
    }
    Ok(())
}

fn parse_json(raw:&str)->Result<Value>{
    let mut clean=raw.trim().to_string();
    if let Some(end)=clean.rfind("</think>"){clean=clean[end+8..].trim().to_string();}
    clean=clean.replace("```json","").replace("```","").trim().to_string();
    for candidate in [clean.clone(),{
        let a=clean.find('{').unwrap_or(usize::MAX);
        let b=clean.rfind('}').unwrap_or(0);
        if a!=usize::MAX && b>=a {clean[a..=b].to_string()} else {String::new()}
    }] {
        if candidate.is_empty(){continue;}
        if let Ok(v)=serde_json::from_str::<Value>(&candidate){
            if let Value::String(inner)=&v{
                if let Ok(parsed)=serde_json::from_str::<Value>(inner){return Ok(parsed);}
            }
            return Ok(v);
        }
    }
    Err(anyhow!("No JSON object in LLM output"))
}

fn fallback_prompt_pair(raw:&str)->Option<PromptPair>{
    let mut text=raw.trim().to_string();
    if let Some(end)=text.rfind("</think>"){text=text[end+8..].trim().to_string();}
    text=text.replace("```","").trim().to_string();

    let lower=text.to_lowercase();
    let positive_markers=["positive_prompt:", "positive prompt:", "positive:"];
    let negative_markers=["negative_prompt:", "negative prompt:", "negative:"];
    let p_marker=positive_markers.iter().find_map(|m|lower.find(m).map(|i|(i,i+m.len())));
    let n_marker=negative_markers.iter().find_map(|m|lower.find(m).map(|i|(i,i+m.len())));

    if let (Some((p_pos,p_value)),Some((n_pos,n_value)))=(p_marker,n_marker){
        if p_pos<n_pos{
            let positive=text[p_value..n_pos].trim();
            let negative=text[n_value..].trim();
            if !positive.is_empty() && !negative.is_empty(){
                return Some(PromptPair{
                    positive_prompt:positive.trim_matches(|c|c=='"'||c=='\'').trim().to_string(),
                    negative_prompt:negative.trim_matches(|c|c=='"'||c=='\'').trim().to_string(),
                    rationale:Some("Recovered from a non-JSON LLM response.".into()),
                });
            }
        }
    }

    if !text.is_empty(){
        return Some(PromptPair{
            positive_prompt:text,
            negative_prompt:"low quality, blurry, bad anatomy, malformed hands, extra fingers, duplicate, text, watermark".into(),
            rationale:Some("LLM did not return JSON; used its text as the positive prompt.".into()),
        });
    }
    None
}

#[tauri::command]
fn parse_prompt_pair(raw:String)->Result<PromptPair,String>{
    match parse_json(&raw) {
        Ok(v)=>Ok(PromptPair{
            positive_prompt:v.get("positive_prompt").and_then(|x|x.as_str()).ok_or("Missing positive_prompt")?.to_string(),
            negative_prompt:v.get("negative_prompt").and_then(|x|x.as_str()).ok_or("Missing negative_prompt")?.to_string(),
            rationale:v.get("rationale").and_then(|x|x.as_str()).map(str::to_string),
        }),
        Err(_)=>fallback_prompt_pair(&raw).ok_or_else(||"LLM returned no usable prompt text.".to_string()),
    }
}

fn strip_exact_ci(text:&str, needle:&str)->String{
    if needle.trim().is_empty(){return text.to_string();}
    let lower=text.to_lowercase();
    let needle_lower=needle.to_lowercase();
    let mut out=String::with_capacity(text.len());
    let mut cursor=0usize;
    while let Some(rel)=lower[cursor..].find(&needle_lower){
        let start=cursor+rel;
        out.push_str(&text[cursor..start]);
        cursor=start+needle.len();
    }
    out.push_str(&text[cursor..]);
    out
}

fn strip_generated_lora_syntax(text:&str)->String{
    let mut out=text.to_string();

    // Remove common LoRA implementation syntax such as <lora:name:1.0>.
    loop{
        let lower=out.to_lowercase();
        let Some(start)=lower.find("<lora:") else {break;};
        let Some(rel_end)=out[start..].find('>') else {break;};
        let end=start+rel_end+1;
        out.replace_range(start..end,"");
    }

    // Remove weighted bracket syntax for style/model/LoRA entries, e.g.
    // [Style - Example@4x0style]1.3
    let mut cleaned=String::with_capacity(out.len());
    let mut cursor=0usize;
    while cursor<out.len(){
        let remainder=&out[cursor..];
        let Some(open_rel)=remainder.find('[') else {
            cleaned.push_str(remainder);
            break;
        };
        let open=cursor+open_rel;
        cleaned.push_str(&out[cursor..open]);
        let Some(close_rel)=out[open+1..].find(']') else {
            cleaned.push_str(&out[open..]);
            break;
        };
        let close=open+1+close_rel;
        let inner=out[open+1..close].trim();
        let lower_inner=inner.to_lowercase();
        let mut end=close+1;
        let tail=&out[end..];
        let mut digits=0usize;
        for ch in tail.chars(){
            if ch.is_ascii_digit() || ch=='.' || ch=='-' {
                digits+=ch.len_utf8();
            } else {
                break;
            }
        }
        let weighted=digits>0;
        let implementation_hint=
            lower_inner.contains("lora") ||
            lower_inner.contains("style -") ||
            lower_inner.contains("model -") ||
            lower_inner.contains("character -") ||
            lower_inner.contains('@');
        if weighted && implementation_hint {
            end+=digits;
        }else{
            cleaned.push_str(&out[open..=close]);
        }
        cursor=end;
    }

    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn finalize_positive_prompt(raw:&str, loras:&[SelectedLora])->String{
    let mut positive=strip_generated_lora_syntax(raw).trim().trim_matches(',').trim().to_string();
    let mut triggers=Vec::new();
    for lora in loras {
        for tag in &lora.activation_tags {
            let tag=tag.trim();
            if tag.is_empty(){continue;}
            positive=strip_exact_ci(&positive,tag);
            if !triggers.iter().any(|x:&String|x.eq_ignore_ascii_case(tag)){
                triggers.push(tag.to_string());
            }
        }
    }
    positive=positive
        .split(',')
        .map(str::trim)
        .filter(|x|!x.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    if triggers.is_empty(){return positive;}
    if positive.is_empty(){return triggers.join(", ");}
    format!("{}, {}",positive,triggers.join(", "))
}

#[tauri::command]
fn finalize_prompt_pair(req:FinalizePromptRequest)->Result<PromptPair,String>{
    Ok(PromptPair{
        positive_prompt:finalize_positive_prompt(&req.prompt_pair.positive_prompt,&req.loras),
        negative_prompt:req.prompt_pair.negative_prompt.trim().to_string(),
        rationale:req.prompt_pair.rationale.clone(),
    })
}

fn comfy_relative_model_name(path: &str, folder: &str, fallback: &str) -> String {
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_lowercase();

    // Prefer a path relative to ComfyUI's models directory.
    // This handles both:
    //   .../models/unet/anima/file.safetensors -> anima\\file.safetensors
    //   .../models/anima/file.safetensors      -> anima\\file.safetensors
    if let Some(models_index) = lower.find("/models/") {
        let mut relative = normalized[models_index + "/models/".len()..].to_string();
        let folder_prefix = format!("{}/", folder);
        let relative_lower = relative.to_lowercase();
        if relative_lower.starts_with(&folder_prefix) {
            relative = relative[folder_prefix.len()..].to_string();
        } else if folder.eq_ignore_ascii_case("unet")
            && relative_lower.starts_with("diffusion_models/")
        {
            // Newer ComfyUI layouts may expose UNETLoader files from
            // models/diffusion_models, but the loader expects the path
            // relative to that directory (e.g. anima\\model.safetensors).
            relative = relative["diffusion_models/".len()..].to_string();
        }
        if !relative.is_empty() {
            return relative.replace('/', "\\");
        }
    }

    // Fallback for paths that cannot be anchored at a ComfyUI models root.
    let marker = format!("/{}/", folder);
    if let Some(index) = lower.find(&marker) {
        return normalized[index + marker.len()..].replace('/', "\\");
    }
    fallback.to_string()
}

#[tauri::command]
fn build_workflow(req:WorkflowRequest)->Result<Value,String>{
    // Reference workflow: Anima UNET + CLIP + VAE + text encoders + sampler + decode + save.
    // Only the model LoRA chain is generated dynamically.
    let mut map=serde_json::Map::new();

    let unet_name = comfy_relative_model_name(
        &req.checkpoint.path,
        "unet",
        Path::new(&req.checkpoint.path)
            .file_name()
            .and_then(|x|x.to_str())
            .unwrap_or(&req.checkpoint.name),
    );
    map.insert("13".into(),json!({
        "class_type":"UNETLoader",
        "inputs":{"unet_name":unet_name,"weight_dtype":"default"}
    }));

    map.insert("4".into(),json!({
        "class_type":"CLIPLoader",
        "inputs":{"clip_name":"anima\\oneObsession_anima29BV1_txt.safetensors","type":"stable_diffusion","device":"default"}
    }));

    map.insert("9".into(),json!({
        "class_type":"VAELoader",
        "inputs":{"vae_name":"anima\\qwen_image_vae.safetensors"}
    }));

    map.insert("5".into(),json!({
        "class_type":"CLIPTextEncode",
        "inputs":{"clip":["4",0],"text":"__POSITIVE_PROMPT__"}
    }));
    map.insert("6".into(),json!({
        "class_type":"CLIPTextEncode",
        "inputs":{"clip":["4",0],"text":"__NEGATIVE_PROMPT__"}
    }));

    map.insert("8".into(),json!({
        "class_type":"EmptyLatentImage",
        "inputs":{"width":req.width,"height":req.height,"batch_size":1}
    }));

    let mut last_model = "13".to_string();
    let mut next_id = 14u32;
    for lora in &req.loras {
        let id = next_id.to_string();
        let lora_name = comfy_relative_model_name(
            &lora.path,
            "loras",
            Path::new(&lora.path)
                .file_name()
                .and_then(|x|x.to_str())
                .unwrap_or(&lora.name),
        );
        map.insert(id.clone(),json!({
            "class_type":"LoraLoaderModelOnly",
            "inputs":{
                "model":[last_model.clone(),0],
                "lora_name":lora_name,
                "strength_model":lora.weight
            }
        }));
        last_model=id;
        next_id+=1;
    }

    map.insert("7".into(),json!({
        "class_type":"KSampler",
        "inputs":{
            "model":[last_model,0],
            "positive":["5",0],
            "negative":["6",0],
            "latent_image":["8",0],
            "seed":req.seed,
            "steps":req.steps,
            "cfg":req.cfg,
            "sampler_name":req.sampler,
            "scheduler":"normal",
            "denoise":1.0
        }
    }));

    map.insert("10".into(),json!({
        "class_type":"VAEDecode",
        "inputs":{"samples":["7",0],"vae":["9",0]}
    }));

    map.insert("12".into(),json!({
        "class_type":"SaveImage",
        "inputs":{"images":["10",0],"filename_prefix":"Anima"}
    }));

    Ok(Value::Object(map))
}

#[tauri::command]
fn inject_prompts(req:InjectRequest)->Result<Value,String>{
    let positive=req.positivePrompt.or(req.positive_prompt)
        .ok_or_else(||"inject_prompts requires positivePrompt".to_string())?;
    let negative=req.negativePrompt.or(req.negative_prompt)
        .ok_or_else(||"inject_prompts requires negativePrompt".to_string())?;

    let mut w=req.workflow;
    if let Some(map)=w.as_object_mut(){
        for node in map.values_mut(){
            if node.get("class_type").and_then(|x|x.as_str())==Some("CLIPTextEncode"){
                if let Some(inputs)=node.get_mut("inputs").and_then(|x|x.as_object_mut()){
                    if inputs.get("text").and_then(|x|x.as_str())==Some("__POSITIVE_PROMPT__"){
                        inputs.insert("text".into(),Value::String(positive.clone()));
                    }
                    if inputs.get("text").and_then(|x|x.as_str())==Some("__NEGATIVE_PROMPT__"){
                        inputs.insert("text".into(),Value::String(negative.clone()));
                    }
                }
            }
        }
    }
    Ok(w)
}

#[tauri::command]
async fn submit_to_comfy(req:SubmitRequest)->Result<Value,String>{
    let response=reqwest::Client::new().post(format!("{}/prompt",base_url(&req.comfy_url))).json(&json!({"prompt":req.workflow,"client_id":"raphael-prompt-forge"})).send().await.map_err(|e|e.to_string())?;
    let status=response.status(); let text=response.text().await.unwrap_or_default();
    if !status.is_success(){return Err(format!("ComfyUI returned HTTP {}: {}",status,text))}
    serde_json::from_str(&text).map_err(|e|format!("Invalid ComfyUI response: {}",e))
}

async fn fetch_comfy_history(client:&reqwest::Client, base:&str, prompt_id:&str)->Result<Option<Value>,String>{
    let response=client.get(format!("{}/history/{}",base,prompt_id)).send().await.map_err(|e|e.to_string())?;
    if !response.status().is_success(){return Ok(None);}
    let value=response.json::<Value>().await.map_err(|e|e.to_string())?;
    Ok(value.get(prompt_id).cloned())
}

fn first_comfy_image(history:&Value)->Option<(String,String,String)>{
    let outputs=history.get("outputs")?.as_object()?;
    for node in outputs.values(){
        let images=node.get("images")?.as_array()?;
        if let Some(image)=images.first(){
            let filename=image.get("filename")?.as_str()?.to_string();
            let subfolder=image.get("subfolder").and_then(|x|x.as_str()).unwrap_or("").to_string();
            let image_type=image.get("type").and_then(|x|x.as_str()).unwrap_or("output").to_string();
            return Some((filename,subfolder,image_type));
        }
    }
    None
}

async fn fetch_comfy_image(client:&reqwest::Client, base:&str, image:(String,String,String))->Result<String,String>{
    let (filename,subfolder,image_type)=image;
    let response=client.get(format!("{}/view",base))
        .query(&[
            ("filename",filename.as_str()),
            ("subfolder",subfolder.as_str()),
            ("type",image_type.as_str()),
        ])
        .send().await.map_err(|e|e.to_string())?;
    if !response.status().is_success(){
        return Err(format!("ComfyUI image fetch returned HTTP {}",response.status()));
    }
    let mime=response.headers().get(reqwest::header::CONTENT_TYPE)
        .and_then(|x|x.to_str().ok())
        .unwrap_or("image/png")
        .split(';').next().unwrap_or("image/png")
        .to_string();
    let bytes=response.bytes().await.map_err(|e|e.to_string())?;
    Ok(format!("data:{};base64,{}",mime,base64::engine::general_purpose::STANDARD.encode(bytes)))
}

async fn monitor_comfy_generation_inner<F>(
    req:MonitorComfyRequest,
    mut emit:F,
)->Result<ComfyGenerationResult,String>
where F:FnMut(ComfyProgress)->Result<(),String> + Send
{
    let base=base_url(&req.comfy_url);
    let ws_url=format!("{}/ws?clientId=raphael-prompt-forge",base.replace("https://","wss://").replace("http://","ws://"));
    let client=reqwest::Client::new();
    let mut socket=connect_async(ws_url).await.ok().map(|(stream, _response)| stream);
    let started=Instant::now();
    let mut last_percent=0.0f32;
    let mut last_current=0u32;
    let mut last_total=0u32;

    let _=emit(ComfyProgress{percent:0.0,current:0,total:1,node:None,status:"waiting".into()});

    loop{
        if let Some(history)=fetch_comfy_history(&client,&base,&req.prompt_id).await?{
            if let Some(image_ref)=first_comfy_image(&history){
                let image_data_url=fetch_comfy_image(&client,&base,image_ref.clone()).await?;
                let _=emit(ComfyProgress{percent:100.0,current:1,total:1,node:None,status:"done".into()});
                return Ok(ComfyGenerationResult{image_data_url:Some(image_data_url),filename:Some(image_ref.0)});
            }
        }
        if started.elapsed()>Duration::from_secs(30*60){
            return Err("ComfyUI generation timed out after 30 minutes.".into());
        }
        if let Some(ws)=socket.as_mut(){
            match timeout(Duration::from_millis(250),ws.next()).await{
                Ok(Some(Ok(Message::Text(text))))=>{
                    if let Ok(value)=serde_json::from_str::<Value>(text.as_ref()){
                        let kind=value.get("type").and_then(|x|x.as_str()).unwrap_or("");
                        let data=value.get("data").cloned().unwrap_or(Value::Null);
                        let message_prompt_id=data.get("prompt_id").and_then(|x|x.as_str()).unwrap_or("");
                        if message_prompt_id==req.prompt_id{
                            match kind{
                                "progress"=>{
                                    let current=data.get("value").and_then(|x|x.as_u64()).unwrap_or(0) as u32;
                                    let total=data.get("max").and_then(|x|x.as_u64()).unwrap_or(1) as u32;
                                    let percent=if total>0 {current as f32*100.0/total as f32}else{last_percent};
                                    last_percent=percent.clamp(0.0,100.0); last_current=current; last_total=total;
                                    let _=emit(ComfyProgress{percent:last_percent,current:last_current,total:last_total,node:data.get("node").and_then(|x|x.as_str()).map(str::to_string),status:"sampling".into()});
                                }
                                "executing"=>{
                                    let node=data.get("node").and_then(|x|x.as_str()).map(str::to_string);
                                    let progress_status=if node.is_some(){"running".into()}else{"finishing".into()};
                                    let _=emit(ComfyProgress{percent:last_percent,current:last_current,total:last_total,node,status:progress_status});
                                }
                                "execution_error"=>{
                                    return Err(data.get("exception_message").and_then(|x|x.as_str()).unwrap_or("ComfyUI execution failed.").to_string());
                                }
                                _=>{}
                            }
                        }
                    }
                }
                Ok(Some(Ok(Message::Close(_))))|Ok(None)=>{socket=None;}
                Ok(Some(Err(_)))=>{socket=None;}
                Err(_)=>{}
                Ok(Some(Ok(_)))=>{}
            }
        }else{sleep(Duration::from_millis(500)).await;}
    }
}

#[tauri::command]
async fn monitor_comfy_generation(
    req:MonitorComfyRequest,
    on_event:Channel<ComfyProgress>,
)->Result<ComfyGenerationResult,String>{
    monitor_comfy_generation_inner(req, |progress| {
        on_event.send(progress).map_err(|e|format!("ComfyUI channel closed: {}",e))
    }).await
}


fn lan_ip() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            let _ = socket.connect("8.8.8.8:80");
            socket.local_addr()
        })
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}

fn dist_directory(app:&AppHandle)->Option<PathBuf>{
    let mut candidates=Vec::new();
    if let Ok(resource)=app.path().resource_dir(){candidates.push(resource.join("dist"));}
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dist"));
    candidates.into_iter().find(|p|p.join("index.html").is_file())
}

fn http_error(message:String)->Response{
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        AxumJson(json!({"error":message}))
    ).into_response()
}

async fn web_command(
    AxumState(state):AxumState<WebApiState>,
    AxumPath(command):AxumPath<String>,
    AxumJson(body):AxumJson<Value>,
)->Response{
    if command=="stream_llm" {
        return web_stream_llm(AxumJson(body)).await.into_response();
    }
    if command=="monitor_comfy_generation" {
        return web_monitor_comfy(AxumJson(body)).await.into_response();
    }

    let req_value=body.get("req").cloned().unwrap_or_else(||body.clone());
    let result:Result<Value,String>=async {
        match command.as_str(){
        "discover_raphael_config"=>serde_json::to_value(discover_raphael_config()).map_err(|e|e.to_string()),
        "discover_raphael_roots"=>serde_json::to_value(discover_raphael_roots()).map_err(|e|e.to_string()),
        "list_provider_models"=>{
            let settings=serde_json::from_value::<LlmSettings>(req_value).map_err(|e|e.to_string())?;
            let models=list_provider_models(settings).await?;
            serde_json::to_value(models).map_err(|e|e.to_string())
        }
        "scan_library"=>{
            let request=serde_json::from_value::<ScanRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(scan_library(request)?).map_err(|e|e.to_string())
        }
        "prepare_generation"=>{
            let request=serde_json::from_value::<PrepareRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(prepare_generation(request)?).map_err(|e|e.to_string())
        }
        "parse_prompt_pair"=>{
            let raw=req_value.as_str().ok_or_else(||"raw prompt text is required".to_string())?;
            serde_json::to_value(parse_prompt_pair(raw.to_string())?).map_err(|e|e.to_string())
        }
        "finalize_prompt_pair"=>{
            let request=serde_json::from_value::<FinalizePromptRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(finalize_prompt_pair(request)?).map_err(|e|e.to_string())
        }
        "build_workflow"=>{
            let request=serde_json::from_value::<WorkflowRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(build_workflow(request)?).map_err(|e|e.to_string())
        }
        "inject_prompts"=>{
            let request=serde_json::from_value::<InjectRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(inject_prompts(request)?).map_err(|e|e.to_string())
        }
        "submit_to_comfy"=>{
            let request=serde_json::from_value::<SubmitRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(submit_to_comfy(request).await?).map_err(|e|e.to_string())
        }
        "load_history"=>serde_json::to_value(load_history(state.app.clone())?).map_err(|e|e.to_string()),
        "append_history"=>{
            let payload=body.get("payload").cloned().unwrap_or(Value::Null);
            serde_json::to_value(append_history(state.app.clone(),payload)?).map_err(|e|e.to_string())
        }
        "path_to_data_url"=>{
            let path=req_value.get("path").and_then(|x|x.as_str())
                .or_else(||req_value.as_str())
                .ok_or_else(||"path is required".to_string())?;
            serde_json::to_value(path_to_data_url(path.to_string())?).map_err(|e|e.to_string())
        }
        _=>Err(format!("Unknown API command: {}",command))
        }
    }.await;
    match result{Ok(value)=>AxumJson(value).into_response(),Err(e)=>http_error(e)}
}

async fn web_stream_llm(
    AxumJson(body):AxumJson<Value>,
)->Sse<impl Stream<Item=Result<Event,std::convert::Infallible>>>{
    let req:Result<LlmRequest,String>=serde_json::from_value(body.get("req").cloned().unwrap_or(body.clone())).map_err(|e|e.to_string());
    let (tx,rx)=tokio::sync::mpsc::unbounded_channel::<Result<Event,std::convert::Infallible>>();
    tokio::spawn(async move{
        match req{
            Ok(req)=>{
                let tx2=tx.clone();
                let mut emit=|delta:LlmDelta|{
                    let event=Event::default().json_data(json!({"text":delta.text,"done":false})).map_err(|e|format!("sse:{}",e))?;
                    tx2.send(Ok(event)).map_err(|_|"sse client disconnected".to_string())
                };
                match stream_llm_inner(req,&mut emit).await{
                    Ok(())=>{
                        let _=tx.send(Ok(Event::default().json_data(json!({"done":true})).unwrap_or_else(|_|Event::default())));
                    }
                    Err(e)=>{
                        let _=tx.send(Ok(Event::default().json_data(json!({"error":e})).unwrap_or_else(|_|Event::default())));
                    }
                }
            }
            Err(e)=>{
                let _=tx.send(Ok(Event::default().json_data(json!({"error":e})).unwrap_or_else(|_|Event::default())));
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

async fn web_monitor_comfy(
    AxumJson(body):AxumJson<Value>,
)->Sse<impl Stream<Item=Result<Event,std::convert::Infallible>>>{
    let req:Result<MonitorComfyRequest,String>=serde_json::from_value(body.get("req").cloned().unwrap_or(body.clone())).map_err(|e|e.to_string());
    let (tx,rx)=tokio::sync::mpsc::unbounded_channel::<Result<Event,std::convert::Infallible>>();
    tokio::spawn(async move{
        match req{
            Ok(req)=>{
                let tx2=tx.clone();
                let mut emit=|progress:ComfyProgress|{
                    let event=Event::default().json_data(json!({"progress":progress})).map_err(|e|format!("sse:{}",e))?;
                    tx2.send(Ok(event)).map_err(|_|"sse client disconnected".to_string())
                };
                match monitor_comfy_generation_inner(req,&mut emit).await{
                    Ok(result)=>{
                        let _=tx.send(Ok(Event::default().json_data(json!({"result":result,"done":true})).unwrap_or_else(|_|Event::default())));
                    }
                    Err(e)=>{
                        let _=tx.send(Ok(Event::default().json_data(json!({"error":e})).unwrap_or_else(|_|Event::default())));
                    }
                }
            }
            Err(e)=>{
                let _=tx.send(Ok(Event::default().json_data(json!({"error":e})).unwrap_or_else(|_|Event::default())));
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

fn now_id()->String{SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().to_string()}
fn history_path(app:&AppHandle)->Result<PathBuf,String>{let dir=app.path().app_data_dir().map_err(|e|e.to_string())?;fs::create_dir_all(&dir).map_err(|e|e.to_string())?;Ok(dir.join("generation-history.json"))}

#[tauri::command]
fn load_history(app:AppHandle)->Result<Vec<HistoryRecord>,String>{let p=history_path(&app)?;if !p.exists(){return Ok(vec![])}serde_json::from_slice(&fs::read(p).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}

#[tauri::command]
fn append_history(app:AppHandle,payload:Value)->Result<HistoryRecord,String>{
    let p=history_path(&app)?;let mut all:Vec<HistoryRecord>=if p.exists(){serde_json::from_slice(&fs::read(&p).map_err(|e|e.to_string())?).unwrap_or_default()}else{vec![]};
    let rec=HistoryRecord{id:now_id(),timestamp:now_id(),payload};all.insert(0,rec.clone());if all.len()>100{all.truncate(100);}
    fs::write(p,serde_json::to_vec_pretty(&all).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;Ok(rec)
}

#[tauri::command]
fn path_to_data_url(path:String)->Result<String,String>{
    let bytes=fs::read(&path).map_err(|e|e.to_string())?;
    let mime=match Path::new(&path).extension().and_then(|x|x.to_str()).unwrap_or("").to_lowercase().as_str(){"png"=>"image/png","jpg"|"jpeg"=>"image/jpeg","webp"=>"image/webp","gif"=>"image/gif",_=>"application/octet-stream"};
    Ok(format!("data:{};base64,{}",mime,base64::engine::general_purpose::STANDARD.encode(bytes)))
}


#[tauri::command]
async fn start_web_host(app:AppHandle,state:tauri::State<'_,AppState>,port:Option<u16>)->Result<Value,String>{
    let mut host=state.web_host.lock().await;
    if let Some(existing)=host.as_ref(){
        return Ok(json!({"running":true,"port":existing.port,"localUrl":existing.lan_url,"lanUrl":existing.lan_url}));
    }

    let chosen_port=port.unwrap_or(1424);
    let lan_host=lan_ip();
    let parsed_lan_host=lan_host.parse::<std::net::Ipv4Addr>().ok();
    let lan_only=parsed_lan_host.map(|ip| ip.is_private() || ip.is_link_local()).unwrap_or(false);
    if !lan_only{
        return Err("No private LAN IPv4 address was detected. Connect the computer to the same local network and try again.".into());
    }
    let listener=tokio::net::TcpListener::bind((lan_host.as_str(),chosen_port))
        .await
        .map_err(|e|format!("LAN host could not bind {}:{}: {}",lan_host,chosen_port,e))?;
    let actual_port=listener.local_addr().map_err(|e|e.to_string())?.port();
    let dist=dist_directory(&app).ok_or("Could not find dist/index.html. Run npm run build first.")?;
    let api_state=WebApiState{app:app.clone()};
    let router=Router::new()
        .route("/api/{command}",post(web_command))
        .fallback_service(ServeDir::new(dist))
        .with_state(api_state);
    let lan=format!("http://{}:{}",lan_host,actual_port);
    let task=tokio::spawn(async move{
        let _=axum::serve(listener,router).await;
    });
    let url=json!({"running":true,"port":actual_port,"localUrl":lan,"lanUrl":lan});
    *host=Some(WebHostRuntime{port:actual_port,lan_url:lan,task});
    Ok(url)
}

#[tauri::command]
async fn stop_web_host(state:tauri::State<'_,AppState>)->Result<(),String>{
    if let Some(runtime)=state.web_host.lock().await.take(){runtime.task.abort();}
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run(){
    tauri::Builder::default()
        .manage(AppState{active_stream:Arc::new(Mutex::new(false)),web_host:Arc::new(Mutex::new(None))})
        .invoke_handler(tauri::generate_handler![
            pick_folder,discover_raphael_config,discover_raphael_roots,scan_library,list_provider_models,
            prepare_generation,stream_llm,parse_prompt_pair,finalize_prompt_pair,build_workflow,inject_prompts,
            submit_to_comfy,monitor_comfy_generation,load_history,append_history,path_to_data_url,start_web_host,stop_web_host
        ])
        .run(tauri::generate_context!())
        .expect("error while running Raphael Prompt Forge");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_prompt_strips_generated_lora_syntax() {
        let loras=vec![SelectedLora{
            id:"x".into(), name:"Style".into(), path:"x".into(), weight:1.0,
            activation_tags:vec!["style_trigger".into()], tags:vec![],
            description:None, character:false, base_model:None,
        }];
        let result=finalize_positive_prompt(
            "anime character, [Style - Nekoya@4x0style]1.3, <lora:foo:1.2>, detailed face",
            &loras,
        );
        assert!(!result.contains("[Style - Nekoya@4x0style]1.3"));
        assert!(!result.contains("<lora:foo:1.2>"));
        assert!(result.contains("style_trigger"));
        assert!(result.contains("detailed face"));
    }

    #[test]
    fn inject_request_accepts_frontend_camel_case_payload() {
        let req: InjectRequest = serde_json::from_value(json!({
            "workflow": {
                "1": {"class_type":"CLIPTextEncode","inputs":{"text":"__POSITIVE_PROMPT__","clip":["0",0]}},
                "2": {"class_type":"CLIPTextEncode","inputs":{"text":"__NEGATIVE_PROMPT__","clip":["0",0]}}
            },
            "positivePrompt": "a detailed portrait",
            "negativePrompt": "blurry, low quality"
        })).expect("camelCase invoke payload must deserialize");

        let injected = inject_prompts(req).expect("inject_prompts must succeed");
        assert_eq!(
            injected["1"]["inputs"]["text"].as_str(),
            Some("a detailed portrait")
        );
        assert_eq!(
            injected["2"]["inputs"]["text"].as_str(),
            Some("blurry, low quality")
        );
    }

    #[test]
    fn inject_request_accepts_legacy_snake_case_payload() {
        let req: InjectRequest = serde_json::from_value(json!({
            "workflow": {"1":{"class_type":"CLIPTextEncode","inputs":{"text":"__POSITIVE_PROMPT__"}}},
            "positive_prompt": "portrait",
            "negative_prompt": "bad anatomy"
        })).expect("snake_case compatibility payload must deserialize");

        let injected = inject_prompts(req).expect("inject_prompts must succeed");
        assert_eq!(
            injected["1"]["inputs"]["text"].as_str(),
            Some("portrait")
        );
    }

    fn workflow_test_checkpoint() -> ModelInfo {
        ModelInfo {
            id:"cp".into(), name:"miaomiaoRealskin_anima13.safetensors".into(),
            kind:"checkpoint".into(),
            path:"D:/ComfyUI/models/unet/anima/miaomiaoRealskin_anima13.safetensors".into(),
            size:1, base_model:Some("anima".into()), tags:vec!["anima".into()],
            activation_tags:vec![], character:false, thumbnail:None,
            source:"test".into(), cache_name:None, cache_description:None,
        }
    }

    fn workflow_test_lora(id: &str, name: &str, weight: f32) -> SelectedLora {
        SelectedLora {
            id:id.into(), name:name.into(),
            path:format!("D:/ComfyUI/models/loras/Anima/{name}.safetensors"),
            weight, activation_tags:vec![], tags:vec!["anima".into()],
            description:None, character:false, base_model:Some("anima".into()),
        }
    }

    #[test]
    fn comfy_relative_model_name_preserves_nested_model_subfolders() {
        assert_eq!(
            comfy_relative_model_name(
                "D:/ComfyUI/models/unet/anima/chosenIrisesMix_v20Anima.safetensors",
                "unet",
                "fallback.safetensors"
            ),
            "anima\\chosenIrisesMix_v20Anima.safetensors"
        );
        assert_eq!(
            comfy_relative_model_name(
                "D:/ComfyUI/models/anima/chosenIrisesMix_v20Anima.safetensors",
                "unet",
                "fallback.safetensors"
            ),
            "anima\\chosenIrisesMix_v20Anima.safetensors"
        );
    }

    #[test]
    fn comfy_relative_model_name_strips_diffusion_models_prefix_for_unet() {
        assert_eq!(
            comfy_relative_model_name(
                "D:/ComfyUI/models/diffusion_models/anima/chosenIrisesMix_v20Anima.safetensors",
                "unet",
                "fallback.safetensors"
            ),
            "anima\\chosenIrisesMix_v20Anima.safetensors"
        );
    }

    #[test]
    fn reference_workflow_uses_no_lora_chain_when_none_selected() {
        let workflow = build_workflow(WorkflowRequest {
            checkpoint:workflow_test_checkpoint(),
            loras:vec![],
            width:1920, height:1080, steps:12, cfg:1.0,
            sampler:"euler".into(), seed:123,
        }).expect("reference workflow must build");

        assert_eq!(workflow["13"]["class_type"], "UNETLoader");
        assert_eq!(workflow["13"]["inputs"]["unet_name"], "anima\\miaomiaoRealskin_anima13.safetensors");
        assert_eq!(workflow["7"]["inputs"]["model"], json!(["13",0]));
        assert!(workflow.get("14").is_none());
        assert_eq!(workflow["10"]["inputs"]["vae"], json!(["9",0]));
        assert_eq!(workflow["12"]["inputs"]["images"], json!(["10",0]));
    }

    #[test]
    fn reference_workflow_adds_exactly_one_lora_node_per_selected_lora() {
        let loras = vec![
            workflow_test_lora("l1","Akane",1.0),
            workflow_test_lora("l2","turbo",0.8),
            workflow_test_lora("l3","third",0.6),
        ];
        let workflow = build_workflow(WorkflowRequest {
            checkpoint:workflow_test_checkpoint(),
            loras:loras,
            width:1920, height:1080, steps:12, cfg:1.0,
            sampler:"euler".into(), seed:123,
        }).expect("reference workflow must build");

        for (id, previous, lora_name, weight) in [
            ("14","13","Anima\\Akane.safetensors",1.0),
            ("15","14","Anima\\turbo.safetensors",0.8),
            ("16","15","Anima\\third.safetensors",0.6),
        ] {
            assert_eq!(workflow[id]["class_type"], "LoraLoaderModelOnly");
            assert_eq!(workflow[id]["inputs"]["model"], json!([previous,0]));
            assert_eq!(workflow[id]["inputs"]["lora_name"], lora_name);
            let actual_weight = workflow[id]["inputs"]["strength_model"]
                .as_f64()
                .expect("strength_model must be a JSON number");
            assert!(
                (actual_weight - weight as f64).abs() < 1e-6,
                "strength_model mismatch: actual={actual_weight}, expected={weight}"
            );
        }
        assert_eq!(workflow["7"]["inputs"]["model"], json!(["16",0]));
        assert!(workflow.get("17").is_none());
    }

    #[test]
    fn reference_workflow_has_no_numeric_node_references() {
        let workflow = build_workflow(WorkflowRequest {
            checkpoint:workflow_test_checkpoint(),
            loras:vec![workflow_test_lora("l1","Akane",1.0)],
            width:1920, height:1080, steps:12, cfg:1.0,
            sampler:"euler".into(), seed:123,
        }).expect("reference workflow must build");

        for node in workflow.as_object().unwrap().values() {
            if let Some(inputs) = node.get("inputs").and_then(Value::as_object) {
                for input in inputs.values() {
                    if let Some(link) = input.as_array() {
                        if link.len() >= 2 {
                            assert!(link[0].is_string(), "ComfyUI link must use string node id: {link:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn full_generation_pipeline_dry_run() {
        let checkpoint = ModelInfo {
            id:"cp1".into(), name:"Anima Base".into(), kind:"checkpoint".into(),
            path:"D:/ComfyUI/models/checkpoints/anima.safetensors".into(), size:1,
            base_model:Some("anima".into()), tags:vec!["anima".into()],
            activation_tags:vec![], character:false, thumbnail:None,
            source:"raphael-model-manager".into(), cache_name:None, cache_description:None,
        };
        let loras = vec![
            ModelInfo {
                id:"l1".into(), name:"Character Alice".into(), kind:"lora".into(),
                path:"D:/ComfyUI/models/loras/alice.safetensors".into(), size:1,
                base_model:Some("anima".into()), tags:vec!["anima".into(),"character".into(),"alice".into()],
                activation_tags:vec!["alice_trigger".into()], character:true, thumbnail:None,
                source:"raphael-model-manager".into(), cache_name:None,
                cache_description:Some("Character identity LoRA for Alice".into()),
            },
            ModelInfo {
                id:"l2".into(), name:"School Uniform".into(), kind:"lora".into(),
                path:"D:/ComfyUI/models/loras/uniform.safetensors".into(), size:1,
                base_model:Some("anima".into()), tags:vec!["anima".into(),"outfit".into()],
                activation_tags:vec!["school_uniform_trigger".into()], character:false, thumbnail:None,
                source:"raphael-model-manager".into(), cache_name:None,
                cache_description:Some("Japanese school uniform clothing".into()),
            },
        ];
        let prepared = prepare_generation(PrepareRequest {
            checkpoint:checkpoint.clone(), loras:loras,
            selected_lora_ids:vec!["l1".into(),"l2".into()],
            setting:"classroom".into(), pose:"standing".into(), expression:"smiling".into(),
            character:"".into(), dress:"".into(), composition:"three-quarter".into(),
            additional:"".into(), random_lora_min:2, random_lora_max:2,
        }).expect("prepare_generation dry run must succeed");
        assert_eq!(prepared.loras.len(), 2);

        let raw_pair = PromptPair {
            positive_prompt:"Alice in a classroom, smiling, school uniform, alice_trigger".into(),
            negative_prompt:"blurry, malformed hands".into(),
            rationale:None,
        };
        let finalized = finalize_prompt_pair(FinalizePromptRequest {
            prompt_pair:raw_pair,
            loras:prepared.loras.clone(),
        }).expect("finalize prompt dry run must succeed");
        assert!(finalized.positive_prompt.ends_with("alice_trigger, school_uniform_trigger"));

        let workflow = build_workflow(WorkflowRequest {
            checkpoint:prepared.checkpoint,
            loras:prepared.loras,
            width:1024, height:1024, steps:28, cfg:6.5,
            sampler:"euler".into(), seed:123,
        }).expect("workflow dry run must succeed");

        let injected = inject_prompts(InjectRequest {
            workflow,
            positivePrompt:None,
            positive_prompt:Some(finalized.positive_prompt.clone()),
            negativePrompt:None,
            negative_prompt:Some(finalized.negative_prompt.clone()),
        }).expect("inject_prompts dry run must succeed");

        let text_nodes:Vec<String> = injected.as_object().unwrap().values()
            .filter(|node| node["class_type"] == "CLIPTextEncode")
            .filter_map(|node| node["inputs"]["text"].as_str().map(str::to_string))
            .collect();
        assert!(text_nodes.iter().any(|x| x == &finalized.positive_prompt));
        assert!(text_nodes.iter().any(|x| x == &finalized.negative_prompt));
    }

    #[test]
    fn activation_prompts_are_deterministically_appended() {
        let pair = PromptPair {
            positive_prompt: "portrait, blue eyes, serene expression, triggerA".into(),
            negative_prompt: "blurry".into(),
            rationale: None,
        };
        let loras = vec![
            SelectedLora {
                id: "1".into(), name:"character".into(), path:"c.safetensors".into(),
                weight:0.8, activation_tags:vec!["triggerA".into(), "char_tag".into()],
                tags:vec!["character".into()], description:None, character:true,
                base_model:Some("anima".into())
            },
            SelectedLora {
                id: "2".into(), name:"style".into(), path:"s.safetensors".into(),
                weight:0.7, activation_tags:vec!["style_tag".into()],
                tags:vec!["style".into()], description:None, character:false,
                base_model:Some("anima".into())
            }
        ];
        let finalized = finalize_prompt_pair(FinalizePromptRequest {
            prompt_pair: pair,
            loras,
        }).expect("finalize_prompt_pair must succeed");

        assert_eq!(
            finalized.positive_prompt,
            "portrait, blue eyes, serene expression, triggerA, char_tag, style_tag"
        );
        assert_eq!(finalized.negative_prompt, "blurry");
    }
}
