use anyhow::{anyhow, Context, Result};
use base64::Engine;
use futures_util::StreamExt;
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Mutex;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: String, name: String, kind: String, path: String, size: u64,
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
struct LlmSettings { provider: String, base_url: String, api_key: String, model: String, temperature: f32, max_tokens: u32 }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SelectedLora {
    id: String, name: String, path: String, weight: f32,
    activation_tags: Vec<String>, character: bool, base_model: Option<String>,
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
    setting: String, pose: String, expression: String, character: String,
    dress: String, composition: String, additional: String,
    random_lora_min: u32, random_lora_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptPair { positive_prompt: String, negative_prompt: String, rationale: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmRequest { settings: LlmSettings, system_prompt: String, user_prompt: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkflowRequest {
    checkpoint: ModelInfo, loras: Vec<SelectedLora>, width: u32, height: u32,
    steps: u32, cfg: f32, sampler: String, seed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InjectRequest { workflow: Value, positive_prompt: String, negative_prompt: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubmitRequest { comfy_url: String, workflow: Value }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryRecord { id: String, timestamp: String, payload: Value }

#[derive(Clone)]
struct AppState { active_stream: Arc<Mutex<bool>> }

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

fn scan_disk(root: &Path) -> (Vec<ModelInfo>,Vec<ModelInfo>) {
    let mut cps=Vec::new(); let mut ls=Vec::new();
    if !root.exists() { return (cps,ls); }
    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() { continue; }
        let ext=entry.path().extension().and_then(|x|x.to_str()).unwrap_or("").to_lowercase();
        if !matches!(ext.as_str(),"safetensors"|"ckpt"|"pt"|"bin") { continue; }
        let rel=norm(&entry.path().strip_prefix(root).unwrap_or(entry.path()).to_string_lossy());
        let kind=if rel.contains("lora"){"lora"}else if rel.contains("checkpoint")||rel.contains("checkpoints"){"checkpoint"}else{continue};
        let size=entry.metadata().map(|m|m.len()).unwrap_or(0);
        let name=entry.path().file_stem().and_then(|x|x.to_str()).unwrap_or("model").to_string();
        let m=ModelInfo{id:"disk-".to_string()+&norm(&entry.path().to_string_lossy()),name,kind:kind.into(),
            path:entry.path().to_string_lossy().to_string(),size,base_model:None,tags:vec![],activation_tags:vec![],
            character:false,thumbnail:None,source:"comfyui".into(),cache_name:None,cache_description:None};
        if kind=="checkpoint"{cps.push(m)}else{ls.push(m)}
    }
    (cps,ls)
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

#[tauri::command]
fn discover_raphael_roots()->Vec<String>{
    let mut roots=Vec::new();
    for key in ["APPDATA","LOCALAPPDATA","USERPROFILE"]{
        if let Ok(base)=std::env::var(key){
            for rel in ["Raphael Model Manager","Raphael-Model-Manager","Raphael Model Registry","Raphael-Model-Registry"]{
                let p=PathBuf::from(&base).join(rel); if p.exists(){roots.push(p.to_string_lossy().to_string());}
            }
        }
    }
    for p in [r"D:\Raphael-Model-Manager",r"D:\Raphael-Model-Registry",r"C:\Raphael-Model-Manager",r"C:\Raphael-Model-Registry"]{
        if Path::new(p).exists(){roots.push(p.to_string());}
    }
    roots.sort(); roots.dedup(); roots
}

#[tauri::command]
fn scan_library(req:ScanRequest)->Result<LibrarySnapshot,String>{
    let (mut cps,mut ls)=scan_disk(Path::new(&req.comfy_root)); let mut roots=vec![req.comfy_root.clone()];
    if let Some(rr)=req.raphael_root.as_deref().filter(|x|!x.trim().is_empty()){
        let cache=scan_cache(Path::new(rr)); roots.push(rr.to_string());
        let cp:Vec<_>=cache.iter().filter(|x|x.kind=="checkpoint").cloned().collect();
        let lr:Vec<_>=cache.iter().filter(|x|x.kind=="lora").cloned().collect();
        merge(&mut cps,&cp); merge(&mut ls,&lr);
    }
    cps.sort_by_key(|x|x.name.to_lowercase()); ls.sort_by_key(|x|x.name.to_lowercase());
    let mut warnings=Vec::new();
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
    let chars:Vec<&ModelInfo>=compatible.iter().copied().filter(|l|l.character||l.tags.iter().any(|t|norm(t)=="character")).collect();
    if chars.is_empty(){return Err("No compatible character LoRA was found. Character identity is restricted to LoRAs marked with the character tag.".into());}
    let wanted=req.character.trim().to_lowercase();
    let character=if wanted.is_empty(){*chars.choose(&mut rand::rng()).unwrap()}else{*chars.iter().find(|l|l.name.to_lowercase().contains(&wanted)||l.tags.iter().any(|t|t.to_lowercase().contains(&wanted))).ok_or("Requested character does not match a compatible character LoRA.")?};
    let count=rand::rng().random_range(req.random_lora_min.max(1)..=req.random_lora_max.max(req.random_lora_min.max(1))) as usize;
    let mut pool:Vec<&ModelInfo>=compatible.into_iter().filter(|x|x.id!=character.id).collect(); pool.shuffle(&mut rand::rng());
    let mut chosen=vec![character]; chosen.extend(pool.into_iter().take(count.saturating_sub(1)));
    let loras=chosen.into_iter().map(|l|SelectedLora{id:l.id.clone(),name:l.name.clone(),path:l.path.clone(),weight:rand::rng().random_range(0.65..=1.0),activation_tags:if l.activation_tags.is_empty(){l.tags.clone()}else{l.activation_tags.clone()},character:l.character||l.tags.iter().any(|t|norm(t)=="character"),base_model:l.base_model.clone()}).collect();
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
async fn stream_llm(app:AppHandle,state:tauri::State<'_,AppState>,req:LlmRequest)->Result<(),String>{
    {let mut busy=state.active_stream.lock().await;if *busy{return Err("An LLM stream is already active.".into())}*busy=true;}
    let result=stream_inner(app.clone(),req).await; *state.active_stream.lock().await=false; result
}

async fn stream_inner(app:AppHandle,req:LlmRequest)->Result<(),String>{
    let client=reqwest::Client::new(); let base=base_url(&req.settings.base_url);
    let (url,body,ollama)=if req.settings.provider=="ollama"{
        (format!("{}/api/chat",base),json!({"model":req.settings.model,"stream":true,"messages":[{"role":"system","content":req.system_prompt},{"role":"user","content":req.user_prompt}],"options":{"temperature":req.settings.temperature,"num_predict":req.settings.max_tokens}}),true)
    }else{
        (if base.ends_with("/v1"){format!("{}/chat/completions",base)}else{format!("{}/v1/chat/completions",base)},json!({"model":req.settings.model,"stream":true,"temperature":req.settings.temperature,"max_tokens":req.settings.max_tokens,"messages":[{"role":"system","content":req.system_prompt},{"role":"user","content":req.user_prompt}]}),false)
    };
    let mut request=client.post(url).json(&body); if !ollama&&!req.settings.api_key.trim().is_empty(){request=request.bearer_auth(req.settings.api_key);}
    let response=request.send().await.map_err(|e|e.to_string())?; if !response.status().is_success(){return Err(format!("LLM returned HTTP {}",response.status()));}
    let mut stream=response.bytes_stream(); let mut buffer=String::new();
    while let Some(chunk)=stream.next().await{
        buffer.push_str(&String::from_utf8_lossy(&chunk.map_err(|e|e.to_string())?));
        while let Some(pos)=buffer.find('\n'){
            let line=buffer[..pos].trim_end_matches('\r').to_string(); buffer=buffer[pos+1..].to_string();
            if line.trim().is_empty(){continue}
            let data=if ollama{line.as_str()}else{line.strip_prefix("data: ").unwrap_or("")};
            if data.is_empty()||data=="[DONE]"{continue}
            let v:Value=match serde_json::from_str(data){Ok(x)=>x,Err(_)=>continue};
            let token=if ollama{v.get("message").and_then(|x|x.get("content")).and_then(|x|x.as_str()).unwrap_or("")}else{v.get("choices").and_then(|x|x.get(0)).and_then(|x|x.get("delta")).and_then(|x|x.get("content")).and_then(|x|x.as_str()).unwrap_or("")};
            if !token.is_empty(){let _=app.emit("llm:delta",json!({"text":token}));}
        }
    }
    let _=app.emit("llm:done",json!({"ok":true})); Ok(())
}

fn parse_json(raw:&str)->Result<Value>{
    let clean=raw.trim(); if let Ok(v)=serde_json::from_str(clean){return Ok(v)}
    let a=clean.find('{').ok_or_else(||anyhow!("No JSON object in LLM output"))?;
    let b=clean.rfind('}').ok_or_else(||anyhow!("Incomplete JSON object"))?;
    serde_json::from_str(&clean[a..=b]).context("Could not parse LLM JSON")
}

#[tauri::command]
fn parse_prompt_pair(raw:String)->Result<PromptPair,String>{
    let v=parse_json(&raw).map_err(|e|e.to_string())?;
    Ok(PromptPair{
        positive_prompt:v.get("positive_prompt").and_then(|x|x.as_str()).ok_or("Missing positive_prompt")?.to_string(),
        negative_prompt:v.get("negative_prompt").and_then(|x|x.as_str()).ok_or("Missing negative_prompt")?.to_string(),
        rationale:v.get("rationale").and_then(|x|x.as_str()).map(str::to_string),
    })
}

#[tauri::command]
fn build_workflow(req:WorkflowRequest)->Result<Value,String>{
    let mut map=serde_json::Map::new();
    map.insert("1".into(),json!({"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":Path::new(&req.checkpoint.path).file_name().and_then(|x|x.to_str()).unwrap_or(&req.checkpoint.name)}}));
    let mut last="1".to_string(); let mut id=2u32;
    for l in &req.loras{
        map.insert(id.to_string(),json!({"class_type":"LoraLoader","inputs":{"model":[last,0],"clip":[last,1],"lora_name":Path::new(&l.path).file_name().and_then(|x|x.to_str()).unwrap_or(&l.name),"strength_model":l.weight,"strength_clip":l.weight}}));
        last=id.to_string(); id+=1;
    }
    let pos=id; let neg=id+1; let latent=id+2; let sampler=id+3; let decode=id+4; let save=id+5;
    map.insert(pos.to_string(),json!({"class_type":"CLIPTextEncode","inputs":{"text":"__POSITIVE_PROMPT__","clip":[last,1]}}));
    map.insert(neg.to_string(),json!({"class_type":"CLIPTextEncode","inputs":{"text":"__NEGATIVE_PROMPT__","clip":[last,1]}}));
    map.insert(latent.to_string(),json!({"class_type":"EmptyLatentImage","inputs":{"width":req.width,"height":req.height,"batch_size":1}}));
    map.insert(sampler.to_string(),json!({"class_type":"KSampler","inputs":{"model":[last,0],"positive":[pos,0],"negative":[neg,0],"latent_image":[latent,0],"seed":req.seed,"steps":req.steps,"cfg":req.cfg,"sampler_name":req.sampler,"scheduler":"normal","denoise":1.0}}));
    map.insert(decode.to_string(),json!({"class_type":"VAEDecode","inputs":{"samples":[sampler,0],"vae":["1",2]}}));
    map.insert(save.to_string(),json!({"class_type":"SaveImage","inputs":{"images":[decode,0],"filename_prefix":"RaphaelPromptForge"}}));
    Ok(Value::Object(map))
}

#[tauri::command]
fn inject_prompts(req:InjectRequest)->Result<Value,String>{
    let mut w=req.workflow;
    if let Some(map)=w.as_object_mut(){for node in map.values_mut(){if node.get("class_type").and_then(|x|x.as_str())==Some("CLIPTextEncode"){if let Some(inputs)=node.get_mut("inputs").and_then(|x|x.as_object_mut()){if inputs.get("text").and_then(|x|x.as_str())==Some("__POSITIVE_PROMPT__"){inputs.insert("text".into(),Value::String(req.positive_prompt.clone()));}if inputs.get("text").and_then(|x|x.as_str())==Some("__NEGATIVE_PROMPT__"){inputs.insert("text".into(),Value::String(req.negative_prompt.clone()));}}}}}
    Ok(w)
}

#[tauri::command]
async fn submit_to_comfy(req:SubmitRequest)->Result<Value,String>{
    let response=reqwest::Client::new().post(format!("{}/prompt",base_url(&req.comfy_url))).json(&json!({"prompt":req.workflow,"client_id":"raphael-prompt-forge"})).send().await.map_err(|e|e.to_string())?;
    let status=response.status(); let text=response.text().await.unwrap_or_default();
    if !status.is_success(){return Err(format!("ComfyUI returned HTTP {}: {}",status,text))}
    serde_json::from_str(&text).map_err(|e|format!("Invalid ComfyUI response: {}",e))
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run(){
    tauri::Builder::default()
        .manage(AppState{active_stream:Arc::new(Mutex::new(false))})
        .invoke_handler(tauri::generate_handler![
            pick_folder,discover_raphael_roots,scan_library,list_provider_models,
            prepare_generation,stream_llm,parse_prompt_pair,build_workflow,inject_prompts,
            submit_to_comfy,load_history,append_history,path_to_data_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running Raphael Prompt Forge");
}
