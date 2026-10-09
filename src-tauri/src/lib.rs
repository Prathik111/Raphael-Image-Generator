use anyhow::{anyhow, Result};
use axum::{
    extract::{DefaultBodyLimit, Json as AxumJson, Path as AxumPath, State as AxumState},
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
use std::{fs, path::{Path, PathBuf}, process::Command, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use tauri::{AppHandle, Manager};
use tauri::ipc::Channel;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_stream::{wrappers::UnboundedReceiverStream, Stream};
use tower_http::services::ServeDir;
use local_ip_address::local_ip;

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
    #[serde(default)] description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LibrarySnapshot { checkpoints: Vec<ModelInfo>, loras: Vec<ModelInfo>, source_roots: Vec<String>, warnings: Vec<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScanRequest { comfy_root: String, #[serde(default)] registry_url: Option<String> }

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
    #[serde(default = "default_max_character_loras")]
    max_character_loras: u32,
    #[serde(default)] registry_url: Option<String>,
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
    llm: Arc<Mutex<LlmSettings>>,
    generation_settings: Arc<Mutex<Value>>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct AppState {
    active_stream: Arc<Mutex<bool>>,
    cancel_generation: Arc<AtomicBool>,
    web_host: Arc<Mutex<Option<WebHostRuntime>>>,
}

#[derive(Clone)]
struct WebApiState {
    app: AppHandle,
    llm: Arc<Mutex<LlmSettings>>,
    generation_settings: Arc<Mutex<Value>>,
    cancel_generation: Arc<AtomicBool>,
}

fn norm(s: &str) -> String { s.trim().to_lowercase().replace([' ', '_', '-', '.', '/'], "") }
fn base_url(s: &str) -> String { s.trim().trim_end_matches('/').to_string() }

fn public_generation_settings(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut public_map = map.clone();
            public_map.remove("llm");
            Value::Object(public_map)
        }
        _ => json!({}),
    }
}

fn web_client_path_is_allowed(path: &str) -> bool {
    path.trim_start().starts_with("registry://")
}

fn is_character_lora_for_checkpoint(
    lora: &ModelInfo,
    checkpoint: &ModelInfo,
) -> bool {
    let character_tagged = lora.character
        || lora.tags.iter().any(|tag| norm(tag) == "character");
    if !character_tagged {
        return false;
    }

    let base_keys: Vec<String> = std::iter::once(checkpoint.base_model.as_deref().unwrap_or(""))
        .chain(checkpoint.tags.iter().map(String::as_str))
        .map(norm)
        .filter(|value| value.len() > 2)
        .collect();

    if base_keys.is_empty() {
        return true;
    }

    let lora_keys: Vec<String> = std::iter::once(lora.base_model.as_deref().unwrap_or(""))
        .chain(lora.tags.iter().map(String::as_str))
        .map(norm)
        .filter(|value| value.len() > 2)
        .collect();

    lora_keys.iter().any(|lora_key| {
        base_keys.iter().any(|base_key| {
            lora_key == base_key
                || lora_key.contains(base_key)
                || base_key.contains(lora_key)
        })
    })
}

#[derive(Debug, Clone, Deserialize)]
struct RegistrySearchResult {
    items: Vec<RegistryModel>,
    total: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryModel {
    id: String,
    name: String,
    model_type: String,
    description: Option<String>,
    base_model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryVersion {
    id: String,
    base_model: Option<String>,
    #[serde(default)]
    activation_prompts: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryFile {
    id: String,
    version_id: Option<String>,
    path: String,
    relative_path: Option<String>,
    filename: String,
    size_bytes: i64,
    status: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryAsset {
    id: String,
    kind: String,
}

fn registry_base_url(override_url: Option<&str>) -> String {
    if let Some(value) = override_url.map(str::trim).filter(|value| !value.is_empty()) {
        return value.trim_end_matches('/').to_string();
    }
    std::env::var("RAPHAEL_REGISTRY_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "http://127.0.0.1:43217".into())
}

fn registry_data_dir() -> PathBuf {
    if let Some(value) = std::env::var_os("RAPHAEL_REGISTRY_DATA_DIR") {
        return PathBuf::from(value);
    }
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("Raphael")
            .join("ModelRegistry")
            .join("data");
    }
    directories::ProjectDirs::from("com", "Raphael", "ModelRegistry")
        .map(|dirs| dirs.data_dir().join("data"))
        .unwrap_or_else(|| PathBuf::from(".raphael-model-registry").join("data"))
}

fn registry_token_path(data_dir: &Path) -> PathBuf {
    std::env::var_os("RAPHAEL_REGISTRY_TOKEN_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("registry.token"))
}

fn registry_token(data_dir: &Path) -> Result<String, String> {
    if let Ok(value) = std::env::var("RAPHAEL_REGISTRY_AUTH_TOKEN") {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Ok(value);
        }
    }

    let path = registry_token_path(data_dir);
    let value = fs::read_to_string(&path)
        .map_err(|error| format!("Raphael Model Registry token could not be read from {}: {}", path.display(), error))?
        .trim()
        .to_string();
    if value.is_empty() {
        return Err(format!("Raphael Model Registry token is empty: {}", path.display()));
    }
    Ok(value)
}

fn registry_executable_candidates() -> Vec<PathBuf> {
    let executable_name = if cfg!(windows) { "raphael-registry.exe" } else { "raphael-registry" };
    let mut candidates = Vec::new();

    if let Some(value) = std::env::var_os("RAPHAEL_REGISTRY_EXECUTABLE") {
        candidates.push(PathBuf::from(value));
    }

    if let Ok(current_exe) = std::env::current_exe() {
        let mut ancestor = current_exe.parent().map(Path::to_path_buf);
        for _ in 0..5 {
            let Some(dir) = ancestor else { break };
            candidates.push(dir.join(executable_name));
            candidates.push(dir.join("resources").join(executable_name));
            candidates.push(dir.join("registry").join(executable_name));
            candidates.push(dir.join("bin").join(executable_name));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("Raphael-Model-Registry").join("target").join("debug").join(executable_name));
                candidates.push(parent.join("Raphael-Model-Registry").join("target").join("release").join(executable_name));
            }
            ancestor = dir.parent().map(Path::to_path_buf);
        }
    }

    if let Some(path_var) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path_var) {
            candidates.push(directory.join(executable_name));
        }
    }

    candidates
}

fn registry_executable() -> Result<PathBuf, String> {
    registry_executable_candidates()
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "Raphael Model Registry is not running and its server executable was not found. Set RAPHAEL_REGISTRY_EXECUTABLE to the full path of raphael-registry{}.",
                if cfg!(windows) { ".exe" } else { "" }
            )
        })
}

async fn registry_health(base_url: &str) -> bool {
    reqwest::Client::new()
        .get(format!("{base_url}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

async fn ensure_registry(override_url: Option<&str>) -> Result<(String, String), String> {
    let base_url = registry_base_url(override_url);
    let data_dir = registry_data_dir();
    let token_file = registry_token_path(&data_dir);

    if !registry_health(&base_url).await {
        let executable = registry_executable()?;
        let mut command = Command::new(&executable);
        command
            .arg("server")
            .env("RAPHAEL_REGISTRY_DATA_DIR", &data_dir);

        if let Ok(value) = std::env::var("RAPHAEL_REGISTRY_AUTH_TOKEN") {
            if !value.trim().is_empty() {
                command.env("RAPHAEL_REGISTRY_AUTH_TOKEN", value.trim());
            }
        } else if let Ok(value) = fs::read_to_string(&token_file) {
            let value = value.trim();
            if !value.is_empty() {
                command.env("RAPHAEL_REGISTRY_AUTH_TOKEN", value);
            }
        }

        command
            .spawn()
            .map_err(|error| format!("Could not start Raphael Model Registry from {}: {}", executable.display(), error))?;

        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if registry_health(&base_url).await {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
        })
        .await
        .map_err(|_| "Raphael Model Registry did not become healthy within 10 seconds.".to_string())?;
    }

    let token = registry_token(&data_dir)?;
    Ok((base_url, token))
}

async fn registry_json<T: serde::de::DeserializeOwned>(
    base_url: &str,
    token: &str,
    path: &str,
) -> Result<T, String> {
    let response = reqwest::Client::new()
        .get(format!("{base_url}{path}"))
        .bearer_auth(token)
        .header("x-raphael-actor", "image-generator")
        .send()
        .await
        .map_err(|error| format!("Registry request failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Registry returned HTTP {status}: {body}"));
    }
    response.json::<T>().await.map_err(|error| format!("Invalid Registry response: {error}"))
}

async fn registry_models(base_url: &str, token: &str, endpoint: &str) -> Result<Vec<RegistryModel>, String> {
    let mut all = Vec::new();
    let mut offset = 0_i64;
    loop {
        let response: RegistrySearchResult =
            registry_json(base_url, token, &format!("{endpoint}?limit=200&offset={offset}")).await?;
        let count = response.items.len() as i64;
        all.extend(response.items);
        if count == 0 || offset + count >= response.total {
            break;
        }
        offset += count;
    }
    Ok(all)
}

fn registry_model_file_path(file: &RegistryFile, comfy_root: Option<&Path>) -> Option<PathBuf> {
    // Registry is authoritative for the model's registered path. Prefer a real
    // host file when it exists, but never discard a Registry model merely
    // because its file cannot currently be resolved from the guessed root.
    let root = comfy_root
        .filter(|path| path.is_dir())
        .and_then(|path| path.canonicalize().ok());

    let registry_path = PathBuf::from(&file.path);
    if registry_path.is_absolute() {
        if registry_path.is_file() {
            return Some(registry_path.canonicalize().unwrap_or(registry_path));
        }
        return Some(registry_path);
    }

    if let Some(root) = root {
        let mut candidates = Vec::new();
        if !file.path.is_empty() {
            candidates.push(root.join(&file.path));
        }
        if let Some(relative) = &file.relative_path {
            candidates.push(root.join(relative));
            let normalized = relative.replace('\\', "/");
            if let Some(stripped) = normalized.strip_prefix("models/") {
                candidates.push(root.join(stripped));
            }
        }
        candidates.push(root.join(&file.filename));

        for candidate in candidates {
            if candidate.is_file() {
                let canonical = candidate.canonicalize().unwrap_or(candidate);
                if canonical.starts_with(&root) {
                    return Some(canonical);
                }
            }
        }

        // Preserve a usable host-relative path for Registry-only metadata.
        if let Some(relative) = file.relative_path.as_deref().filter(|value| !value.trim().is_empty()) {
            return Some(root.join(relative));
        }
        if !file.path.trim().is_empty() {
            return Some(root.join(&file.path));
        }
        if !file.filename.trim().is_empty() {
            return Some(root.join(&file.filename));
        }
    }

    if !file.path.trim().is_empty() {
        return Some(PathBuf::from(&file.path));
    }
    file.relative_path
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| (!file.filename.trim().is_empty()).then(|| PathBuf::from(&file.filename)))
}

async fn hydrate_registry_model(
    model: RegistryModel,
    base_url: &str,
    token: &str,
    comfy_root: Option<&Path>,
) -> Result<Vec<ModelInfo>, String> {
    let encoded = urlencoding::encode(&model.id);
    let versions_path = format!("/api/v1/models/{encoded}/versions");
    let files_path = format!("/api/v1/models/{encoded}/files");
    let tags_path = format!("/api/v1/models/{encoded}/tags");
    let assets_path = format!("/api/v1/models/{encoded}/assets");
    let versions_future = registry_json::<Vec<RegistryVersion>>(base_url, token, &versions_path);
    let files_future = registry_json::<Vec<RegistryFile>>(base_url, token, &files_path);
    let tags_future = registry_json::<Vec<String>>(base_url, token, &tags_path);
    let assets_future = registry_json::<Vec<RegistryAsset>>(base_url, token, &assets_path);
    let (versions, files, tags, assets) = tokio::join!(versions_future, files_future, tags_future, assets_future);
    let versions = versions.unwrap_or_default();
    let files = files.unwrap_or_default();
    let tags = tags.unwrap_or_default();
    let assets = assets.unwrap_or_default();

    // A Registry model is a metadata container and may own multiple installed
    // files/versions. The Image Generator must represent each available file
    // separately; otherwise two installed variants with the same display name
    // collapse into one selectable model.
    let available_files: Vec<&RegistryFile> = files
        .iter()
        .filter(|file| file.status.eq_ignore_ascii_case("available"))
        .collect();

    // Model records with no remaining Registry files are intentionally hidden.
    // Model Manager deletes files first and may leave the metadata container
    // behind, so treating an empty model as a real image model would resurrect
    // deleted entries using the model name as a fake path.
    if available_files.is_empty() {
        return Ok(Vec::new());
    }

    let latest_version = versions.first();

    let mut thumbnail_candidates = Vec::new();
    for kind in ["thumbnail", "cover", "preview", "gallery"] {
        thumbnail_candidates.extend(
            assets.iter()
                .filter(|asset| asset.kind.eq_ignore_ascii_case(kind))
                .map(|asset| format!("registry://{}/{}", model.id, asset.id)),
        );
    }
    let thumbnail = (!thumbnail_candidates.is_empty())
        .then(|| thumbnail_candidates.join("|"));

    let mut result = Vec::with_capacity(available_files.len());

    for file in available_files {
        let file_path = match registry_model_file_path(file, comfy_root) {
            Some(path) => path,
            None => continue,
        };

        let selected_version = file
            .version_id
            .as_deref()
            .and_then(|version_id| versions.iter().find(|version| version.id == version_id))
            .or(latest_version);

        let base_model = selected_version
            .and_then(|version| version.base_model.clone())
            .or_else(|| model.base_model.clone());

        let activation_tags = selected_version
            .map(|version| version.activation_prompts.clone())
            .unwrap_or_default();

        result.push(ModelInfo {
            // File identity, not display name, is the selectable identity.
            // Include the Registry model ID so the ID remains globally tied to
            // its canonical metadata record while still distinguishing files.
            id: format!("{}::file:{}", model.id, file.id),
            name: model.name.clone(),
            kind: model.model_type.clone(),
            path: file_path.to_string_lossy().to_string(),
            size: file.size_bytes.max(0) as u64,
            base_model,
            tags: tags.clone(),
            activation_tags,
            character: tags.iter().any(|tag| norm(tag) == "character"),
            thumbnail: thumbnail.clone(),
            source: "raphael-registry".into(),
            description: model.description.clone(),
        });
    }

    Ok(result)
}

async fn scan_registry_library(req: &ScanRequest) -> Result<LibrarySnapshot, String> {
    let root_path = Path::new(&req.comfy_root);
    let root = root_path.is_dir().then_some(root_path);

    let (base_url, token) = ensure_registry(req.registry_url.as_deref()).await?;

    // Prefer the Registry's typed, server-side paginated endpoints, but
    // also read the canonical model catalog once. This is a defensive
    // compatibility path for older Registry databases whose model_type
    // casing may not match the typed SQL predicate exactly.
    let (checkpoint_result, lora_result, canonical_result) = tokio::join!(
        registry_models(&base_url, &token, "/api/v1/checkpoints"),
        registry_models(&base_url, &token, "/api/v1/loras"),
        registry_models(&base_url, &token, "/api/v1/models"),
    );

    let mut used_canonical_fallback = false;
    let (mut checkpoint_models, mut lora_models): (Vec<RegistryModel>, Vec<RegistryModel>) =
        match (checkpoint_result, lora_result) {
            (Ok(checkpoints), Ok(loras)) => (checkpoints, loras),
            _ => {
                used_canonical_fallback = true;
                let all_models = canonical_result.clone()
                    .map_err(|error| error)?;
                let checkpoints = all_models
                    .iter()
                    .filter(|model| model.model_type.eq_ignore_ascii_case("checkpoint"))
                    .cloned()
                    .collect();
                let loras = all_models
                    .iter()
                    .filter(|model| model.model_type.eq_ignore_ascii_case("lora"))
                    .cloned()
                    .collect();
                (checkpoints, loras)
            }
        };

    if let Ok(all_models) = canonical_result {
        for model in all_models {
            if model.model_type.eq_ignore_ascii_case("checkpoint")
                && !checkpoint_models.iter().any(|item| item.id == model.id)
            {
                checkpoint_models.push(model);
            } else if model.model_type.eq_ignore_ascii_case("lora")
                && !lora_models.iter().any(|item| item.id == model.id)
            {
                lora_models.push(model);
            }
        }
    }

    let mut checkpoints = Vec::new();
    let mut loras = Vec::new();
    let mut hydration_failures = 0usize;

    for chunk in checkpoint_models.chunks(16) {
        let hydrated = futures_util::stream::iter(chunk.iter().cloned())
            .map(|model| {
                let root = root.map(Path::to_path_buf);
                let base_url = base_url.clone();
                let token = token.clone();
                async move {
                    hydrate_registry_model(model, &base_url, &token, root.as_deref()).await
                }
            })
            .buffer_unordered(16);
        tokio::pin!(hydrated);
        while let Some(result) = hydrated.next().await {
            match result {
                Ok(models) => checkpoints.extend(models),
                Err(_) => hydration_failures += 1,
            }
        }
    }

    for chunk in lora_models.chunks(16) {
        let hydrated = futures_util::stream::iter(chunk.iter().cloned())
            .map(|model| {
                let root = root.map(Path::to_path_buf);
                let base_url = base_url.clone();
                let token = token.clone();
                async move {
                    hydrate_registry_model(model, &base_url, &token, root.as_deref()).await
                }
            })
            .buffer_unordered(16);
        tokio::pin!(hydrated);
        while let Some(result) = hydrated.next().await {
            match result {
                Ok(models) => loras.extend(models),
                Err(_) => hydration_failures += 1,
            }
        }
    }

    checkpoints.sort_by(|a, b| {
        a.name.to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.base_model.clone().unwrap_or_default().to_lowercase().cmp(&b.base_model.clone().unwrap_or_default().to_lowercase()))
            .then_with(|| a.path.to_lowercase().cmp(&b.path.to_lowercase()))
    });
    loras.sort_by(|a, b| {
        a.name.to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.base_model.clone().unwrap_or_default().to_lowercase().cmp(&b.base_model.clone().unwrap_or_default().to_lowercase()))
            .then_with(|| a.path.to_lowercase().cmp(&b.path.to_lowercase()))
    });

    let mut warnings = Vec::new();
    if checkpoint_models.is_empty() {
        warnings.push("The Raphael Model Registry returned no checkpoint records.".into());
    }
    if lora_models.is_empty() {
        warnings.push("The Raphael Model Registry returned no LoRA records.".into());
    }
    if used_canonical_fallback {
        warnings.push(
            "The Registry checkpoint/LoRA routes were unavailable; the canonical model catalog fallback was used.".into()
        );
    }
    if hydration_failures > 0 {
        warnings.push(format!(
            "{} Registry model records could not be fully hydrated; available catalog records remain visible.",
            hydration_failures
        ));
    }

    let mut source_roots = vec![base_url];
    if let Some(root) = root {
        source_roots.insert(0, root.to_string_lossy().to_string());
    }

    Ok(LibrarySnapshot {
        checkpoints,
        loras,
        source_roots,
        warnings,
    })
}


#[tauri::command]
fn pick_folder()->Result<PickResult,String>{
    Ok(PickResult{path:rfd::FileDialog::new().pick_folder().map(|x|x.to_string_lossy().to_string())})
}

#[derive(Debug, Serialize)]
struct RaphaelConfig {
    models_root: Option<String>,
    registry_url: Option<String>,
}

#[tauri::command]
fn discover_raphael_config() -> RaphaelConfig {
    let models_root = std::env::var("RAPHAEL_COMFY_MODELS_ROOT").ok();
    let registry_url = Some(registry_base_url(None));
    RaphaelConfig { models_root, registry_url }
}

#[tauri::command]
fn discover_raphael_roots() -> Vec<String> {
    vec![registry_base_url(None)]
}

#[tauri::command]
async fn scan_library(req: ScanRequest) -> Result<LibrarySnapshot, String> {
    scan_registry_library(&req).await
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

fn default_max_character_loras() -> u32 { 1 }

fn registry_model_id(id: &str) -> &str {
    id.split_once("::file:")
        .map(|(model_id, _)| model_id)
        .unwrap_or(id)
}

#[tauri::command]
async fn prepare_generation(req: PrepareRequest) -> Result<PreparedGeneration, String> {
    let (registry_url, token) = ensure_registry(req.registry_url.as_deref()).await?;

    let keys: Vec<String> = req.checkpoint.tags.iter()
        .chain(req.checkpoint.base_model.iter())
        .chain(std::iter::once(&req.checkpoint.name))
        .map(|value| norm(value))
        .filter(|value| value.len() > 2)
        .collect();

    let checkpoint_registry_id = registry_model_id(&req.checkpoint.id).to_string();
    let mut compatible_ids = std::collections::HashSet::new();

    if req.selected_lora_ids.is_empty() {
        let encoded_checkpoint = urlencoding::encode(&checkpoint_registry_id);
        let result: serde_json::Value = registry_json(
            &registry_url, &token,
            &format!("/api/v1/models/{encoded_checkpoint}/compatibility?type=lora")
        ).await?;

        if let Some(candidates) = result.get("candidates").and_then(|value| value.as_array()) {
            for candidate in candidates {
                if let Some(id) = candidate.get("id").and_then(|value| value.as_str()) {
                    compatible_ids.insert(registry_model_id(id).to_string());
                }
            }
        }
    } else {
        for lora_id in &req.selected_lora_ids {
            let checkpoint = urlencoding::encode(&checkpoint_registry_id);
            let lora_registry_id = registry_model_id(lora_id);
            let lora = urlencoding::encode(lora_registry_id);
            let result: serde_json::Value = registry_json(
                &registry_url, &token,
                &format!("/api/v1/compatibility?checkpoint={checkpoint}&lora={lora}")
            ).await?;
            if result.get("compatible").and_then(|value| value.as_bool()) == Some(true) {
                compatible_ids.insert(lora_registry_id.to_string());
            } else {
                return Err(format!("LoRA '{}' is not compatible with checkpoint '{}'.", lora_id, req.checkpoint.name));
            }
        }
    }

    let compatible: Vec<&ModelInfo> = req.loras.iter()
        .filter(|lora| compatible_ids.contains(registry_model_id(&lora.id)))
        .collect();

    if compatible.is_empty() {
        return Err("No compatible LoRAs were found in the Raphael Model Registry for the selected checkpoint.".into());
    }

    let manual: Vec<&ModelInfo> = req.selected_lora_ids.iter()
        .filter_map(|id| compatible.iter().copied().find(|lora| lora.id == *id))
        .collect();

    let chars: Vec<&ModelInfo> = compatible.iter().copied()
        .filter(|lora| is_character_lora_for_checkpoint(lora, &req.checkpoint))
        .collect();

    if chars.is_empty() {
        return Err(
            "No compatible character LoRA was found for the selected checkpoint base model. Random LoRA selection requires at least one Registry character LoRA matching the checkpoint base-model tag."
                .into()
        );
    }

    let max_character_loras=req.max_character_loras.max(1) as usize;
    let manual_character_count=manual.iter().filter(|lora| is_character_lora_for_checkpoint(lora, &req.checkpoint)).count();
    if manual_character_count>max_character_loras {
        return Err(format!(
            "Selected LoRA stack contains {manual_character_count} character LoRAs, exceeding the configured maximum of {max_character_loras}."
        ));
    }

    let wanted = req.character.trim().to_lowercase();
    let requested_character = if !wanted.is_empty() {
        Some(*chars.iter().find(|lora| {
            lora.name.to_lowercase().contains(&wanted)
                || lora.tags.iter().any(|tag| tag.to_lowercase().contains(&wanted))
        }).ok_or("Requested character does not match a compatible Registry character LoRA.")?)
    } else {
        None
    };

    let manual_matching_character = requested_character.and_then(|wanted_lora| {
        manual.iter().copied().find(|lora| lora.id == wanted_lora.id)
    });
    let manual_character=manual.iter().copied()
        .find(|lora| is_character_lora_for_checkpoint(lora, &req.checkpoint));

    let character = requested_character
        .or(manual_matching_character)
        .or(manual_character)
        .unwrap_or_else(|| *chars.choose(&mut rand::rng()).unwrap());

    let chosen: Vec<&ModelInfo> = if !manual.is_empty() {
        let mut picked = manual.clone();
        if let Some(requested)=requested_character {
            if !picked.iter().any(|lora| lora.id == requested.id) {
                if manual_character_count>=max_character_loras {
                    return Err(format!(
                        "The requested character LoRA cannot be added because the selected stack already contains the configured maximum of {max_character_loras} character LoRAs."
                    ));
                }
                picked.insert(0, requested);
            }
        } else if manual_character.is_none() {
            if manual_character_count>=max_character_loras {
                return Err(format!(
                    "A character LoRA is required, but the selected stack already uses the maximum of {max_character_loras} character LoRAs."
                ));
            }
            picked.insert(0, character);
        }
        picked
    } else {
        let min_count = req.random_lora_min.max(1);
        let max_count = req.random_lora_max.max(min_count);
        let count = rand::rng().random_range(min_count..=max_count) as usize;
        let mut pool: Vec<&ModelInfo> = compatible.into_iter().filter(|lora| lora.id != character.id).collect();
        pool.shuffle(&mut rand::rng());
        let mut picked = vec![character];
        let mut character_count=1usize;
        for lora in pool {
            if picked.len()>=count { break; }
            let is_character=is_character_lora_for_checkpoint(lora, &req.checkpoint);
            if is_character && character_count>=max_character_loras { continue; }
            picked.push(lora);
            if is_character { character_count+=1; }
        }
        picked
    };

    let loras = chosen.into_iter().map(|lora| SelectedLora {
        id: lora.id.clone(),
        name: lora.name.clone(),
        path: lora.path.clone(),
        weight: rand::rng().random_range(0.65..=1.0),
        activation_tags: lora.activation_tags.clone(),
        tags: lora.tags.clone(),
        description: lora.description.clone(),
        character: lora.character || lora.tags.iter().any(|tag| norm(tag) == "character"),
        base_model: lora.base_model.clone(),
    }).collect();

    // Leave unspecified scene fields empty so the planner, not preparation-time
    // independent randomization, chooses a coherent scene around one creative anchor.
    let scene = SceneSelection {
        setting: req.setting.trim().into(),
        pose: req.pose.trim().into(),
        expression: req.expression.trim().into(),
        character: character.name.clone(),
        dress: req.dress.trim().into(),
        composition: req.composition.trim().into(),
    };

    Ok(PreparedGeneration { checkpoint: req.checkpoint, loras, scene, compatibility_keys: keys })
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
    let cancel_generation=state.cancel_generation.clone();
    let result=stream_llm_inner(req, cancel_generation, |delta| {
        on_event.send(delta).map_err(|e|format!("LLM stream channel closed: {}",e))
    }).await;
    *state.active_stream.lock().await=false;
    result
}

async fn stream_llm_inner<F>(req:LlmRequest, cancel_generation:Arc<AtomicBool>, mut emit:F)->Result<(),String>
where F:FnMut(LlmDelta)->Result<(),String> + Send
{
    if cancel_generation.load(Ordering::SeqCst) {
        return Err("Generation stopped.".into());
    }
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
        if cancel_generation.load(Ordering::SeqCst) {
            return Err("Generation stopped.".into());
        }
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

fn finalize_positive_prompt(raw:&str)->String{
    strip_generated_lora_syntax(raw)
        .trim()
        .trim_matches(',')
        .trim()
        .split(',')
        .map(str::trim)
        .filter(|x|!x.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

#[tauri::command]
fn finalize_prompt_pair(req:FinalizePromptRequest)->Result<PromptPair,String>{
    Ok(PromptPair{
        positive_prompt:finalize_positive_prompt(&req.prompt_pair.positive_prompt),
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


fn checkpoint_path_uses_unet_loader(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_lowercase();
    normalized.contains("models/unet/")
        || normalized.contains("models/diffusion_models/")
}

#[tauri::command]
fn build_workflow(req:WorkflowRequest)->Result<Value,String>{
    // Select the loader from the model's actual ComfyUI location.
    // Files under models/unet or models/diffusion_models are UNETLoader
    // inputs; files under models/checkpoints are full CheckpointLoaderSimple
    // inputs. This prevents a checkpoint-path model from being submitted to
    // UNETLoader, which ComfyUI correctly rejects during validation.
    let mut map=serde_json::Map::new();

    let use_unet_loader = checkpoint_path_uses_unet_loader(&req.checkpoint.path);

    let model_input = if use_unet_loader {
        let unet_name = comfy_relative_model_name(
            &req.checkpoint.path,
            "unet",
            Path::new(&req.checkpoint.path)
                .file_name()
                .and_then(|x|x.to_str())
                .unwrap_or(&req.checkpoint.name),
        );
        json!({
            "class_type":"UNETLoader",
            "inputs":{"unet_name":unet_name,"weight_dtype":"default"}
        })
    } else {
        let checkpoint_name = comfy_relative_model_name(
            &req.checkpoint.path,
            "checkpoints",
            Path::new(&req.checkpoint.path)
                .file_name()
                .and_then(|x|x.to_str())
                .unwrap_or(&req.checkpoint.name),
        );
        json!({
            "class_type":"CheckpointLoaderSimple",
            "inputs":{"ckpt_name":checkpoint_name}
        })
    };
    map.insert("13".into(),model_input);

    // Some Anima checkpoint files are packaged without an embedded CLIP/text
    // encoder or VAE. Always use the known-good external Anima encoders rather
    // than trusting CheckpointLoaderSimple's optional outputs.
    map.insert("4".into(),json!({
        "class_type":"CLIPLoader",
        "inputs":{"clip_name":"anima\\oneObsession_anima29BV1_txt.safetensors","type":"stable_diffusion","device":"default"}
    }));

    map.insert("9".into(),json!({
        "class_type":"VAELoader",
        "inputs":{"vae_name":"anima\\qwen_image_vae.safetensors"}
    }));

    let clip_ref = json!(["4",0]);
    let vae_ref = json!(["9",0]);

    map.insert("5".into(),json!({
        "class_type":"CLIPTextEncode",
        "inputs":{"clip":clip_ref.clone(),"text":"__POSITIVE_PROMPT__"}
    }));
    map.insert("6".into(),json!({
        "class_type":"CLIPTextEncode",
        "inputs":{"clip":clip_ref,"text":"__NEGATIVE_PROMPT__"}
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
        "inputs":{"samples":["7",0],"vae":vae_ref}
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

async fn interrupt_comfy(comfy_url:&str)->Result<(),String>{
    let base=base_url(comfy_url);
    let response=reqwest::Client::new()
        .post(format!("{}/interrupt",base))
        .send().await
        .map_err(|e|e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("ComfyUI interrupt returned HTTP {}",response.status()));
    }
    Ok(())
}

#[tauri::command]
async fn start_generation(
    state:tauri::State<'_,AppState>,
)->Result<(),String>{
    state.cancel_generation.store(false, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
async fn stop_generation(
    state:tauri::State<'_,AppState>,
    comfy_url:Option<String>,
)->Result<(),String>{
    state.cancel_generation.store(true, Ordering::SeqCst);
    if let Some(url)=comfy_url {
        if !url.trim().is_empty() {
            let _=interrupt_comfy(&url).await;
        }
    }
    Ok(())
}

async fn submit_to_comfy_inner(
    req:SubmitRequest,
    cancel_generation:Arc<AtomicBool>,
)->Result<Value,String>{
    if cancel_generation.load(Ordering::SeqCst) {
        return Err("Generation stopped.".into());
    }
    let response=reqwest::Client::new().post(format!("{}/prompt",base_url(&req.comfy_url))).json(&json!({"prompt":req.workflow,"client_id":"raphael-prompt-forge"})).send().await.map_err(|e|e.to_string())?;
    let status=response.status(); let text=response.text().await.unwrap_or_default();
    if cancel_generation.load(Ordering::SeqCst) {
        let _=interrupt_comfy(&req.comfy_url).await;
        return Err("Generation stopped.".into());
    }
    if !status.is_success(){return Err(format!("ComfyUI returned HTTP {}: {}",status,text))}
    serde_json::from_str(&text).map_err(|e|format!("Invalid ComfyUI response: {}",e))
}

#[tauri::command]
async fn submit_to_comfy(
    state:tauri::State<'_,AppState>,
    req:SubmitRequest,
)->Result<Value,String>{
    submit_to_comfy_inner(req,state.cancel_generation.clone()).await
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
    cancel_generation:Arc<AtomicBool>,
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
        if cancel_generation.load(Ordering::SeqCst) {
            return Err("Generation stopped.".into());
        }
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
    state:tauri::State<'_,AppState>,
    req:MonitorComfyRequest,
    on_event:Channel<ComfyProgress>,
)->Result<ComfyGenerationResult,String>{
    monitor_comfy_generation_inner(req, state.cancel_generation.clone(), |progress| {
        on_event.send(progress).map_err(|e|format!("ComfyUI channel closed: {}",e))
    }).await
}


fn lan_ip() -> Result<String,String> {
    let address = local_ip()
        .map_err(|error| format!("Could not determine the local LAN IPv4 address: {error}"))?;
    match address {
        std::net::IpAddr::V4(ip) if ip.is_private() || ip.is_link_local() => Ok(ip.to_string()),
        std::net::IpAddr::V4(ip) => Err(format!("Local address {ip} is not a private LAN address. Connect the computer and mobile device to the same private network and try again.")),
        std::net::IpAddr::V6(ip) => Err(format!("Local address {ip} is IPv6-only. A private IPv4 LAN address is required for the current mobile web host.")),
    }
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
    if command=="start_generation" {
        state.cancel_generation.store(false, Ordering::SeqCst);
        return AxumJson(json!({"started":true})).into_response();
    }
    if command=="stream_llm" {
        return web_stream_llm(state.clone(), AxumJson(body)).await.into_response();
    }
    if command=="stop_generation" {
        state.cancel_generation.store(true, Ordering::SeqCst);
        let comfy_url=body.get("comfyUrl").and_then(|v|v.as_str()).unwrap_or("");
        if !comfy_url.trim().is_empty() {
            let _=interrupt_comfy(comfy_url).await;
        }
        return AxumJson(json!({"stopped":true})).into_response();
    }
    if command=="monitor_comfy_generation" {
        return web_monitor_comfy(state.clone(), AxumJson(body)).await.into_response();
    }

    let req_value=body.get("req").cloned().unwrap_or_else(||body.clone());
    let result:Result<Value,String>=async {
        match command.as_str(){
        "discover_raphael_config"=>serde_json::to_value(discover_raphael_config()).map_err(|e|e.to_string()),
        "discover_raphael_roots"=>serde_json::to_value(discover_raphael_roots()).map_err(|e|e.to_string()),
        "list_provider_models"=>{
            let settings = state.llm.lock().await.clone();
            let models=list_provider_models(settings.clone()).await?;
            serde_json::to_value(models).map_err(|e|e.to_string())
        }
        "get_host_llm_config"=>{
            let settings = state.llm.lock().await.clone();
            let generation_settings=public_generation_settings(&*state.generation_settings.lock().await);
            // Expose only non-secret LLM configuration to LAN clients. The
            // API key deliberately stays host-side and is never serialized here.
            serde_json::to_value(json!({
                "provider": settings.provider,
                "baseUrl": settings.base_url,
                "model": settings.model,
                "temperature": settings.temperature,
                "maxTokens": settings.max_tokens,
                "contextTokens": settings.context_tokens,
                "generationSettings": generation_settings
            })).map_err(|e|e.to_string())
        }
        "get_host_generation_settings"=>{
            let generation_settings=public_generation_settings(&*state.generation_settings.lock().await);
            serde_json::to_value(generation_settings).map_err(|e|e.to_string())
        }
        "submit_to_comfy"=>{
            if state.cancel_generation.load(Ordering::SeqCst) {
                return Err("Generation stopped.".into());
            }
            let request:SubmitRequest=serde_json::from_value(req_value).map_err(|e|e.to_string())?;
            let result=submit_to_comfy_inner(request,state.cancel_generation.clone()).await?;
            if state.cancel_generation.load(Ordering::SeqCst) {
                return Err("Generation stopped.".into());
            }
            Ok(result)
        }
        "update_web_host_generation_settings"=>{
            let settings_value=body.get("generationSettings").cloned().unwrap_or(req_value);
            *state.generation_settings.lock().await=settings_value;
            Ok(Value::Null)
        }
        "update_web_host_llm"=>{
            let settings_value=body.get("llmSettings").cloned().unwrap_or(req_value);
            let mut settings=serde_json::from_value::<LlmSettings>(settings_value).map_err(|e|e.to_string())?;
            let mut host_settings=state.llm.lock().await;
            if settings.api_key.trim().is_empty() {
                settings.api_key=host_settings.api_key.clone();
            }
            *host_settings=settings;
            Ok(Value::Null)
        }
        "scan_library"=>{
            let request=serde_json::from_value::<ScanRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(scan_library(request).await?).map_err(|e|e.to_string())
        }
        "prepare_generation"=>{
            let request=serde_json::from_value::<PrepareRequest>(req_value).map_err(|e|e.to_string())?;
            serde_json::to_value(prepare_generation(request).await?).map_err(|e|e.to_string())
        }
        "parse_prompt_pair"=>{
            let raw=body
                .get("raw")
                .and_then(|value|value.as_str())
                .or_else(||req_value.as_str())
                .ok_or_else(||"raw prompt text is required".to_string())?;
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
        "history_revision"=>serde_json::to_value(history_revision(state.app.clone()).await?).map_err(|e|e.to_string()),
        "load_history"=>serde_json::to_value(load_history(state.app.clone()).await?).map_err(|e|e.to_string()),
        "load_history_index"=>serde_json::to_value(load_history_index(state.app.clone()).await?).map_err(|e|e.to_string()),
        "load_history_item"=>{
            let id=req_value.get("id").and_then(Value::as_str).ok_or_else(||"history item id is required".to_string())?;
            serde_json::to_value(load_history_item(state.app.clone(),id.to_string()).await?).map_err(|e|e.to_string())
        }
        "load_history_image"=>{
            let id=req_value.get("id").and_then(Value::as_str).ok_or_else(||"history image id is required".to_string())?;
            serde_json::to_value(load_history_image(state.app.clone(),id.to_string()).await?).map_err(|e|e.to_string())
        }
        "append_history"=>{
            let payload=body.get("payload").cloned().unwrap_or(Value::Null);
            serde_json::to_value(append_history(state.app.clone(),payload).await?).map_err(|e|e.to_string())
        }
        "path_to_data_url"=>{
            let path=req_value.get("path").and_then(|x|x.as_str())
                .or_else(||req_value.as_str())
                .ok_or_else(||"path is required".to_string())?;
            if !web_client_path_is_allowed(path) {
                return Err("LAN clients may only request Registry thumbnail assets.".into());
            }
            serde_json::to_value(path_to_data_url(path.to_string()).await?).map_err(|e|e.to_string())
        }
        _=>Err(format!("Unknown API command: {}",command))
        }
    }.await;
    match result{Ok(value)=>AxumJson(value).into_response(),Err(e)=>http_error(e)}
}

async fn web_stream_llm(
    state: WebApiState,
    AxumJson(body):AxumJson<Value>,
)->Sse<impl Stream<Item=Result<Event,std::convert::Infallible>>>{
    let host_settings = state.llm.lock().await.clone();
    let parsed:Result<LlmRequest,String>=serde_json::from_value(body.get("req").cloned().unwrap_or(body.clone())).map_err(|e|e.to_string());
    let req = parsed.map(|mut request| {
        request.settings.provider = host_settings.provider.clone();
        request.settings.base_url = host_settings.base_url.clone();
        request.settings.api_key = host_settings.api_key.clone();
        request
    });
    let (tx,rx)=tokio::sync::mpsc::unbounded_channel::<Result<Event,std::convert::Infallible>>();
    tokio::spawn(async move{
        match req{
            Ok(req)=>{
                let tx2=tx.clone();
                let mut emit=|delta:LlmDelta|{
                    let event=Event::default().json_data(json!({"text":delta.text,"done":false})).map_err(|e|format!("sse:{}",e))?;
                    tx2.send(Ok(event)).map_err(|_|"sse client disconnected".to_string())
                };
                match stream_llm_inner(req,state.cancel_generation.clone(),&mut emit).await{
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
    state:WebApiState,
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
                match monitor_comfy_generation_inner(req,state.cancel_generation.clone(),&mut emit).await{
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
static HISTORY_FILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn history_path(app:&AppHandle)->Result<PathBuf,String>{
    let dir=app.path().app_data_dir().map_err(|e|e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
    Ok(dir.join("generation-history.json"))
}

fn history_revision_file(path:&Path)->Result<String,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    if !path.exists(){return Ok("missing".into());}
    let metadata=fs::metadata(path).map_err(|e|e.to_string())?;
    let modified=metadata.modified()
        .map_err(|e|e.to_string())?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(format!("{}:{}",metadata.len(),modified))
}

fn history_images_dir(path:&Path)->PathBuf{
    path.parent().unwrap_or_else(||Path::new(".")).join("generation-history-images")
}

fn history_image_extension(mime:&str)->(&'static str,&'static str){
    match mime.to_ascii_lowercase().as_str(){
        "image/jpeg"|"image/jpg"=>("jpg","image/jpeg"),
        "image/webp"=>("webp","image/webp"),
        "image/gif"=>("gif","image/gif"),
        _=>("png","image/png"),
    }
}

fn store_history_image(path:&Path,data_url:&str)->Result<Option<String>,String>{
    let Some((header,encoded))=data_url.split_once(',') else{return Ok(None)};
    if !header.starts_with("data:") || !header.contains(";base64"){return Ok(None);}
    let mime=header[5..].split(';').next().unwrap_or("image/png");
    let (extension,_)=history_image_extension(mime);
    let Ok(bytes)=base64::engine::general_purpose::STANDARD.decode(encoded) else{return Ok(None)};
    let directory=history_images_dir(path);
    fs::create_dir_all(&directory).map_err(|e|format!("Could not create history image directory: {e}"))?;
    let filename=format!("{}.{}",now_id(),extension);
    fs::write(directory.join(&filename),bytes).map_err(|e|format!("Could not save generated history image: {e}"))?;
    Ok(Some(filename))
}

fn write_history_records_file(path:&Path,records:&[HistoryRecord])->Result<(),String>{
    let serialized=serde_json::to_vec(records).map_err(|e|e.to_string())?;
    let temp=path.with_file_name(format!("generation-history-{}.tmp",now_id()));
    fs::write(&temp,serialized).map_err(|e|format!("Could not write temporary generation history: {e}"))?;
    if let Err(error)=fs::rename(&temp,path){
        let _=fs::remove_file(&temp);
        return Err(format!("Could not replace generation history file: {error}"));
    }
    Ok(())
}

fn read_history_records_file(path:&Path)->Result<Vec<HistoryRecord>,String>{
    if !path.exists(){return Ok(vec![]);}
    serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?)
        .map_err(|e|format!("Could not read generation history JSON: {e}"))
}

// One-time migration: older history entries stored complete base64 images inside
// generation-history.json. Move valid image data into sidecar files so subsequent
// history reads and appends touch only compact metadata.
fn migrate_history_images(path:&Path,records:&mut [HistoryRecord])->Result<bool,String>{
    let mut changed=false;
    for record in records.iter_mut(){
        if record.payload.get("_historyImageFile").and_then(Value::as_str).is_some(){continue;}
        let Some(data_url)=record.payload.get("imageDataUrl").and_then(Value::as_str).map(str::to_string) else{continue};
        if let Some(filename)=store_history_image(path,&data_url)?{
            if let Some(object)=record.payload.as_object_mut(){
                object.remove("imageDataUrl");
                object.insert("_historyImageFile".into(),Value::String(filename));
                changed=true;
            }
        }
    }
    if changed{write_history_records_file(path,records)?;}
    Ok(changed)
}

fn history_image_data_url(path:&Path,filename:&str)->Option<String>{
    // The file name is generated by this application; reject path components
    // in persisted metadata to prevent a malformed history file escaping its directory.
    if Path::new(filename).file_name()?.to_str()?!=filename{return None;}
    let bytes=fs::read(history_images_dir(path).join(filename)).ok()?;
    let extension=Path::new(filename).extension().and_then(|x|x.to_str()).unwrap_or("png");
    let mime=match extension.to_ascii_lowercase().as_str(){
        "jpg"|"jpeg"=>"image/jpeg",
        "webp"=>"image/webp",
        "gif"=>"image/gif",
        _=>"image/png",
    };
    Some(format!("data:{};base64,{}",mime,base64::engine::general_purpose::STANDARD.encode(bytes)))
}

fn hydrate_history_payload(path:&Path,payload:&mut Value,include_image:bool){
    let filename=payload.get("_historyImageFile").and_then(Value::as_str).map(str::to_string);
    if let Some(object)=payload.as_object_mut(){
        object.remove("_historyImageFile");
        if let Some(filename)=filename{
            if include_image{
                if let Some(data_url)=history_image_data_url(path,&filename){
                    object.insert("imageDataUrl".into(),Value::String(data_url));
                }
            }else{
                object.insert("hasImage".into(),Value::Bool(true));
            }
        }else if let Some(data_url)=object.get("imageDataUrl").and_then(Value::as_str){
            if !include_image{
                let has_image=!data_url.is_empty();
                object.remove("imageDataUrl");
                object.insert("hasImage".into(),Value::Bool(has_image));
            }
        }else if include_image{
            object.remove("hasImage");
        }else{
            object.insert("hasImage".into(),Value::Bool(false));
        }
        if include_image{object.remove("hasImage");}
    }
}

fn load_history_file(path:&Path)->Result<Vec<HistoryRecord>,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    let mut all=read_history_records_file(path)?;
    migrate_history_images(path,&mut all)?;
    for record in &mut all{hydrate_history_payload(path,&mut record.payload,true);}
    Ok(all)
}

fn load_history_index_file(path:&Path)->Result<Vec<HistoryRecord>,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    let mut all=read_history_records_file(path)?;
    migrate_history_images(path,&mut all)?;
    for record in &mut all{
        hydrate_history_payload(path,&mut record.payload,false);
        // The history list only needs identity, timestamp, model summaries and
        // image availability. Prompts, large workflows and settings are fetched
        // only after the user opens a particular record.
        if let Some(object)=record.payload.as_object_mut(){
            object.remove("generationSettings");
            object.remove("workflow");
            object.remove("positivePrompt");
            object.remove("negativePrompt");
            object.remove("rationale");
        }
    }
    Ok(all)
}

fn load_history_item_file(path:&Path,id:&str)->Result<Value,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    let mut all=read_history_records_file(path)?;
    migrate_history_images(path,&mut all)?;
    let Some(record)=all.iter_mut().find(|record|
        record.payload.get("id").and_then(Value::as_str)==Some(id)||record.id==id
    ) else{return Err("Generation history item was not found.".into())};
    // Full record metadata is loaded separately from image bytes. The UI
    // requests the image only when the detail view needs to display it.
    hydrate_history_payload(path,&mut record.payload,false);
    Ok(record.payload.clone())
}

fn load_history_image_file(path:&Path,id:&str)->Result<Option<String>,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    let mut all=read_history_records_file(path)?;
    migrate_history_images(path,&mut all)?;
    let Some(record)=all.iter().find(|record|
        record.payload.get("id").and_then(Value::as_str)==Some(id)||record.id==id
    ) else{return Ok(None)};
    if let Some(data_url)=record.payload.get("imageDataUrl").and_then(Value::as_str){
        return Ok(Some(data_url.to_string()));
    }
    Ok(record.payload.get("_historyImageFile").and_then(Value::as_str)
        .and_then(|filename|history_image_data_url(path,filename)))
}

fn append_history_file(path:&Path,payload:Value)->Result<HistoryRecord,String>{
    let _guard=HISTORY_FILE_LOCK.lock()
        .map_err(|_|"Generation history lock is poisoned.".to_string())?;
    let mut all=read_history_records_file(path)
        .map_err(|e|format!("Could not read existing generation history JSON; refusing to overwrite it: {e}"))?;
    migrate_history_images(path,&mut all)?;
    let mut stored_payload=payload;
    if let Some(data_url)=stored_payload.get("imageDataUrl").and_then(Value::as_str).map(str::to_string){
        if let Some(filename)=store_history_image(path,&data_url)?{
            if let Some(object)=stored_payload.as_object_mut(){
                object.remove("imageDataUrl");
                object.insert("_historyImageFile".into(),Value::String(filename));
            }
        }
    }
    let rec=HistoryRecord{id:now_id(),timestamp:now_id(),payload:stored_payload};
    all.insert(0,rec.clone());
    if all.len()>100{all.truncate(100);}
    write_history_records_file(path,&all)?;

    // Keep disk usage bounded when the oldest history records are discarded.
    let retained:std::collections::HashSet<String>=all.iter()
        .filter_map(|record|record.payload.get("_historyImageFile").and_then(Value::as_str).map(str::to_string))
        .collect();
    let directory=history_images_dir(path);
    if let Ok(entries)=fs::read_dir(&directory){
        for entry in entries.flatten(){
            if let Some(name)=entry.file_name().to_str().map(str::to_string){
                if !retained.contains(&name){let _=fs::remove_file(entry.path());}
            }
        }
    }
    Ok(rec)
}

#[tauri::command]
async fn history_revision(app:AppHandle)->Result<String,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || history_revision_file(&path))
        .await
        .map_err(|e|format!("History revision worker failed: {e}"))?
}

#[tauri::command]
async fn load_history(app:AppHandle)->Result<Vec<HistoryRecord>,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || load_history_file(&path))
        .await
        .map_err(|e|format!("History read worker failed: {e}"))?
}

#[tauri::command]
async fn load_history_index(app:AppHandle)->Result<Vec<HistoryRecord>,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || load_history_index_file(&path))
        .await
        .map_err(|e|format!("History index read worker failed: {e}"))?
}

#[tauri::command]
async fn load_history_item(app:AppHandle,id:String)->Result<Value,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || load_history_item_file(&path,&id))
        .await
        .map_err(|e|format!("History item read worker failed: {e}"))?
}

#[tauri::command]
async fn load_history_image(app:AppHandle,id:String)->Result<Option<String>,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || load_history_image_file(&path,&id))
        .await
        .map_err(|e|format!("History image read worker failed: {e}"))?
}

#[tauri::command]
async fn append_history(app:AppHandle,payload:Value)->Result<HistoryRecord,String>{
    let path=history_path(&app)?;
    tokio::task::spawn_blocking(move || append_history_file(&path,payload))
        .await
        .map_err(|e|format!("History write worker failed: {e}"))?
}

#[tauri::command]
async fn path_to_data_url(path: String) -> Result<String, String> {
    if let Some(reference) = path.strip_prefix("registry://") {
        let mut parts = reference.splitn(2, '/');
        let model_id = parts.next().unwrap_or_default();
        let asset_id = parts.next().unwrap_or_default();
        if model_id.is_empty() || asset_id.is_empty() {
            return Err("Invalid Registry asset reference.".into());
        }

        let (base_url, token) = ensure_registry(None).await?;
        let response = reqwest::Client::new()
            .get(format!("{}/api/v1/models/{}/assets/{}/content", base_url, urlencoding::encode(model_id), urlencoding::encode(asset_id)))
            .bearer_auth(token)
            .header("x-raphael-actor", "image-generator")
            .send()
            .await
            .map_err(|error| format!("Registry asset request failed: {error}"))?;

        let status = response.status();
        if !status.is_success() {
            return Err(format!("Registry asset request returned HTTP {status}"));
        }

        let mime = response.headers().get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("image/png")
            .split(';').next().unwrap_or("image/png").to_string();
        let bytes = response.bytes().await.map_err(|error| format!("Registry asset read failed: {error}"))?;
        return Ok(format!("data:{};base64,{}", mime, base64::engine::general_purpose::STANDARD.encode(bytes)));
    }

    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    let mime = match Path::new(&path).extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    };
    Ok(format!("data:{};base64,{}", mime, base64::engine::general_purpose::STANDARD.encode(bytes)))
}

#[tauri::command]
async fn start_web_host(
    app: AppHandle,
    state: tauri::State<'_,AppState>,
    port: Option<u16>,
    llm_settings: Option<LlmSettings>,
    generation_settings: Option<Value>,
)->Result<Value,String>{
    let mut host=state.web_host.lock().await;
    if let Some(existing)=host.as_ref(){
        if let Some(settings) = llm_settings {
            *existing.llm.lock().await = settings;
        }
        if let Some(settings) = generation_settings {
            *existing.generation_settings.lock().await = settings;
        }
        return Ok(json!({"running":true,"port":existing.port,"localUrl":existing.lan_url,"lanUrl":existing.lan_url}));
    }

    let chosen_port=port.unwrap_or(1424);
    let lan_host=lan_ip()?;
    let listener=tokio::net::TcpListener::bind(("0.0.0.0",chosen_port))
        .await
        .map_err(|e|format!("LAN host could not bind 0.0.0.0:{}: {}",chosen_port,e))?;
    let actual_port=listener.local_addr().map_err(|e|e.to_string())?.port();
    let dist=dist_directory(&app).ok_or("Could not find dist/index.html. Run npm run build first.")?;
    let llm = Arc::new(Mutex::new(llm_settings.unwrap_or_else(|| LlmSettings {
        provider: "ollama".into(),
        base_url: "http://127.0.0.1:11434".into(),
        api_key: String::new(),
        model: String::new(),
        temperature: 0.72,
        max_tokens: 8192,
        context_tokens: 32768,
    })));
    let generation_settings=Arc::new(Mutex::new(generation_settings.unwrap_or_else(||json!({}))));
    let api_state=WebApiState{
        app:app.clone(),
        llm:llm.clone(),
        generation_settings:generation_settings.clone(),
        cancel_generation:state.cancel_generation.clone(),
    };
    let router=Router::new()
        .route("/health", axum::routing::get(|| async {
            AxumJson(json!({"ok":true,"service":"raphael-prompt-forge"}))
        }))
        .route("/api/{command}",post(web_command))
        .fallback_service(ServeDir::new(dist))
        // Generation history includes base64-encoded images. Axum's default
        // JSON body limit (2 MiB) rejects many real outputs before they can be
        // persisted on the host. Keep a finite, LAN-friendly upper bound.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(api_state);
    let lan=format!("http://{}:{}",lan_host,actual_port);
    let task=tokio::spawn(async move{
        let _=axum::serve(listener,router).await;
    });
    let url=json!({"running":true,"port":actual_port,"localUrl":lan,"lanUrl":lan});
    *host=Some(WebHostRuntime{port:actual_port,lan_url:lan,llm,generation_settings,task});
    Ok(url)
}

#[tauri::command]
async fn get_web_host_shared_settings(state:tauri::State<'_,AppState>)->Result<Value,String>{
    // This command is called by the desktop Tauri UI, not the LAN HTTP API.
    // Clone the shared stores before awaiting them so the outer host lock is
    // not held while reading settings.
    let (llm_store,generation_settings_store)={
        let host=state.web_host.lock().await;
        let Some(runtime)=host.as_ref() else {
            return Err("LAN web host is not running.".into());
        };
        (runtime.llm.clone(),runtime.generation_settings.clone())
    };
    let settings=llm_store.lock().await.clone();
    let generation_settings=public_generation_settings(&*generation_settings_store.lock().await);
    Ok(json!({
        "provider":settings.provider,
        "baseUrl":settings.base_url,
        "model":settings.model,
        "temperature":settings.temperature,
        "maxTokens":settings.max_tokens,
        "contextTokens":settings.context_tokens,
        "generationSettings":generation_settings
    }))
}

#[tauri::command]
async fn update_web_host_llm(state:tauri::State<'_,AppState>,llm_settings:LlmSettings)->Result<(),String>{
    let host=state.web_host.lock().await;
    let Some(runtime)=host.as_ref() else {
        return Err("LAN web host is not running.".into());
    };
    *runtime.llm.lock().await = llm_settings;
    Ok(())
}

#[tauri::command]
async fn update_web_host_generation_settings(
    state:tauri::State<'_,AppState>,
    generation_settings:Value,
)->Result<(),String>{
    let host=state.web_host.lock().await;
    let Some(runtime)=host.as_ref() else {
        return Err("LAN web host is not running.".into());
    };
    *runtime.generation_settings.lock().await=generation_settings;
    Ok(())
}

#[tauri::command]
async fn stop_web_host(state:tauri::State<'_,AppState>)->Result<(),String>{
    if let Some(runtime)=state.web_host.lock().await.take(){runtime.task.abort();}
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run(){
    tauri::Builder::default()
        .manage(AppState{
            active_stream:Arc::new(Mutex::new(false)),
            cancel_generation:Arc::new(AtomicBool::new(false)),
            web_host:Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            pick_folder,discover_raphael_config,discover_raphael_roots,scan_library,list_provider_models,
            prepare_generation,stream_llm,parse_prompt_pair,finalize_prompt_pair,build_workflow,inject_prompts,
            submit_to_comfy,monitor_comfy_generation,start_generation,stop_generation,history_revision,load_history,load_history_index,load_history_item,load_history_image,append_history,path_to_data_url,start_web_host,get_web_host_shared_settings,update_web_host_llm,update_web_host_generation_settings,stop_web_host
        ])
        .run(tauri::generate_context!())
        .expect("error while running Raphael Prompt Forge");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn web_client_thumbnail_access_is_registry_only() {
        assert!(web_client_path_is_allowed("registry://model-123/asset-456"));
        assert!(web_client_path_is_allowed("  registry://model-123/asset-456"));
        assert!(!web_client_path_is_allowed("C:\\Windows\\System32\\drivers\\etc\\hosts"));
        assert!(!web_client_path_is_allowed("D:/ComfyUI/models/checkpoints/model.safetensors"));
    }

    #[test]
    fn registry_absolute_file_path_does_not_require_client_models_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model_path = temp.path().join("host-model.safetensors");
        std::fs::write(&model_path, b"model").expect("model file");

        let file = RegistryFile {
            id: "test-file".into(),
            version_id: None,
            path: model_path.to_string_lossy().to_string(),
            relative_path: None,
            filename: "host-model.safetensors".into(),
            size_bytes: 5,
            status: "available".into(),
        };

        let resolved = registry_model_file_path(&file, Some(Path::new("D:\\nonexistent\\client\\models")))
            .expect("Registry absolute path should resolve on the host");
        assert_eq!(resolved, model_path.canonicalize().unwrap());
    }


    #[test]
    fn finalize_prompt_strips_generated_lora_syntax() {
        let result=finalize_positive_prompt(
            "anime character, [Style - Nekoya@4x0style]1.3, <lora:foo:1.2>, detailed face",
        );
        assert!(!result.contains("[Style - Nekoya@4x0style]1.3"));
        assert!(!result.contains("<lora:foo:1.2>"));
        assert!(result.contains("anime character"));
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
            source:"test".into(), description:None,
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

    #[tokio::test]
    #[ignore]
    async fn full_generation_pipeline_dry_run() {
        let checkpoint = ModelInfo {
            id:"cp1".into(), name:"Anima Base".into(), kind:"checkpoint".into(),
            path:"D:/ComfyUI/models/checkpoints/anima.safetensors".into(), size:1,
            base_model:Some("anima".into()), tags:vec!["anima".into()],
            activation_tags:vec![], character:false, thumbnail:None,
            source:"raphael-registry".into(), description:None,
        };
        let loras = vec![
            ModelInfo {
                id:"l1".into(), name:"Character Alice".into(), kind:"lora".into(),
                path:"D:/ComfyUI/models/loras/alice.safetensors".into(), size:1,
                base_model:Some("anima".into()), tags:vec!["anima".into(),"character".into(),"alice".into()],
                activation_tags:vec!["alice_trigger".into()], character:true, thumbnail:None,
                source:"raphael-registry".into(), description:Some("Character identity LoRA for Alice".into()),
            },
            ModelInfo {
                id:"l2".into(), name:"School Uniform".into(), kind:"lora".into(),
                path:"D:/ComfyUI/models/loras/uniform.safetensors".into(), size:1,
                base_model:Some("anima".into()), tags:vec!["anima".into(),"outfit".into()],
                activation_tags:vec!["school_uniform_trigger".into()], character:false, thumbnail:None,
                source:"raphael-registry".into(), description:Some("Japanese school uniform clothing".into()),
            },
        ];
        let prepared = prepare_generation(PrepareRequest {
            checkpoint:checkpoint.clone(), loras:loras,
            selected_lora_ids:vec!["l1".into(),"l2".into()],
            setting:"classroom".into(), pose:"standing".into(), expression:"smiling".into(),
            character:"".into(), dress:"".into(), composition:"three-quarter".into(),
            additional:"".into(), random_lora_min:2, random_lora_max:2,
            max_character_loras:1,
            registry_url:None,
        }).await.expect("prepare_generation dry run must succeed");
        assert_eq!(prepared.loras.len(), 2);

        let raw_pair = PromptPair {
            positive_prompt:"Alice in a classroom with alice_trigger, smiling in school_uniform_trigger".into(),
            negative_prompt:"blurry, malformed hands".into(),
            rationale:None,
        };
        let finalized = finalize_prompt_pair(FinalizePromptRequest {
            prompt_pair:raw_pair,
            loras:prepared.loras.clone(),
        }).expect("finalize prompt dry run must succeed");
        assert!(finalized.positive_prompt.contains("alice_trigger"));
        assert!(finalized.positive_prompt.contains("school_uniform_trigger"));

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
    fn activation_prompts_are_preserved_in_llm_chosen_positions() {
        let pair = PromptPair {
            positive_prompt: "portrait, triggerA, blue eyes, style_tag, serene expression".into(),
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
            "portrait, triggerA, blue eyes, style_tag, serene expression"
        );
        assert!(!finalized.positive_prompt.contains("char_tag"));
        assert_eq!(finalized.negative_prompt, "blurry");
    }

    #[test]
    fn history_index_keeps_images_out_of_metadata_and_loads_them_on_demand() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("generation-history.json");
        let image = "data:image/png;base64,AQID";
        append_history_file(
            &path,
            json!({
                "id":"generation-lazy-image-test",
                "imageDataUrl":image,
                "checkpoint":{"name":"test"},
                "positivePrompt":"full positive prompt",
                "negativePrompt":"full negative prompt",
                "workflow":{"node":"workflow-payload"},
                "generationSettings":{"systemPrompt":"large settings payload"}
            }),
        ).expect("history append should succeed");

        let stored = std::fs::read_to_string(&path).expect("read compact history");
        assert!(
            !stored.contains(image),
            "image bytes must not remain embedded in the history metadata file"
        );

        let index = load_history_index_file(&path).expect("load history index");
        assert_eq!(index.len(), 1);
        assert_eq!(index[0].payload["hasImage"], true);
        assert!(index[0].payload.get("imageDataUrl").is_none());
        assert!(index[0].payload.get("positivePrompt").is_none());
        assert!(index[0].payload.get("negativePrompt").is_none());
        assert!(index[0].payload.get("workflow").is_none());
        assert!(index[0].payload.get("generationSettings").is_none());

        assert_eq!(
            load_history_image_file(&path, "generation-lazy-image-test")
                .expect("load history image"),
            Some(image.to_string())
        );
        let item = load_history_item_file(&path, "generation-lazy-image-test")
            .expect("load full history item");
        assert_eq!(item["hasImage"], true);
        assert!(item.get("imageDataUrl").is_none());
        assert_eq!(item["positivePrompt"], "full positive prompt");
        assert_eq!(item["negativePrompt"], "full negative prompt");
        assert_eq!(item["workflow"]["node"], "workflow-payload");
        assert_eq!(item["generationSettings"]["systemPrompt"], "large settings payload");
    }

    #[test]
    fn history_archive_round_trips_image_payload_larger_than_two_mib() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("generation-history.json");
        let image = format!("data:image/png;base64,{}", "A".repeat(3 * 1024 * 1024));

        append_history_file(&path, json!({"imageDataUrl": image}))
            .expect("large image history should be persisted");
        let records = load_history_file(&path).expect("history should be readable");

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].payload["imageDataUrl"].as_str().map(str::len),
            Some("data:image/png;base64,".len() + 3 * 1024 * 1024)
        );
    }

    #[test]
    fn concurrent_history_appends_do_not_overwrite_each_other() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("generation-history.json");

        std::thread::scope(|scope| {
            for worker in 0..8 {
                let path = &path;
                scope.spawn(move || {
                    for item in 0..10 {
                        append_history_file(path, json!({"worker":worker,"item":item}))
                            .expect("history append should succeed");
                    }
                });
            }
        });

        let records = load_history_file(&path).expect("history should be readable");
        assert_eq!(records.len(), 80, "every concurrent append must be retained");
        let unique: std::collections::HashSet<String> = records
            .iter()
            .map(|record| format!("{}:{}", record.payload["worker"], record.payload["item"]))
            .collect();
        assert_eq!(unique.len(), 80, "no appended record should be duplicated");
    }

    #[test]
    fn malformed_history_is_not_silently_overwritten() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("generation-history.json");
        std::fs::write(&path, b"{ invalid json").expect("write malformed fixture");

        let result = append_history_file(&path, json!({"shouldNotReplaceArchive":true}));

        assert!(result.is_err(), "malformed history must return an error");
        assert_eq!(
            std::fs::read(&path).expect("history file must remain"),
            b"{ invalid json",
            "malformed history must be preserved for recovery"
        );
    }


    #[test]
    fn history_revision_changes_after_append_and_stays_stable_without_writes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("generation-history.json");

        let before = history_revision_file(&path).expect("revision for missing history");
        append_history_file(&path, json!({"generation":1})).expect("append initial record");
        let after = history_revision_file(&path).expect("revision after first write");
        let stable = history_revision_file(&path).expect("revision without another write");

        assert_ne!(before, after, "a new history record must change the revision");
        assert_eq!(after, stable, "unchanged history must keep a stable revision");
        append_history_file(&path, json!({"generation":2})).expect("append second record");
        let second = history_revision_file(&path).expect("revision after second write");
        assert_ne!(after, second, "a later append must change the revision");
    }

}
