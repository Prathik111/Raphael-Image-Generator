# Raphael Prompt Forge

Standalone Tauri + React desktop app for generating ComfyUI prompts/workflows from local LLMs and a local ComfyUI LoRA library.

## Key behavior

- Uses the **exact Raphael Model Manager background scene** extracted from the Raphael prototype's embedded background HTML. The file is carried as `public/raphael-background.html` and rendered directly as the full-screen background.
- Lets you choose **Ollama** or any **OpenAI-compatible** local/server endpoint. Ollama model discovery uses its local model list; OpenAI-compatible discovery uses `/v1/models`.
- Uses the **Raphael Model Registry as the canonical model metadata source**. The Registry supplies model identity, versions, files, tags, activation prompts, descriptions, compatibility and assets; the ComfyUI directory is used only to verify that the Registry-registered physical file is present.
- Presents checkpoints as Raphael-style cards using Registry metadata and Registry-hosted thumbnail/cover assets.
- Uses the Registry compatibility API for checkpoint/LoRA compatibility, including explicit Registry relationships and matching base-model metadata.
- Character identity is restricted to compatible LoRAs marked with the `character` tag; the LLM is explicitly forbidden from inventing a character outside the selected character LoRA.
- Randomizes compatible LoRA count, LoRA weights, scene setting, pose, expression, dress and composition whenever those fields are left blank.
- Streams the generated prompt response live into the UI while the LLM is producing it.
- Uses separate backend tools for deterministic workflow construction and prompt injection before the ComfyUI request.
- Records generations locally with checkpoint, LoRA stack, scene, positive prompt, negative prompt, workflow and ComfyUI prompt id.

## Run on Windows

```powershell
npm install
npm run tauri:dev
```

Production build:

```powershell
npm run tauri:build
```

## Setup

Open **SETTINGS** first:

1. Choose **OLLAMA** or **OPENAI COMPATIBLE**.
2. Set the base URL and press **GET MODELS**. For llama.cpp, the default shown is `http://127.0.0.1:8080/v1`.
3. Choose the model.
4. Choose your ComfyUI `models` directory.
5. Set the **Registry URL**, normally `http://127.0.0.1:43217`. The image generator automatically checks the Registry health and starts `raphael-registry server` when it is not running.
6. Set the ComfyUI API URL, normally `http://127.0.0.1:8188`.
7. Scan the library, select a checkpoint card, enter any scene constraints and press **GENERATE**.

## Backend command/tool boundaries

- `scan_library` — Registry model discovery/hydration + physical-file validation against the selected ComfyUI models root.
- `prepare_generation` — Registry compatibility checking + random LoRA stack + random scene choices.
- `stream_llm` — provider-specific streaming adapter.
- `build_workflow` — deterministic standard ComfyUI API workflow construction.
- `inject_prompts` — deterministic replacement of prompt markers in the workflow.
- `submit_to_comfy` — actual ComfyUI `/prompt` submission.
- `append_history` / `load_history` — local generation archive.

The workflow builder is intentionally isolated behind a command boundary, so you can later replace the standard KSampler workflow with the exact deterministic workflow template you use in your ComfyUI installation without changing the UI or prompt engine.


## Registry startup

The host expects the Registry server executable to be discoverable as `raphael-registry` or `raphael-registry.exe`, or you can set `RAPHAEL_REGISTRY_EXECUTABLE` to its full path. The Registry data directory and token follow the Registry application standard location; `RAPHAEL_REGISTRY_DATA_DIR`, `RAPHAEL_REGISTRY_AUTH_TOKEN`, and `RAPHAEL_REGISTRY_URL` can override discovery when required.
