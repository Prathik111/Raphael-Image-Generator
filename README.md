# Raphael Prompt Forge

Standalone Tauri + React desktop app for generating ComfyUI prompts/workflows from local LLMs and a local ComfyUI LoRA library.

## Key behavior

- Uses the **exact Raphael Model Manager background scene** extracted from the Raphael prototype's embedded background HTML. The file is carried as `public/raphael-background.html` and rendered directly as the full-screen background.
- Lets you choose **Ollama** or any **OpenAI-compatible** local/server endpoint. Ollama model discovery uses its local model list; OpenAI-compatible discovery uses `/v1/models`.
- Scans a real ComfyUI `models` directory and merges Raphael Model Manager / Registry JSON cache metadata where matching files are found.
- Presents checkpoints as Raphael-style cards with cached thumbnails, names, tags, base metadata and cache provenance.
- Uses strict LoRA compatibility: explicit base metadata must match the selected checkpoint or the LoRA must have a matching normalized base tag/name. Empty/unknown LoRA metadata is not treated as automatically compatible.
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

## LAN web host

When WEB HOST is enabled from the Tauri desktop app, the desktop PC is the Raphael host. LAN browsers are clients of that host and do not need Ollama or ComfyUI installed locally.

- Ollama model discovery and LLM generation run on the host PC.
- ComfyUI submission and generation monitoring run on the host PC.
- Generation settings, provider settings, model selection, scene settings, LoRA selection, model roots and ComfyUI URL are persisted on the host and synchronized to connected clients.
- Generation history is already stored on the host and is shared by LAN clients.
- Clients poll the host settings revision so changes made on the desktop or another LAN client propagate automatically.

The LAN client must use the host web URL shown by WEB HOST. A LAN client does not use its own localhost:11434 or localhost:8188 for generation.

## Setup

Open **SETTINGS** first:

1. Choose **OLLAMA** or **OPENAI COMPATIBLE**.
2. Set the base URL and press **GET MODELS**. For llama.cpp, the default shown is `http://127.0.0.1:8080/v1`.
3. Choose the model.
4. Choose your ComfyUI `models` directory.
5. Set the Raphael Model Manager / Registry root, or use the auto-discovered value when available.
6. Set the ComfyUI API URL, normally `http://127.0.0.1:8188`.
7. Scan the library, select a checkpoint card, enter any scene constraints and press **GENERATE**.

## Backend command/tool boundaries

- `scan_library` — ComfyUI file scan + Raphael metadata merge.
- `prepare_generation` — strict compatibility checking + random LoRA stack + random scene choices.
- `stream_llm` — provider-specific streaming adapter.
- `build_workflow` — deterministic standard ComfyUI API workflow construction.
- `inject_prompts` — deterministic replacement of prompt markers in the workflow.
- `submit_to_comfy` — actual ComfyUI `/prompt` submission.
- `append_history` / `load_history` — local generation archive.

The workflow builder is intentionally isolated behind a command boundary, so you can later replace the standard KSampler workflow with the exact deterministic workflow template you use in your ComfyUI installation without changing the UI or prompt engine.
