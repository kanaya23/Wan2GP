# Copilot Instructions for Wan2GP

## Build, test, and lint commands

### Environment setup / maintenance
- Interactive installer (GPU-aware package matrix from `setup_config.json`):
  - `python3 setup.py install --env venv`
  - `python3 setup.py install --env uv`
  - `python3 setup.py install --env conda`
- Common maintenance commands:
  - `python3 setup.py run`
  - `python3 setup.py update`
  - `python3 setup.py upgrade`
  - `python3 setup.py migrate`
  - `python3 setup.py status`
  - `python3 setup.py manage`
- Shell wrappers exist under `scripts/` (`install.sh`, `run.sh`, `update.sh`) and all `cd` to repo root before invoking `setup.py`.

### Run / headless execution
- Launch UI: `python wgp.py`
- Process saved queue headlessly: `python wgp.py --process path/to/queue.zip`
- Process exported settings JSON headlessly: `python wgp.py --process path/to/settings.json`
- Dry-run validation without generation: `python wgp.py --process path/to/queue.zip --dry-run`

### Single-test style command
- There is no centralized unit-test suite; use one-task queue validation as the practical single-run check:
  - `python wgp.py --process single_task_queue.zip --dry-run`

### Lint/build status
- No repository-level lint target (ruff/flake8/pylint config not present as a runnable project command).
- No dedicated build step beyond dependency installation and runtime startup.

## Deep architecture map

### 1) Bootstrap and runtime modes
- `setup.py` is an environment orchestrator (install/run/update/migrate/upgrade/status/manage), not a packaging-only setup script.
- Hardware detection + `setup_config.json` choose Python/Torch/Triton/Sage/Flash/kernel stack; `create_wgp_config()` seeds `wgp_config.json` profiles/attention/compile defaults.
- `wgp.py` has two execution modes:
  - **UI mode** (default): launches Gradio app.
  - **Headless mode** (`--process`): parses `.zip` queue or `.json` settings, optional `--dry-run`, then runs CLI worker.

### 2) `wgp.py` startup sequence (control flow)
- Parse CLI (`_parse_args`) and register family-specific LoRA CLI args via handlers.
- Load `models/_settings.json` (`primary_settings`), then load/create `wgp_config.json` (with normalization for profiles, output paths, MMAudio config).
- Configure checkpoint search roots through `shared/utils/files_locator.py` (`checkpoints_paths`).
- Build handler registry with `map_family_handlers()` from `family_handlers` list.
- Build model registry by loading `defaults/*.json` and `finetunes/*.json`, then calling `init_model_def()` (handler defaults + JSON overrides).
- Resolve active model, quantization/dtype/attention/profile policy, optional preload model (`preload_model_policy`).
- Initialize `WAN2GPApplication` and branch into CLI processing or Gradio UI launch.

### 3) Configuration layering and persistence
- Main config surfaces:
  - `setup_config.json` (install-time component matrix).
  - `wgp_config.json` (runtime/server behavior).
  - `models/_settings.json` (global base defaults).
  - `settings/*_settings.json` (per-model UI defaults).
  - `defaults/*.json` + `finetunes/*.json` (model definitions/capabilities).
  - `queue.zip` / `error_queue.zip` / settings JSON (task persistence).
- Effective task settings flow:
  - queue/settings file params
  - merged over `primary_settings`
  - normalized by `fix_settings()`
  - validated/overridden by `validate_settings()` before generation.

### 4) Model-family architecture
- Family handlers live under `models/**/**_handler.py` (Wan/Ovi/DF, Hunyuan, LTX Video, LTX2, LongCat, Flux, Qwen Image, Kandinsky5, Z-Image, TTS families).
- Handler contract is static-method oriented (`query_supported_types`, `query_model_def`, `query_model_files`, `register_lora_cli_args`, `get_lora_dir`, `load_model`, optional `custom_preprocess`, `validate_generative_*`, etc.).
- `map_family_handlers()` enforces one handler per base model type and collects compatibility maps (`models_eqv_map`, `models_comp_map`).
- Definition merge order for each model type:
  - handler `query_model_def()` defaults
  - JSON `model` object
  - top-level JSON settings attached as `model_def["settings"]`.

### 5) Model artifact resolution + downloads
- File selection uses `get_model_filename()` + quantization token routing (`int8`, `fp8`, bf16/fp16 markers, module URLs, recursive references).
- Local resolution uses `get_local_model_filename()` + `files_locator` search roots.
- `download_models()` handles:
  - shared preprocessing/postprocessing assets (pose, RAFT flow, depth-anything, SAM mask, wav2vec, pyannote, RIFE, etc.),
  - model weights (URL/hub/local),
  - text encoders (`text_encoder_URLs`),
  - preload/aux VAE URLs,
  - model-declared LoRAs,
  - handler-declared extra repos/files via `query_model_files()`.

### 6) Model loading and memory/offload stack
- `load_models()` decides quantization/dtype/profile/module list/text-encoder path, then calls handler `load_model(...)`.
- Handler returns model + pipe map; `init_pipe()` builds offload budgets/pinned-memory behavior by profile.
- `mmgp.offload.profile(...)` wraps runtime model residency/offloading/compile behavior.
- Runtime LM decoder engine is selected by `resolve_lm_decoder_engine()` (`legacy`/`cg`/`vllm`) with `vllm_support` capability probe.

### 7) Queue lifecycle (UI and CLI share the same core)
- UI queue entries are built by `add_video_task()` (params + thumbnails + `plugin_data`).
- Save/load:
  - `_save_queue_to_zip()` serializes `queue.json` + media attachments (`ATTACHMENT_KEYS`).
  - `_parse_queue_zip()` and `_parse_settings_json()` restore tasks.
  - `load_queue_action()` merges loaded tasks into active queue with ID remap.
- Processing:
  - `process_tasks()` async worker for UI streaming progress/preview/errors.
  - `process_tasks_cli()` for headless console processing.
  - Both execute `generate_video(...)` per task.

### 8) Validation gate before generation
- `validate_settings()` is the main canonical validator/normalizer:
  - prompt-template parsing,
  - model-specific prompt validation hooks,
  - custom setting parsing (`custom_settings`),
  - LoRA multiplier validation,
  - guide/audio/mask/sliding-window compatibility checks,
  - image/video/audio mode coercion.
- Returns normalized overrides used for generation and queue payload.

### 9) Generation pipeline internals (`generate_video`)
- Entry split:
  - `edit_*` modes route to `edit_video()` (postprocessing/remux pipeline),
  - normal modes perform diffusion/audio/image generation.
- Core stages:
  - model reload decision (model/profile/vae-upsampler changes),
  - attention backend selection + VAE tiling sizing,
  - LoRA resolution/loading/synchronization,
  - source/control/reference preprocessing,
  - sliding-window planning (overlap/reuse/discard logic),
  - `wan_model.generate(...)` call with all computed conditionings,
  - stitching/trimming/overlap handling,
  - postprocessing (RIFE temporal upsample, spatial upsample, film grain),
  - soundtrack handling (source audio mux, custom audio, MMAudio synthesis),
  - output write + metadata embedding + gallery/audio list updates.

### 10) Media technology map
- **Preprocessing (`preprocessing/`)**: DWPose, Canny, Scribble, RAFT optical flow, MiDaS/Depth-Anything depth, face extraction/ArcFace embeds, vocals extraction, speaker separation, matting/segmentation utilities.
- **Postprocessing (`postprocessing/`)**: RIFE interpolation, film grain, MMAudio generation stack (including vocoder/synchformer helpers).
- **Shared AV utilities (`shared/utils/audio_video.py`, metadata helpers)** handle muxing, codec output, metadata IO.

### 11) UI architecture (Gradio)
- `create_ui()` loads CSS/JS, appends plugin JS, creates base tabs, and mounts generated form from `generate_video_tab()`.
- `generate_video_tab()` is metadata-driven: many controls/visibility rules come from model definition flags (`model_def`).
- Two core tabs:
  - **Video Generator** (queue creation/execution),
  - **Edit** (edit queued task and reinsert).
- UI state uses `state["gen"]` as runtime queue/progress/output cache.

### 12) Plugin architecture
- Core API in `shared/utils/plugins.py`:
  - `WAN2GPPlugin` base class,
  - `PluginManager` discovery/install/update/uninstall/catalog,
  - `WAN2GPApplication` integration layer for tabs/events/component insertion.
- Extension points used in runtime:
  - `request_component`, `request_global`, `insert_after`,
  - custom JS injection via `add_custom_js`,
  - data hook `before_metadata_save` via `run_data_hooks(...)`.
- Task-scoped `plugin_data` is preserved in queue serialization and passed through generation/metadata flows.

### 13) Performance and kernel technologies
- Attention backends (`shared/attention.py`): `sdpa`, `flash`, `xformers`, `sage`, `sage2`, `sage3` (capability-filtered).
- Quantization/type handlers registered at startup: scaled FP8, NVFP4, Nunchaku int4/fp4, GGUF.
- Optional INT8 kernel injection (`shared/kernels/quanto_int8_inject`) controlled by `enable_int8_kernels`.
- Memory profiles (1/2/3/3.5/4/4.5/5) drive `mmgp` offload budget strategy.
- LM decoder runtime can switch between `legacy`, `cg`, and `vllm` engines depending on model support and runtime probe results.

## Key repository conventions

### Model-definition conventions
- Keep base definitions in `defaults/`; do not edit them for custom models.
- Put customizations in `finetunes/` using the same schema (`model` subtree + settings object).
- Overriding a default model is done by creating a `finetunes/<same_name>.json`; properties merge with finetune values taking priority.
- `URLs`, `URLs2`, `modules`, `loras`, and `preload_URLs` may reference another model id string, not only explicit URL lists.

### Quantized filename conventions (used by selector logic)
- File naming is parsed for quantization/type routing; use lower-case tokens consistently (`bf16`, `fp16`, `quanto`, `int8`, etc.).
- For finetunes, preserve the naming pattern described in `docs/FINETUNES.md`; selection logic in `get_model_filename()` depends on these tokens.

### Adding a new model family
- Implement a new handler under `models/.../*_handler.py` with the same static interface as existing handlers.
- Register it in the `family_handlers` list in `wgp.py`.
- Ensure family-specific LoRA CLI flags are added through `register_lora_cli_args` and resolved in `get_lora_dir`.

### Path + runtime expectations
- Most logic assumes execution from repo root (relative paths like `defaults/`, `finetunes/`, `settings/`, `plugins/`, `ckpts/`).
- Settings files are per model in `settings/*_settings.json`.
- Queue persistence uses `queue.zip` (or configured `--config` folder paths); headless processing reuses the same parsers as UI queue loading.

### Plugin conventions
- Plugin package lives under `plugins/<plugin_name>/` and must include `__init__.py` + `plugin.py` subclassing `WAN2GPPlugin`.
- Plugin UI integration relies on `request_component`, `request_global`, `insert_after`, and optional custom JS hooks.
