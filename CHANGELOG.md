# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Settings: **Allow access from other machines** toggle (off by default). Off binds `127.0.0.1`, on binds `0.0.0.0` on the next restart; authentication stays mandatory for network access and an explicit `--host` still overrides it.

### Fixed
- Settings save no longer fails when a numeric field is blank (the idle-timeout default 15 was not a dropdown option); added the 15-minute option and coerced blank values.

### Security
- Decoders capped (8192 px per side, 256 MB decoder allocations, 256 million pixels per animation): a decompression bomb is refused before allocation.
- Model identifiers that resolve to the models directory itself (empty, `/`, `.`) are refused, so a delete cannot wipe the library.
- SECURITY.md documents what an API client can and cannot do to the host.
- `keys.json` stores SHA-256 digests of API keys instead of the raw tokens (legacy files migrate on first start); digests are never returned by `GET /api/keys`.

### Changed
- Rust edition 2024 with `rust-version = "1.88"` (the oldest toolchain that has every std API the code uses). Nested `if let` chains became let chains; no behaviour change.

## [0.3.0]: 2026-09-20

### Added
- **Companion**: a second virtual WebGL robot that learns from the person in front of it. Head follows motion, taught poses are mirrored by similarity, taught sounds and poses trigger behaviours (nod, shake, wave, cheer, dance, startle, sleep). `src/companion/`, `/api/companion/*`, persisted cues in `~/.jepctl/companion.json`.
- **Microphone capture** (`cpal`) with a 12 s ring, `/api/mic/*`, and **sounds**: few-shot audio prototypes (`/api/sounds`, `~/.jepctl/sounds.json`) matched like gestures.
- **Audio slot**: an audio model loads next to the vision model instead of replacing it; `/api/status` reports `audio_model` and `audio_weights`; `POST /api/models/unload` takes an optional `model_name`.
- Companion feedback: live Eyes (model view) and Ears (waveform, `GET /api/mic/waveform`) panels, a lesson card with the captured frame or clip, and an acknowledge reaction on every lesson.
- **Mirror mode** for the arm: "Teach this pose" registers the camera view and maps it to the current joints; in Mirror mode every taught pose is blended by match score so the arm follows between them. Telemetry carries `mirror` weights.

### Fixed
- Inference no longer records an autograd graph: models are rebuilt on detached weights after loading. A V-JEPA 2 clip forward peaked at 13 GB of GPU buffers (and froze the machine during gesture capture); it now peaks under 2 GB.
- Embedding requests are serialized on one inference lock, so concurrent captures and the stream cannot multiply the working set.
- Status on Metal can report live GPU allocation (`gpu_allocated_bytes` helper).

### Added
- **Robot Twin**: `src/robot/` hardware abstraction layer (virtual arm feeding a raw WebGL
  twin, serial backend behind `--features serial` with Feetech STS style frames), safety
  guard (joint limits, 1.5 rad/s ramp, E-stop), 30 Hz controller with manual, gesture
  shadowing (Mode A), safety gate (Mode B), a Learn mode and goal seeking (Mode C);
  `/api/robot/*` endpoints and a 30 Hz WebSocket; new Robot Twin tab.
- **Latent world model learned from the camera**: transitions `(z, action, z next)` from
  JEPA embeddings of camera frames train a ridge dynamics model in embedding space; Mode C
  plans inside it (96 candidates per step), persisted in `~/.jepctl/robot_world_model.json`.
- **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`): 3D tubelet embedding, 3D rotary attention,
  encoder-only strict loading (388/388), per-head attention to bound memory (2.5 GB peak vs.
  70 GB with full score tensors). Camera stream and gestures embed a 16-frame clip.
- **Audio modality** with AudioMAE (`gaunernst/vit_base_patch16_1024_128.audiomae_as2m`):
  Kaldi-style log-mel front-end, WAV decoding, rectangular single-channel ViT, mean pooling.
- **Clip and audio files** in `POST /api/embed`, the Embed tab and `jepctl embed`: GIF / animated
  WebP / WAV natively, MP4 / WebM / MP3 / FLAC / OGG through `ffmpeg` when present.
- **Region of interest**: `GET/PUT/DELETE /api/camera/roi`, editor in the Gestures tab, applied
  to every camera embedding and carried in gesture bundles.
- Manifest fields `tubelet_size`, `input_width`, `in_chans`, `audio`, `pooling`.
- Gesture bundles: `GET /api/gestures/export`, `POST /api/gestures/import`, and
  `jepctl gestures list|export|import|match|remove`: tune on one machine, deploy on many.
- `{ } API` controls throughout the testbench showing the exact request (curl / JS / Python)
  behind each action, and an Integration page with a quick-start example.
- Header status bar (model, checkpoint coverage, camera, latency); `camera_active` on `/api/status`.
- Event console in Live (filter errors/detections, copy).
- `jepctl tags --json`; logs on stderr so CLI output pipes cleanly.

### Changed
- Testbench ergonomics for desktop/integrator use: English everywhere, honest section titles,
  Workspace/Admin navigation with keyboard support (`role="tab"`, arrows), toasts and accessible
  dialogs instead of `alert()`/`confirm()`, focus rings, WCAG AA contrast tokens, no horizontal
  overflow down to 1100 px, larger click targets. Desktop window titled with the version.

## [0.2.0]: 2026-09-20

### Added
- **Explainable few-shot gestures** (`src/gestures.rs`): multi-sample prototypes, neutral pose,
  contrastive scoring, runner-up margin, per-frame decision trace and per-patch difference map.
  Registry persisted per model in `~/.jepctl/gestures.json`.
- **Model view**: `GET /api/camera/frame` returns exactly what the network receives; the
  sandbox shows it with a heatmap and a reasoning table.
- **Backbone variants** (`plain` / `cls` / `dinov2`) with LayerScale, CLS pooling, fused-QKV
  loading and positional-embedding resampling. DINOv2 and timm ViTs now load at 100 %.
- `WeightReport` on `/api/status` and `/api/models/load`; `422` on unsupported checkpoints.
- Per-model preprocessing (`image_size`, `normalization`) threaded through every entry point.
- `--cors-origins` / `JEPA_CORS_ORIGINS`; `?token=` on the SSE stream.
- HTTP integration tests, CI on three OSes, release workflow, issue/PR templates,
  CONTRIBUTING / SECURITY / ARCHITECTURE docs.

### Changed
- Gesture registration and live matching both embed the **server** camera frame through one
  pipeline (previously browser pixels vs. server pixels).
- `default` Cargo features are now empty; pass `--features metal|cuda` explicitly.
- Catalog trimmed to checkpoints that verifiably load; `facebook/ijepa_vitb16_1k` and
  `facebookresearch/jepa:vjepa_vitl16` (non-existent on the Hub), BEiT and SigLIP
  (unsupported architectures) removed.
- CORS is same-origin by default; camera, ring buffer, stream and gesture endpoints require a token.

### Fixed
- Models could silently run on random weights when a checkpoint was missing or its layout unknown.
- HF ViT positional embeddings (with CLS row) were rejected and replaced by sin-cos.
- V-JEPA path ignored positional embeddings.
- `POST /api/embed` embedded the camera ring buffer instead of an uploaded video (now `501`).
- Gesture thumbnails never displayed (`thumbnail_base64` vs `thumbnail`).
- Starting the stream blocked on the browser camera permission prompt.
- `/api/auth/token` was readable cross-origin (token theft from any web page).

## [0.1.0]: 2026-09-19

Initial import: candle ViT runtime, Hugging Face pull, REST/SSE daemon, embedded testbench,
camera ring buffer, bearer-token auth, packaging scripts.
