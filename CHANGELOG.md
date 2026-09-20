# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Gesture bundles: `GET /api/gestures/export`, `POST /api/gestures/import`, and
  `jepa gestures list|export|import|match|remove` — tune on one machine, deploy on many.
- `{ } API` controls throughout the testbench showing the exact request (curl / JS / Python)
  behind each action, and an Integration page with a quick-start example.
- Header status bar (model, checkpoint coverage, camera, latency); `camera_active` on `/api/status`.
- Event console in Live (filter errors/detections, copy).
- `jepa tags --json`; logs on stderr so CLI output pipes cleanly.

### Changed
- Testbench ergonomics for desktop/integrator use: English everywhere, honest section titles,
  Workspace/Admin navigation with keyboard support (`role="tab"`, arrows), toasts and accessible
  dialogs instead of `alert()`/`confirm()`, focus rings, WCAG AA contrast tokens, no horizontal
  overflow down to 1100 px, larger click targets. Desktop window titled with the version.

## [0.2.0] — 2026-09-20

### Added
- **Explainable few-shot gestures** (`src/gestures.rs`): multi-sample prototypes, neutral pose,
  contrastive scoring, runner-up margin, per-frame decision trace and per-patch difference map.
  Registry persisted per model in `~/.jepa/gestures.json`.
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

## [0.1.0] — 2026-09-19

Initial import: candle ViT runtime, Hugging Face pull, REST/SSE daemon, embedded testbench,
camera ring buffer, bearer-token auth, packaging scripts.
