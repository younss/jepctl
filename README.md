# jepa — local runtime for JEPA-style vision encoders

[![CI](https://github.com/younss/jepctl/actions/workflows/ci.yml/badge.svg)](https://github.com/younss/jepctl/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`jepa` is to **representation-learning encoders** what Ollama is to LLMs: a single pure-Rust binary that pulls a checkpoint from Hugging Face, runs it locally (Metal, CUDA or CPU), and exposes it through a CLI, a REST/SSE API and an embedded web testbench.

It targets *non-generative* vision models — **I-JEPA**, **DINOv2**, plain **ViT** — whose output is an embedding, not text. On top of raw embeddings it ships an explainable **few-shot gesture sandbox**: register a few reference poses from your webcam, then watch, frame by frame, *why* the model does or does not recognise them.

> **Status:** early (0.2). The engine, catalog, API and gesture pipeline are tested and honest about what they can do. Video encoders (V-JEPA) are not supported yet — see [Roadmap](#roadmap).

---

## Quick start

```bash
# Build (pick your accelerator; none = multithreaded CPU)
cargo build --release --features metal   # Apple Silicon
cargo build --release --features cuda    # NVIDIA
cargo build --release                    # CPU only

# Pull a verified model and start the daemon + testbench
./target/release/jepa pull facebook/dinov2-small
./target/release/jepa serve --no-auth      # http://127.0.0.1:11435
```

Open the testbench, go to **8. Gesture Sandbox**, start the camera, register a *Neutral* pose and two or three gestures with 3–5 samples each, and read the reasoning table.

### Requirements

- A recent stable Rust (`rust-toolchain.toml` selects stable; CI tracks it).
- macOS: Xcode command-line tools. Linux: `libgtk-3-dev libwebkit2gtk-4.1-dev libxdo-dev libayatana-appindicator3-dev` (desktop window) — the daemon itself needs nothing. Windows: MSVC build tools + WebView2 (preinstalled on Windows 11).
- A camera is optional: without one the daemon serves a synthetic test pattern so every code path still runs.

---

## Verified model catalog

Every entry below has been loaded end-to-end with **100 % checkpoint coverage** (`GET /api/status → weights.loaded == weights.expected`). A model that does not fully load is refused — `jepa` never runs on random weights.

| Model | Family | Dim | Params | Input | Pooled output | Notes |
|---|---|---|---|---|---|---|
| `facebook/ijepa_vith14_1k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | the reference JEPA encoder |
| `facebook/ijepa_vith14_22k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | IN-22k pre-training |
| `facebook/dinov2-small` | DINOv2 ViT-S/14 | 384 | 22M | 224, ImageNet norm | CLS | **best default for gestures**: fast and very discriminative |
| `facebook/dinov2-base` | DINOv2 ViT-B/14 | 768 | 86M | 224, ImageNet norm | CLS | |
| `google/vit-base-patch16-224` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | supervised IN-21k→1k |
| `timm/vit_base_patch16_224.augreg_in21k` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | fused-QKV timm layout |

DINOv2 checkpoints ship a 37×37 positional grid (518 px); it is bicubically resampled to the 16×16 grid used at 224 px, as in the reference implementation.

Anything else can be tried through a custom [Jepafile](#jepafile-custom-models); if its layout is not one of the four the loader understands (HF ViT/I-JEPA, HF DINOv2, timm/Meta fused-QKV), loading fails with the list of missing tensors.

---

## CLI

```bash
jepa serve [--host 127.0.0.1] [--port 11435] [--no-auth] [--device auto|metal|cuda|cpu] [--cors-origins a,b]
jepa app | jepa gui                 # daemon + native desktop window
jepa run <model>                    # daemon with a model preloaded
jepa pull <hf-repo>                 # download model.safetensors + Jepafile.json into ~/.jepa/models
jepa tags | jepa list               # installed models
jepa rm <model>
jepa embed <image> [--model m] [--format json|raw]
jepa stream [--camera 0] [--fps 10] [--model m]
jepa key generate --name "CI" --role admin|inference [--days 90]
jepa key list | jepa key revoke <prefix>
```

`--no-auth` is refused on any host other than `127.0.0.1`/`localhost`.

---

## REST & SSE API

All endpoints live under `/api`. With authentication enabled (the default) send `Authorization: Bearer <token>`; the admin token is printed at first start and stored in `~/.jepa/auth.token`. Roles: `admin` (models, keys, settings) and `inference` (everything else).

### System & catalog

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/status` | – | health, backend, memory, `active_model`, **`weights {loaded, expected, source}`** |
| GET | `/api/tags` | – | installed models |
| GET | `/api/catalog` | – | verified catalog (what the Models tab shows) |
| POST | `/api/models/load` | admin | `{ "model_name" }` → `{ weights }`; `409` not downloaded, `422` checkpoint layout unsupported |
| POST | `/api/models/unload` | admin | |
| DELETE | `/api/models/{name}` | admin | |
| POST | `/api/pull` | admin | `{ "repo_id" }`, streamed JSON progress |
| POST | `/api/manifests` | admin | register a custom Jepafile |

### Inference

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/embed` | inference | multipart `file` (PNG/JPEG/WebP) → `{ model, dimension, latency_ms, embedding, patch_embeddings }`. Video files: `501`. |
| GET | `/api/embed/stream?fps=10&threshold=0.70&margin=0.04[&token=]` | inference | SSE, one event per new camera frame: `{ frame_index, model, latency_ms, embedding, gesture_match? }`. Failures arrive as `event: error`. `token=` exists because `EventSource` cannot set headers. |
| POST | `/api/energy` | inference | `{ vector1, vector2, threshold }` → L2 / cosine distance, anomaly flag |

### Camera

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/cameras` | – | detected devices |
| POST | `/api/camera/start?device=0&fps=10` | inference | |
| POST | `/api/camera/stop` | inference | |
| GET | `/api/camera/frame` | inference | JPEG of **exactly what the model receives** (centre crop, model input size); `X-Frame-Sequence` header |
| GET | `/api/ring-buffer` | inference | last 16 frame thumbnails |

### Few-shot gestures

A gesture is a *prototype*: the L2-normalised mean of one or more reference embeddings, bound to the model that produced them. The registry persists in `~/.jepa/gestures.json`.

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/gestures` | inference | `{ "name", "is_neutral"?, "from_camera": true }` embeds the current camera frame and **adds a sample** (call again for more). Alternatives: `"embedding": [...]` or `"image_base64": "data:image/jpeg;base64,..."`. |
| GET | `/api/gestures[?model=…\|?all=true]` | inference | gestures of the active model |
| DELETE | `/api/gestures/{name}` | inference | |
| DELETE | `/api/gestures[?all=true]` | inference | clear the active model's gestures |
| POST | `/api/gestures/match` | inference | `{ "from_camera": true \| "embedding" \| "image_base64", "threshold"?, "margin"? }` → decision trace below |

```json
{
  "matched": "Open Hand", "detected": true, "confidence": 0.81, "margin": 0.23,
  "threshold": 0.70, "margin_required": 0.04, "method": "contrastive",
  "reason": "'Open Hand' scored 0.81 with margin 0.230",
  "scores": [
    { "name": "Open Hand", "is_neutral": false, "raw_cosine": 0.99, "contrastive": 0.72, "combined": 0.81, "sample_count": 4 },
    { "name": "Rest",      "is_neutral": true,  "raw_cosine": 0.98, "contrastive": 0.21, "combined": 0.48, "sample_count": 3 }
  ],
  "patch_diff": [0.01, 0.02, "…"], "grid_size": 16
}
```

**How the decision is made.** `raw_cosine` is the naive similarity to the prototype; with the same background, face and lighting it saturates near 1.0 for *every* pose and discriminates nothing. `contrastive` is the cosine after removing the centroid of all prototypes — the shared component — leaving the gesture's own signature. `combined = 0.35·raw + 0.65·max(contrastive, 0)`. A detection requires `combined ≥ threshold`, a lead of `margin_required` over the runner-up, and that the best candidate is not a neutral pose. `patch_diff` is the per-patch dissimilarity to the best prototype over the ViT grid, for the heatmap.

### Security & settings

`POST/GET /api/keys`, `DELETE /api/keys/{prefix}`, `GET /api/audit` (admin); `GET/POST /api/settings`; `GET /api/auth/token` (same-origin testbench bootstrap only).

The API is **same-origin by default**: no CORS headers are sent, so a web page from another origin cannot read responses. Pass `--cors-origins http://localhost:5173` (or `JEPA_CORS_ORIGINS`) to allow a browser app. See [SECURITY.md](SECURITY.md).

---

## Web testbench

Served from the binary at `http://127.0.0.1:11435/` — no build step, no external assets.

1. **Overview** — latency, FPS, embeddings count, hardware telemetry.
2. **Models** — catalog, pull with progress, load/unload, delete, Jepafile editor.
3. **Image Playground** — drop an image, see the patch grid and the pooled vector.
4. **Video & Camera** — server camera control, ring-buffer scrubber, SSE log.
5. **Anomaly & Energy** — lock a baseline, chart the drift, alerts and webhook.
6. **Security & Keys** — scoped tokens, audit trail.
7. **Settings** — backend override, memory watermark, storage paths.
8. **Gesture Sandbox** — *model view* (the exact frame the network receives) with a per-patch difference heatmap; 3 gesture slots + 1 neutral slot fed by the server camera; a per-frame **reasoning table** (raw / contrastive / combined per gesture, margin, threshold, server and client decisions with their reasons); threshold, margin and smoothing controls; audio/theme/hold actions.

Tips for good detections: register the neutral pose first, take 3–5 samples per gesture while moving slightly, keep the hand large in the frame, and prefer `facebook/dinov2-small`.

---

## Jepafile (custom models)

`~/.jepa/models/<org>/<name>/Jepafile.json` next to `model.safetensors`:

```json
{
  "name": "facebook/dinov2-large",
  "repo_id": "facebook/dinov2-large",
  "architecture": "DINOv2 ViT-L/14",
  "modality": "image",
  "patch_size": 14, "embed_dim": 1024, "num_layers": 24, "num_heads": 16, "image_size": 224,
  "frames": null,
  "variant": "dinov2",          // plain | cls | dinov2   (inferred from the name if omitted)
  "normalization": "imagenet",  // imagenet | inception    (inferred if omitted)
  "mlp_ratio": 4.0
}
```

Register it with `POST /api/manifests` or the Models tab, then `jepa pull` / load. `image_size` must be a multiple of `patch_size`; positional embeddings are resampled if the checkpoint was trained at another resolution.

---

## Architecture

```
src/
├── main.rs              CLI (clap) and daemon bootstrap
├── engine/              candle backbone: vit.rs (variants, weight mapping), ijepa.rs, vjepa.rs, device.rs
├── gestures.rs          prototypes, contrastive matching, decision trace, persistence
├── hub/                 verified catalog, Jepafile schema, safetensors downloader
├── media/               image preprocessing, camera capture (nokhwa), ring buffer + "model view"
├── server/              axum routes, handlers, gesture handlers, auth middleware, integration tests
├── auth.rs / config.rs  bearer tokens + RBAC, ~/.jepa layout
└── ui/                  embedded single-page testbench (vanilla JS, no build)
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the data flow, the weight-loading contract and why the gesture pipeline is built the way it is.

---

## Roadmap

- **V-JEPA / V-JEPA 2**: the current `VJepaModel` is a frame-wise ViT with mean pooling and cannot load Meta's checkpoints (3D tubelet embedding, RoPE). No verified video model is offered until it can. Contributions welcome — start from `docs/ARCHITECTURE.md § Adding a model family`.
- Region of interest for gestures (crop around the hand before embedding).
- `jepa embed` for video files.
- Audio-JEPA.

---

## Contributing

Bug reports, model requests and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow (`cargo fmt`, `clippy -D warnings`, `cargo test`, and how to verify a checkpoint really loads). Security issues: [SECURITY.md](SECURITY.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Model weights keep their own licenses (I-JEPA and DINOv2: CC-BY-NC 4.0 / Apache-2.0 respectively — check each Hugging Face card).
