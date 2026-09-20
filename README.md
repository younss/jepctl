# jepa — local runtime for JEPA-style vision encoders

[![CI](https://github.com/younss/jepctl/actions/workflows/ci.yml/badge.svg)](https://github.com/younss/jepctl/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`jepa` is to **representation-learning encoders** what Ollama is to LLMs: a single pure-Rust binary that pulls a checkpoint from Hugging Face, runs it locally (Metal, CUDA or CPU), and exposes it through a CLI, a REST/SSE API and an embedded web testbench.

It targets *non-generative* encoders — **I-JEPA**, **V-JEPA 2**, **DINOv2**, plain **ViT**, **AudioMAE** — whose output is an embedding, not text: images, video clips and audio. On top of raw embeddings it ships an explainable **few-shot gesture sandbox**: register a few reference poses from your webcam, then watch, frame by frame, *why* the model does or does not recognise them.

> **Status:** early (0.3). Every model in the catalog loads with verified checkpoint coverage and every entry point is tested; see [Verification](#verification) for what is and is not checked numerically.

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
| `facebook/vjepa2-vitl-fpc64-256` | **V-JEPA 2** ViT-L/16 (video) | 1024 | 300M | 16 frames × 256², ImageNet norm | mean of space-time tokens | 3D tubelets + 3D RoPE; ~0.9 s / clip on Apple Silicon, ≈10 s on CPU |
| `gaunernst/vit_base_patch16_1024_128.audiomae_as2m` | **AudioMAE** ViT-B/16 (audio) | 768 | 86M | 10.24 s log-mel 1024×128 @ 16 kHz | mean of patches | Kaldi fbank front-end built in; CC-BY-4.0 |
| `facebook/ijepa_vith14_1k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | the reference JEPA encoder |
| `facebook/ijepa_vith14_22k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | IN-22k pre-training |
| `facebook/dinov2-small` | DINOv2 ViT-S/14 | 384 | 22M | 224, ImageNet norm | CLS | **best default for gestures**: fast and very discriminative |
| `facebook/dinov2-base` | DINOv2 ViT-B/14 | 768 | 86M | 224, ImageNet norm | CLS | |
| `google/vit-base-patch16-224` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | supervised IN-21k→1k |
| `timm/vit_base_patch16_224.augreg_in21k` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | fused-QKV timm layout |

DINOv2 checkpoints ship a 37×37 positional grid (518 px); it is bicubically resampled to the 16×16 grid used at 224 px, as in the reference implementation. V-JEPA 2's predictor is ignored (encoder only). AudioMAE pools by **mean**: its CLS token was never trained as a summary and yields identical vectors for every input.

Anything else can be tried through a custom [Jepafile](#jepafile-custom-models); if its layout is not one of the five the loader understands (HF ViT/I-JEPA, HF DINOv2, HF V-JEPA 2, timm/Meta fused-QKV, AudioMAE), loading fails with the list of missing tensors.

### Verification

Each catalog entry was pulled and loaded (`weights.loaded == weights.expected`), embedded twice for determinism, and compared across clearly different inputs. Implementations follow the reference code line by line (see `docs/ARCHITECTURE.md`), including V-JEPA 2's tiled-sin/interleaved-rotation quirk and Kaldi's Povey window. What has **not** been done: a bit-for-bit comparison with PyTorch outputs, because this repository has no Python dependency. If you run one, please open an issue with the numbers.

### Inputs

| Kind | Formats | Notes |
|---|---|---|
| Image | PNG, JPEG, WebP | centre crop + resize to the model input |
| Clip | GIF, animated WebP natively; MP4/WebM/MOV with `ffmpeg` on `PATH` (or `JEPA_FFMPEG`) | uniformly sub-sampled to the model's clip length (16 for V-JEPA 2); image models embed each frame and average |
| Audio | WAV natively; MP3/FLAC/OGG with `ffmpeg` | any sample rate, down-mixed to mono, resampled to 16 kHz, 10.24 s window (zero-padded / cropped) |
| Camera | server camera via `nokhwa` | optional **region of interest** crop, see Gestures |

---

## CLI

```bash
jepa serve [--host 127.0.0.1] [--port 11435] [--no-auth] [--device auto|metal|cuda|cpu] [--cors-origins a,b]
jepa app | jepa gui                 # daemon + native desktop window
jepa run <model>                    # daemon with a model preloaded
jepa pull <hf-repo>                 # download model.safetensors + Jepafile.json into ~/.jepa/models
jepa tags | jepa list               # installed models
jepa rm <model>
jepa embed <image|clip|audio> [--model m] [--format json|raw]
jepa stream [--camera 0] [--fps 10] [--model m]
jepa key generate --name "CI" --role admin|inference [--days 90]
jepa key list | jepa key revoke <prefix>
jepa gestures list [--json]
jepa gestures export [--model m] [-o bundle.json] [--threshold 0.7] [--margin 0.04] [--no-thumbnails]
jepa gestures import bundle.json [--replace]
jepa gestures match photo.jpg [--model m] [--threshold] [--margin]   # exit 0 detected, 1 not detected
jepa gestures remove <name> | --model m
```

Logs go to stderr, so `jepa tags --json | jq` works. Gesture commands read and write the same `~/.jepa/gestures.json` as the GUI and the API (stop the daemon before `import`, or use the API).

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
| POST | `/api/embed` | inference | multipart `file`: image, clip (GIF/WebP/MP4/WebM) or audio (WAV/MP3/FLAC/OGG) → `{ model, dimension, latency_ms, embedding, patch_embeddings }`. The active model must match the modality (`400` otherwise). |
| GET | `/api/embed/stream?fps=10&threshold=0.70&margin=0.04[&token=]` | inference | SSE, one event per new camera frame: `{ frame_index, model, latency_ms, embedding, gesture_match? }`. Failures arrive as `event: error`. `token=` exists because `EventSource` cannot set headers. |
| POST | `/api/energy` | inference | `{ vector1, vector2, threshold }` → L2 / cosine distance, anomaly flag |

### Camera

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/cameras` | – | detected devices |
| POST | `/api/camera/start?device=0&fps=10` | inference | |
| POST | `/api/camera/stop` | inference | |
| GET | `/api/camera/frame` | inference | JPEG of **exactly what the model receives** (centre crop, model input size); `X-Frame-Sequence` header |
| GET | `/api/camera/frame?full=true` | inference | full downscaled frame for the ROI editor (`X-Frame-Width/Height`) |
| GET / PUT / DELETE | `/api/camera/roi` | inference | region of interest `{ x, y, w, h }` normalised on the raw frame; applied before every camera embedding and stored in exported bundles |
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
| GET | `/api/gestures/export[?model=…\|?all=true][&threshold=&margin=&thumbnails=false]` | inference | portable bundle `{ version, threshold, margin, gestures[] }` — train on one machine, deploy on many |
| POST | `/api/gestures/import[?replace=true]` | inference | load a bundle; `replace` removes existing gestures of the bundle's models first |

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

## Desktop app / testbench

`jepa app` opens the testbench in a native window; `jepa serve` serves the same page at `http://127.0.0.1:11435/`. No build step, no external assets, keyboard-navigable (↑/↓ between sections, Esc closes dialogs).

The header is the single source of truth: **model · checkpoint coverage · camera · last latency**.

**Workspace**
- **Overview** — latency, FPS, embeddings count, hardware telemetry.
- **Models** — verified catalog (served by `/api/catalog`), pull with progress, load/unload, Jepafile editor.
- **Embed** — drop an image, see the patch grid and the pooled vector.
- **Live** — server camera, ring-buffer scrubber, and an **event console** (filter errors / detections, copy a line).
- **Energy** — lock a baseline, chart the drift, alerts and webhook.
- **Gestures** — *model view* (the exact frame the network receives) with a per-patch difference heatmap; a **region-of-interest editor** (drag on the full frame to crop around the hand); 3 gesture slots + 1 neutral slot; a per-frame **"Why this decision"** table (raw / contrastive / combined per gesture, margin, threshold, server and client decisions with their reasons); threshold, margin and smoothing; **Export / Import bundle** (carries threshold, margin and ROI). Works with image models (latest frame) and V-JEPA 2 (16-frame clip).

**Admin**
- **Integration & Keys** — base URL, auth mode, active model, a quick-start example in curl / JavaScript / Python, scoped tokens, audit trail.
- **Settings** — backend override, memory watermark, storage paths.

Every action button has a **`{ } API`** control that shows the exact request the app sends — with the threshold and margin you just tuned — in curl, JavaScript or Python, ready to paste into your application.

Tips for good detections: register the neutral pose first, take 3–5 samples per gesture while moving slightly, keep the hand large in the frame, and prefer `facebook/dinov2-small`.

---

## Jepafile (custom models)

`~/.jepa/models/<org>/<name>/Jepafile.json` next to `model.safetensors`:

```json
{
  "name": "facebook/dinov2-large",
  "repo_id": "facebook/dinov2-large",
  "architecture": "DINOv2 ViT-L/14",
  "modality": "image",            // image | video | audio
  "patch_size": 14, "embed_dim": 1024, "num_layers": 24, "num_heads": 16, "image_size": 224,
  "frames": null,                 // video: frames per clip (V-JEPA 2: 16)
  "tubelet_size": null,           // video: temporal patch (V-JEPA 2: 2)
  "input_width": null,            // audio: mel bins when the input is not square (128)
  "in_chans": null,               // audio: 1
  "audio": null,                  // audio: { sample_rate, n_mels, frames, mean, std }
  "variant": "dinov2",            // plain | cls | dinov2 | vjepa2 (inferred from the name if omitted)
  "pooling": "cls",               // cls | mean (inferred; MAE-style encoders need mean)
  "normalization": "imagenet",    // imagenet | inception (inferred if omitted)
  "mlp_ratio": 4.0
}
```

Register it with `POST /api/manifests` or the Models tab, then `jepa pull` / load. `image_size` (and `input_width`) must be multiples of `patch_size`; square positional grids are resampled if the checkpoint was trained at another resolution.

---

## Architecture

```
src/
├── main.rs              CLI (clap) and daemon bootstrap
├── engine/              candle: vit.rs (2D backbone, weight mapping), ijepa.rs, vjepa2.rs (3D tubelets + RoPE), audio.rs, device.rs
├── gestures.rs          prototypes, contrastive matching, decision trace, persistence
├── hub/                 verified catalog, Jepafile schema, safetensors downloader
├── media/               image/video/audio decoding & preprocessing, camera capture (nokhwa), ring buffer + "model view" + ROI
├── server/              axum routes, handlers, gesture handlers, auth middleware, integration tests
├── auth.rs / config.rs  bearer tokens + RBAC, ~/.jepa layout
└── ui/                  embedded single-page testbench (vanilla JS, no build)
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the data flow, the weight-loading contract and why the gesture pipeline is built the way it is.

---

## Roadmap

- **Audio-JEPA**: the audio modality is in place (front-end, rectangular ViT, catalog), but no Audio-JEPA checkpoint is published in a loadable format today (`ltuncay/Audio-JEPA` is a PyTorch Lightning pickle). AudioMAE is the verified audio encoder; an Audio-JEPA safetensors release would be a manifest entry away.
- Numerical parity tests against PyTorch reference outputs (needs a contributor with a Python environment — see Verification).
- Larger V-JEPA 2 variants (`vith`, `vitg`) once someone can verify memory/latency on their hardware.
- Native H.264 decoding without `ffmpeg`, if a dependable pure-Rust decoder appears.
- Microphone capture for live audio, mirroring the camera pipeline.

---

## Contributing

Bug reports, model requests and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow (`cargo fmt`, `clippy -D warnings`, `cargo test`, and how to verify a checkpoint really loads). Security issues: [SECURITY.md](SECURITY.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Model weights keep their own licenses (I-JEPA and DINOv2: CC-BY-NC 4.0 / Apache-2.0 respectively — check each Hugging Face card).
