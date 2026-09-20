# jepctl: local runtime for JEPA style encoders

[![CI](https://github.com/younss/jepctl/actions/workflows/ci.yml/badge.svg)](https://github.com/younss/jepctl/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`jepctl` is to **representation-learning encoders** what Ollama is to LLMs: a single pure-Rust binary that pulls a checkpoint from Hugging Face, runs it locally (Metal, CUDA or CPU), and exposes it through a CLI, a REST/SSE API and an embedded desktop testbench.

It targets *non-generative* encoders whose output is an embedding, not text: **I-JEPA**, **V-JEPA 2**, **DINOv2**, plain **ViT** for images and video, **AudioMAE** for audio. On top of raw embeddings it ships:

- an explainable **few-shot gesture sandbox**: register a few reference poses from your webcam, then watch, frame by frame, *why* the model does or does not recognise them;
- a **robot twin**: a 6 DOF arm that learns a latent world model from what the camera sees and plans inside it, previewed in WebGL and gated before it touches hardware.

> **Status:** early (0.2). Every model in the catalog loads with verified checkpoint coverage and every entry point is tested; see [Verification](#verification) for what is and is not checked numerically.

---

## Quick start

```bash
# Build (pick your accelerator; none = multithreaded CPU)
cargo build --release --features metal   # Apple Silicon
cargo build --release --features cuda    # NVIDIA
cargo build --release                    # CPU only

# Pull a verified model and start the daemon + testbench
./target/release/jepctl pull facebook/dinov2-small
./target/release/jepctl serve --no-auth      # http://127.0.0.1:11435
```

Open the testbench, go to **8. Gesture Sandbox**, start the camera, register a *Neutral* pose and two or three gestures with 3-5 samples each, and read the reasoning table.

### Requirements

- A recent stable Rust (`rust-toolchain.toml` selects stable; CI tracks it).
- macOS: Xcode command-line tools. Linux: `libgtk-3-dev libwebkit2gtk-4.1-dev libxdo-dev libayatana-appindicator3-dev` (desktop window): the daemon itself needs nothing. Windows: MSVC build tools + WebView2 (preinstalled on Windows 11).
- A camera is optional: without one the daemon serves a synthetic test pattern so every code path still runs.

---

## Verified model catalog

Every entry below has been loaded end-to-end with **100 % checkpoint coverage** (`GET /api/status → weights.loaded == weights.expected`). A model that does not fully load is refused: `jepctl` never runs on random weights.

| Model | Family | Dim | Params | Input | Pooled output | Notes |
|---|---|---|---|---|---|---|
| `facebook/vjepa2-vitl-fpc64-256` | **V-JEPA 2** ViT-L/16 (video) | 1024 | 300M | 16 frames x 256 px, ImageNet norm | mean of space-time tokens | 3D tubelets + 3D RoPE; about 0.9 s per clip on Apple Silicon, about 10 s on CPU |
| `gaunernst/vit_base_patch16_1024_128.audiomae_as2m` | **AudioMAE** ViT-B/16 (audio) | 768 | 86M | 10.24 s log-mel 1024x128 @ 16 kHz | mean of patches | Kaldi fbank front-end built in; CC-BY-4.0 |
| `facebook/ijepa_vith14_1k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | the reference JEPA encoder |
| `facebook/ijepa_vith14_22k` | I-JEPA ViT-H/14 | 1280 | 632M | 224, ImageNet norm | mean of patches | IN-22k pre-training |
| `facebook/dinov2-small` | DINOv2 ViT-S/14 | 384 | 22M | 224, ImageNet norm | CLS | **best default for gestures**: fast and very discriminative |
| `facebook/dinov2-base` | DINOv2 ViT-B/14 | 768 | 86M | 224, ImageNet norm | CLS | |
| `google/vit-base-patch16-224` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | supervised IN-21k→1k |
| `timm/vit_base_patch16_224.augreg_in21k` | ViT-B/16 | 768 | 86M | 224, 0.5/0.5 norm | CLS | fused-QKV timm layout |

DINOv2 checkpoints ship a 37×37 positional grid (518 px); it is bicubically resampled to the 16×16 grid used at 224 px, as in the reference implementation.

V-JEPA 2's predictor is ignored (encoder only). AudioMAE pools by **mean**: its CLS token was never trained as a summary and yields identical vectors for every input.

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
jepctl serve [--host 127.0.0.1] [--port 11435] [--no-auth] [--device auto|metal|cuda|cpu] [--cors-origins a,b]
jepctl app | jepctl gui                 # daemon + native desktop window
jepctl run <model>                    # daemon with a model preloaded
jepctl pull <hf-repo>                 # download model.safetensors + Jepafile.json into ~/.jepctl/models
jepctl tags | jepctl list               # installed models
jepctl rm <model>
jepctl embed <image|clip|audio> [--model m] [--format json|raw]
jepctl stream [--camera 0] [--fps 10] [--model m]
jepctl key generate --name "CI" --role admin|inference [--days 90]
jepctl key list | jepctl key revoke <prefix>
jepctl gestures list [--json]
jepctl gestures export [--model m] [-o bundle.json] [--threshold 0.7] [--margin 0.04] [--no-thumbnails]
jepctl gestures import bundle.json [--replace]
jepctl gestures match photo.jpg [--model m] [--threshold] [--margin]   # exit 0 detected, 1 not detected
jepctl gestures remove <name> | --model m
```

Logs go to stderr, so `jepctl tags --json | jq` works. Gesture commands read and write the same `~/.jepctl/gestures.json` as the GUI and the API (stop the daemon before `import`, or use the API).

`--no-auth` is refused on any host other than `127.0.0.1`/`localhost`.

---

## REST & SSE API

All endpoints live under `/api`. With authentication enabled (the default) send `Authorization: Bearer <token>`; the admin token is printed at first start and stored in `~/.jepctl/auth.token`. Roles: `admin` (models, keys, settings) and `inference` (everything else).

### System & catalog

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/status` | - | health, backend, memory, `active_model`, **`weights {loaded, expected, source}`** |
| GET | `/api/tags` | - | installed models |
| GET | `/api/catalog` | - | verified catalog (what the Models tab shows) |
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
| GET | `/api/cameras` | - | detected devices |
| POST | `/api/camera/start?device=0&fps=10` | inference | |
| POST | `/api/camera/stop` | inference | |
| GET | `/api/camera/frame` | inference | JPEG of **exactly what the model receives** (centre crop, model input size); `X-Frame-Sequence` header |
| GET | `/api/camera/frame?full=true` | inference | full downscaled frame for the ROI editor (`X-Frame-Width/Height`) |
| GET / PUT / DELETE | `/api/camera/roi` | inference | region of interest `{ x, y, w, h }` normalised on the raw frame; applied before every camera embedding and stored in exported bundles |
| GET | `/api/ring-buffer` | inference | last 16 frame thumbnails |

### Few-shot gestures

A gesture is a *prototype*: the L2-normalised mean of one or more reference embeddings, bound to the model that produced them. The registry persists in `~/.jepctl/gestures.json`.

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/gestures` | inference | `{ "name", "is_neutral"?, "from_camera": true }` embeds the current camera frame and **adds a sample** (call again for more). Alternatives: `"embedding": [...]` or `"image_base64": "data:image/jpeg;base64,..."`. |
| GET | `/api/gestures[?model=…\|?all=true]` | inference | gestures of the active model |
| DELETE | `/api/gestures/{name}` | inference | |
| DELETE | `/api/gestures[?all=true]` | inference | clear the active model's gestures |
| POST | `/api/gestures/match` | inference | `{ "from_camera": true \| "embedding" \| "image_base64", "threshold"?, "margin"? }` → decision trace below |
| GET | `/api/gestures/export[?model=…\|?all=true][&threshold=&margin=&thumbnails=false]` | inference | portable bundle `{ version, threshold, margin, gestures[] }`: train on one machine, deploy on many |
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

**How the decision is made.** `raw_cosine` is the naive similarity to the prototype; with the same background, face and lighting it saturates near 1.0 for *every* pose and discriminates nothing. `contrastive` is the cosine after removing the centroid of all prototypes: the shared component: leaving the gesture's own signature. `combined = 0.35·raw + 0.65·max(contrastive, 0)`. A detection requires `combined ≥ threshold`, a lead of `margin_required` over the runner-up, and that the best candidate is not a neutral pose. `patch_diff` is the per-patch dissimilarity to the best prototype over the ViT grid, for the heatmap.

### Security & settings

`POST/GET /api/keys`, `DELETE /api/keys/{prefix}`, `GET /api/audit` (admin); `GET/POST /api/settings`; `GET /api/auth/token` (same-origin testbench bootstrap only).

The API is **same-origin by default**: no CORS headers are sent, so a web page from another origin cannot read responses. Pass `--cors-origins http://localhost:5173` (or `JEPA_CORS_ORIGINS`) to allow a browser app. See [SECURITY.md](SECURITY.md).

---

## Desktop app / testbench

`jepctl app` opens the testbench in a native window; `jepctl serve` serves the same page at `http://127.0.0.1:11435/`. No build step, no external assets, keyboard-navigable (↑/↓ between sections, Esc closes dialogs).

The header is the single source of truth: **model · checkpoint coverage · camera · last latency**.

**Workspace**
- **Overview**: latency, FPS, embeddings count, hardware telemetry.
- **Models**: verified catalog (served by `/api/catalog`), pull with progress, load/unload, Jepafile editor.
- **Embed**: drop an image, see the patch grid and the pooled vector.
- **Live**: server camera, ring-buffer scrubber, and an **event console** (filter errors / detections, copy a line).
- **Energy**: lock a baseline, chart the drift, alerts and webhook.
- **Robot Twin**: WebGL arm, joint sliders, backend switch, mode selector, E-stop, safety gate banner, energy gauge and gesture map editor.
- **Gestures**: *model view* (the exact frame the network receives) with a per-patch difference heatmap; 3 gesture slots + 1 neutral slot; a per-frame **"Why this decision"** table (raw / contrastive / combined per gesture, margin, threshold, server and client decisions with their reasons); threshold, margin and smoothing; **Export / Import bundle**.

**Admin**
- **Integration & Keys**: base URL, auth mode, active model, a quick-start example in curl / JavaScript / Python, scoped tokens, audit trail.
- **Settings**: backend override, memory watermark, storage paths.

Every action button has a **`{ } API`** control that shows the exact request the app sends: with the threshold and margin you just tuned: in curl, JavaScript or Python, ready to paste into your application.

Tips for good detections: register the neutral pose first, take 3-5 samples per gesture while moving slightly, keep the hand large in the frame, and prefer `facebook/dinov2-small`.

---

## Robot Twin (digital twin and arm control)

`jepctl` can drive a 6 DOF arm with a gripper. Everything is previewed on a WebGL twin rendered in the desktop window (raw WebGL, no library, works offline) and gated before it reaches hardware.

```bash
cargo run --release --features metal              # virtual arm only
cargo run --release --features "metal serial"     # plus the physical arm over USB serial
```

| Layer | What it does |
|---|---|
| `RobotBackend` (HAL) | `connect`, `disconnect`, `set_joint_targets`, `get_joint_positions`, `get_gripper_position`, `emergency_stop`. Two implementations: `VirtualWebGlBackend` (in memory, feeds the twin) and `PhysicalSerialBackend` (feature `serial`, Feetech STS3215 / SO-100 style frames: sync write of goal positions, torque on/off, present position read back). |
| `SafetyGuard` | Joint limits in radians (J1 base ±2.6, J2 shoulder ±1.8, J3 elbow ±2.0, J4 wrist pitch ±1.8, J5 wrist roll ±2.6, J6 wrist rotate ±3.1, gripper 0 to 1), velocity ramp of 1.5 rad/s per joint at 30 Hz (no direct jumps), emergency stop that locks every command until an admin resets it. |
| Controller | 30 Hz loop that ramps toward targets, drives the backend and publishes telemetry on a `watch` channel. |

**Modes**

- **Manual**: sliders (or the API) set targets.
- **Mode A, gesture shadowing**: detections from the Gestures tab are mapped to actions through an editable map (`open_gripper`, `close_gripper`, `joint_delta`, `pose`, `approve`, `stop`). Defaults: Open Hand opens the gripper, Fist closes it, Victory approves the pending command.
- **Mode B, safety gate**: forced on with the physical backend. Commands are held as a pending pose, drawn as an amber ghost on the twin, and sent to the hardware only after "Approve and execute" (or the approve gesture).
- **Learn (exploring)**: the arm babbles with small random actions. After every move, once settled, the **camera** frame is embedded with the active JEPA model; the transition `(z_t, a, z_{t+1})` trains a latent world model `z_{t+1} = z_t + W[a; 1]` (ridge regression in embedding space, refit after every observation, persisted in `~/.jepctl/robot_world_model.json`). Nothing is predicted in pixel space: that is the JEPA principle applied to control.
- **Mode C, reach a visual goal**: `POST /api/robot/goal` embeds what the camera sees now as `z_goal`. At each step the controller samples 96 candidate actions, predicts their outcome **inside the learned model**, executes the one with the lowest predicted energy `E = ||z - z_goal||_2 / sqrt(dim)` (the `/api/energy` metric, plus a little exploration noise), observes the real result and learns from it. Until 12 transitions exist the policy is random; the telemetry says which one is in use, and shows predicted versus observed energy so you can judge the model.

**What the agent observes.** With the physical arm it is the raw camera frame (ROI applied). With the virtual backend the twin is **drawn into the camera frame** (same kinematics as the WebGL view, server side), so the picture depends on the arm's own joints and there is something to learn; by default the background is a frozen camera frame ("Freeze background") so that only the arm changes between observations. `GET /api/robot/view` shows exactly this picture; the tab previews it.

Typical run in the simulator: 45 s of Learn, capture the goal at a pose, scramble, Mode C brings all six joints back within about 0.1 rad in 15 to 30 s with a model fit error around 0.05. The search stops on `converged` (energy below 2.5 times the camera noise floor) or `plateau` (60 steps without improvement); a single camera cannot disambiguate every pose, so a plateau at a view-equivalent pose is expected now and then.

**API** (inference role unless noted)

| Method | Path | Description |
|---|---|---|
| GET | `/api/robot/status` | backend, connection, mode, E-stop, safety gate, actual and target joints, pending command, limits, goal progress |
| POST | `/api/robot/target` (admin) | `{ "backend": "virtual" \| "physical" }` (physical needs `--features serial`) |
| POST | `/api/robot/joints` | `{ "joints": [6 rad], "gripper": 0..1, "approved": false }` |
| POST | `/api/robot/approve` | execute the pending command |
| POST | `/api/robot/mode` | `{ "mode": "manual" \| "shadowing" \| "goal_seeking", "safety_gate"?: bool }` |
| POST / DELETE | `/api/robot/goal` | capture (`{ "image_base64"? }`, camera otherwise) / forget the latent goal |
| POST | `/api/robot/observe` | feed one observation (`image_base64` for replays / external cameras); the server camera does this automatically |
| GET / DELETE | `/api/robot/world-model` | learned transitions summary / forget everything learned (admin) |
| GET | `/api/robot/view` | JPEG of what the agent observes (camera, ROI, twin overlay when virtual) |
| POST | `/api/robot/background` | refresh the frozen background from the live camera (drops learned transitions, they were observed over the old one) |
| POST | `/api/robot/e-stop` | engage the emergency stop |
| POST | `/api/robot/reset-safety` (admin) | release it after inspection |
| GET / PUT | `/api/robot/gesture-map` | gesture name to action mapping |
| GET | `/api/robot/ws[?token=]` | WebSocket: telemetry at 30 Hz, accepts `{ joints, gripper, approved }` back |

Serial settings (port, baud, servo IDs, tick calibration, direction) live under `robot_hardware` in `~/.jepctl/settings.json`; defaults target `/dev/ttyUSB0` at 1 000 000 baud with IDs 1 to 7. The physical protocol has been written from the STS3215 register map and is unit tested at the frame level, but it has not been run against a real arm here: treat the first connection as a bench test with the E-stop within reach.

## Jepafile (custom models)

`~/.jepctl/models/<org>/<name>/Jepafile.json` next to `model.safetensors`:

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

Register it with `POST /api/manifests` or the Models tab, then `jepctl pull` / load. `image_size` must be a multiple of `patch_size`; positional embeddings are resampled if the checkpoint was trained at another resolution.

---

## Architecture

```
src/
├── main.rs              CLI (clap) and daemon bootstrap
├── engine/              candle: vit.rs (2D backbone, weight mapping), ijepa.rs, vjepa2.rs (3D tubelets + RoPE), audio.rs, device.rs
├── gestures.rs          prototypes, contrastive matching, decision trace, persistence
├── robot/               HAL (virtual + serial), safety guard, controller (modes A, B, C)
├── hub/                 verified catalog, Jepafile schema, safetensors downloader
├── media/               image/video/audio decoding & preprocessing, camera capture (nokhwa), ring buffer + "model view" + ROI
├── server/              axum routes, handlers, gesture handlers, auth middleware, integration tests
├── auth.rs / config.rs  bearer tokens + RBAC, ~/.jepctl layout
└── ui/                  embedded single-page testbench (vanilla JS, no build)
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the data flow, the weight-loading contract and why the gesture pipeline is built the way it is.

---

## Roadmap

- **Audio-JEPA**: the audio modality is in place (front-end, rectangular ViT, catalog), but no Audio-JEPA checkpoint is published in a loadable format today (`ltuncay/Audio-JEPA` is a PyTorch Lightning pickle). AudioMAE is the verified audio encoder; an Audio-JEPA safetensors release would be a manifest entry away.
- Numerical parity tests against PyTorch reference outputs (needs a contributor with a Python environment, see Verification).
- Larger V-JEPA 2 variants (`vith`, `vitg`) once someone can verify memory and latency on their hardware.
- Native H.264 decoding without `ffmpeg`, if a dependable pure-Rust decoder appears.
- Microphone capture for live audio, mirroring the camera pipeline.
- Running the serial robot backend against a real SO-100 arm (the protocol is frame-tested only).

## Contributing

Bug reports, model requests and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow (`cargo fmt`, `clippy -D warnings`, `cargo test`, and how to verify a checkpoint really loads). Security issues: [SECURITY.md](SECURITY.md).

## License

Apache-2.0: see [LICENSE](LICENSE). Model weights keep their own licenses (I-JEPA and DINOv2: CC-BY-NC 4.0 / Apache-2.0 respectively: check each Hugging Face card).
