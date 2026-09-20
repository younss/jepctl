# JEPA: Local Runtime, CLI, and Testbench for Joint-Embedding Predictive Architectures

JEPA is a production-ready, pure-Rust local runtime, desktop testbench, and CLI purpose-built for non-generative representation learning architectures (I-JEPA, V-JEPA, and future Audio-JEPA variants). It operates similarly to Ollama, but is engineered specifically for representation learning, spatial patch extraction, spatio-temporal video embedding, and real-time visual anomaly detection.

Written in 100% pure Rust, it compiles to a single binary distribution, features native hardware acceleration (Metal on Apple Silicon, CUDA on Linux/Windows, and multi-threaded CPU fallback), and enforces strict local security controls.

---

## Key Features

* **Pure Rust Inference**: Built on Hugging Face `candle-core`, `candle-nn`, and `candle-transformers` with zero Python runtime dependencies.
* **Unified Model Support**:
  * **I-JEPA**: 2D Image Transformer (ViT-H/14, ViT-L/16, ViT-B/16) with spatial patch token extraction.
  * **V-JEPA**: Spatio-temporal video encoder processing `[B, C, T, H, W]` tensors via a 16-frame sliding ring buffer.
* **Multi-Modal Interaction**:
  1. **Native Desktop Application**: Standalone native GUI window (`jepa app` or `jepa gui`).
  2. **Native CLI**: `jepa run`, `jepa pull`, `jepa embed`, `jepa stream`, `jepa tags`, `jepa key`.
  3. **Embedded Dark-Mode Web UI**: Served directly from the single binary on `http://127.0.0.1:11435/`.
  4. **REST and SSE Endpoints**: For external clients and automation pipelines.
* **Strict Security Controls**:
  * Default binding to `127.0.0.1:11435`. LAN binding (`0.0.0.0`) requires explicit configuration and enforces authentication.
  * Constant-time Bearer token verification (`subtle::ConstantTimeEq`).
  * Role-Based Access Control (RBAC): `admin` and `inference` scopes.
  * Path traversal protection preventing model files from escaping `~/.jepa/models/`.
  * Magic byte format sniffing and hard payload ceilings (20 MB for images, 200 MB for video).
* **Live Camera Stream**: Cross-platform video capture via `nokhwa` (AVFoundation, V4L2, Media Foundation) with a 16-frame sliding buffer.
* **Continuous Energy & Anomaly Scoring**: Real-time Euclidean (L2) and cosine dissimilarity tracking against a locked nominal baseline latent vector, with browser audio synthesizer alerts and external webhook dispatch.

---

## Architecture and Directory Structure

```text
jepa-runtime/
├── Cargo.toml                   # Dependencies, edition 2021, and backend feature gates
├── build.rs                     # Platform entitlement injection and asset tracking
├── LICENSE                      # Apache 2.0 License
├── packaging/
│   ├── macos/
│   │   ├── Info.plist           # Camera permission strings (NSCameraUsageDescription)
│   │   ├── com.jepa.daemon.plist# LaunchAgent definition for login persistence
│   │   └── build_pkg.sh         # macOS .pkg package generator
│   ├── linux/
│   │   ├── jepa.service         # systemd user service definition
│   │   └── build_deb.sh         # Debian .deb and tarball packaging script
│   └── windows/
│       └── installer.iss        # Inno Setup installer script
├── src/
│   ├── main.rs                  # CLI entrypoint and command dispatcher
│   ├── desktop.rs               # Native desktop window wrapper (Tao + Wry)
│   ├── config.rs                # Cross-platform paths (~/.jepa) and security constants
│   ├── auth.rs                  # Bearer token generation, RBAC, and constant-time validation
│   ├── types.rs                 # Shared DTOs, schemas, and error definitions
│   ├── engine/
│   │   ├── mod.rs               # Engine abstraction traits and supervisor
│   │   ├── device.rs            # Hardware detection: Metal -> CUDA -> CPU
│   │   ├── vit.rs               # Vision Transformer backbone in pure Candle
│   │   ├── ijepa.rs             # 2D image encoder and patch extractor
│   │   └── vjepa.rs             # Spatio-temporal video encoder [B, C, T, H, W]
│   ├── hub/
│   │   ├── mod.rs               # Local registry catalog manager
│   │   ├── downloader.rs        # HF safetensors stream downloader with progress
│   │   └── manifest.rs          # Jepafile schema and validation logic
│   ├── media/
│   │   ├── mod.rs
│   │   ├── image.rs             # Magic bytes, bicubic resize, ImageNet normalization
│   │   ├── capture.rs           # Thread-safe camera feed capture via nokhwa
│   │   └── ring_buffer.rs       # 16-frame circular sliding window for V-JEPA
│   ├── server/
│   │   ├── mod.rs               # Axum server initialization
│   │   ├── middleware.rs        # Auth guard, audit logger, and payload limits
│   │   ├── routes.rs            # REST and SSE streaming route definitions
│   │   ├── handlers.rs          # Handlers for embedding, streaming, and energy scoring
│   │   └── ui_assets.rs         # Embedded static Web UI assets (HTML, CSS, JS)
│   └── ui/
│       ├── index.html           # Single-page testbench interface
│       ├── app.js               # Reactive UI logic, webcam canvas, SSE listener
│       └── styles.css           # Clean dark-mode interface styles
```

---

## Installation and Compilation

### Prerequisites

* Rust 1.80+ (Rust 2021 edition).
* Platform build tools:
  * **macOS**: Xcode Command Line Tools (`xcode-select --install`).
  * **Linux**: `build-essential`, `libasound2-dev`, `libv4l-dev`.
  * **Windows**: MSVC C++ Build Tools.

### Build from Source

```bash
# Clone the repository
git clone https://github.com/facebookresearch/jepa.git
cd jepa

# Build optimized release binary (Universal CPU fallback)
cargo build --release

# Build with Apple Silicon GPU acceleration (Metal)
cargo build --release --features metal

# Build with NVIDIA GPU acceleration (CUDA)
cargo build --release --features cuda
```

Binary will be produced at `target/release/jepa`.

---

## CLI Usage Guide

### Launch Native Desktop Application

```bash
# Launch native desktop GUI window with daemon automatically running in background
jepa app

# Alternatively use the gui alias or flag
jepa gui
jepa serve --gui
```

### Start the Daemon (Headless / Server Mode)

```bash
# Start on default loopback (127.0.0.1:11435)
jepa serve

# Start in development mode with auth disabled (loopback only)
jepa serve --no-auth

# Bind to custom interface and port
jepa serve --host 0.0.0.0 --port 11435
```

### Pull Model Weights

```bash
# Pull I-JEPA Huge (ViT-H/14, 1280 dimensions)
jepa pull facebook/ijepa_vith14_1k

# Pull V-JEPA Large (ViT-L/16, 1024 dimensions)
jepa pull facebookresearch/jepa:vjepa_vitl16
```

### Inspect and Manage Models

```bash
# List all locally installed models and verified catalog entries
jepa tags

# Remove an installed model
jepa rm facebook/ijepa_vith14_1k
```

### Compute Latent Embeddings

```bash
# Embed an image file with JSON output
jepa embed photo.jpg --model facebook/ijepa_vitb16_1k

# Embed with raw comma-separated floats output
jepa embed photo.png --format raw
```

### Stream Continuous Embeddings

```bash
# Stream embeddings from camera 0 at 10 FPS
jepa stream --camera 0 --fps 10
```

### API Token Management

```bash
# Generate a new Admin token
jepa key generate --name "Admin CI Key" --role admin

# Generate an Inference-only token with 90-day expiry
jepa key generate --name "Camera Client" --role inference --days 90

# List active tokens
jepa key list

# Revoke a token by prefix
jepa key revoke jepa_sec_a1c
```

---

## Embedded Web Testbench (8 Sections)

Open `http://127.0.0.1:11435/` in any modern browser to access the zero-dependency, dark-mode testbench:

1. **Section 1: Overview (System Dashboard)**
   * Forward pass latency (ms), continuous stream FPS, total embeddings computed, and daemon uptime.
   * Telemetry cards showing accelerator hardware, memory usage, CPU threads, and quick actions.
2. **Section 2: Models (Catalog & Manager)**
   * Installed models table with parameter count, disk footprint, load into GPU, and delete controls.
   * One-click pull buttons for verified models.
   * Model pull bar with real-time transfer speed (MB/s) and percentage progress.
   * In-browser Jepafile manifest editor.
3. **Section 3: Image Playground**
   * Drag-and-drop file target (PNG, JPEG, WebP).
   * Inspection canvas rendering the input image overlaid with a 14x14 or 16x16 ViT patch grid.
   * Heatmap bar visualization of the pooled latent vector.
   * One-click vector copy and JSON export.
4. **Section 4: Video & Camera Stream**
   * Live web camera selector and target frame rate controls (5, 10, 15 FPS).
   * 16-frame visual sliding ring buffer scrubber strip showing live thumbnail history.
   * Live Server-Sent Events (SSE) feed outputting chunked spatio-temporal updates.
5. **Section 5: Anomaly & Energy Monitor**
   * "Lock Current State as Nominal Baseline" button to lock normal physical conditions.
   * Real-time Canvas line chart plotting continuous L2 / cosine dissimilarity.
   * Interactive alert threshold slider.
   * Audio synthesizer alert tone (Web Audio API) and flashing alert banner.
   * Configurable alert webhook POST trigger.
6. **Section 6: Security & API Keys**
   * Active API keys table with prefix, role, and revoke buttons.
   * Modal dialog to generate scoped tokens.
   * Local Loopback vs LAN access toggle with warning protections.
   * Rolling audit trail of the last 100 API calls.
7. **Section 7: Settings**
   * Compute backend override (Auto-Detect, Metal, CUDA, CPU).
   * GPU memory high-watermark ratio (e.g. 0.80) and idle model unload timeout.
   * Storage directory path inspector.
8. **Section 8: Few-Shot Gesture Sandbox**
   * "Model view": the exact 224×224 centre crop the network receives, with an optional per-patch difference heatmap against the best-matching prototype.
   * Registration slots (3 gestures + 1 neutral rest pose) fed by the server camera, multiple samples per gesture, persisted per model.
   * Reasoning table per frame: raw cosine vs. contrastive vs. combined score for every gesture, runner-up margin, server and client decisions with their reasons.
   * Threshold, margin and temporal smoothing controls; audio/theme/hold-timer actions on detection.

---

## REST and Streaming API Reference

All requests accept an optional `Authorization: Bearer <token>` header when authentication is active.

### System & Catalog

* `GET /api/status`: Returns system health, active backend, memory consumption, loaded model and its `weights` coverage report.
* `GET /api/tags`: Returns array of installed model manifests and sizes.
* `POST /api/models/load`: `{ "model_name": "facebook/ijepa_vith14_1k" }`. Returns `{ weights: { loaded, expected, source } }`; loading fails (`422`) when the checkpoint layout does not cover every backbone parameter, and (`409`) when the model is not downloaded. A model never runs on random weights.
* `POST /api/models/unload`: Evicts currently active model from GPU/system memory.
* `DELETE /api/models/{name}`: Deletes a model from local disk.
* `POST /api/pull`: Streamed JSON progress events downloading a model from Hugging Face.

### Inference & Anomaly Scoring

* `POST /api/embed`: Multipart form upload accepting `file` (PNG, JPEG, WebP). Returns `{ model, dimension, latency_ms, embedding, patch_embeddings }`. Video files are not supported yet (`501`).
* `GET /api/embed/stream?fps=10&threshold=0.70&margin=0.04`: Server-Sent Events (SSE) endpoint emitting one event per new camera frame: `{ frame_index, model, latency_ms, embedding, gesture_match? }`. Errors (no model, no camera) arrive as `event: error`.
* `POST /api/energy`: Accepts `{ "vector1": [...], "vector2": [...], "threshold": 0.45 }`. Returns `{ l2_distance, cosine_similarity, cosine_dissimilarity, anomaly }`.

### Hardware & Security

* `GET /api/cameras`: Returns detected camera devices.
* `POST /api/camera/start?device=0&fps=10`: Starts background camera capture.
* `POST /api/camera/stop`: Stops camera capture.
* `GET /api/camera/frame`: JPEG of exactly what the model receives (224×224 centre crop of the latest frame). `X-Frame-Sequence` header identifies the frame.
* `GET /api/ring-buffer`: Returns `{ count, thumbnails }` with base64 data URIs.

### Few-Shot Gestures

Gestures are prototypes (L2-normalised mean of one or more reference embeddings) bound to the model that encoded them. They persist in `~/.jepa/gestures.json`.

* `POST /api/gestures`: `{ "name": "Open Hand", "from_camera": true, "is_neutral": false }` embeds the latest server camera frame and adds it as a sample (call again to add more samples). Alternatives to `from_camera`: `"embedding": [...]` or `"image_base64": "data:image/jpeg;base64,..."`.
* `GET /api/gestures`: Gestures of the active model (`?model=<name>` or `?all=true`).
* `DELETE /api/gestures/{name}` / `DELETE /api/gestures?all=true`: Remove one / all gestures.
* `POST /api/gestures/match`: `{ "from_camera": true, "threshold": 0.70, "margin": 0.04 }` (or `embedding` / `image_base64`). Returns the full decision trace:

```json
{
  "matched": "Open Hand", "detected": true, "confidence": 0.81, "margin": 0.23,
  "threshold": 0.70, "margin_required": 0.04, "method": "contrastive",
  "reason": "'Open Hand' scored 0.81 with margin 0.230",
  "scores": [
    { "name": "Open Hand", "is_neutral": false, "raw_cosine": 0.99, "contrastive": 0.72, "combined": 0.81, "sample_count": 4 },
    { "name": "Rest",      "is_neutral": true,  "raw_cosine": 0.98, "contrastive": 0.21, "combined": 0.48, "sample_count": 3 }
  ],
  "patch_diff": [0.01, 0.02, "..."], "grid_size": 16
}
```

`raw_cosine` is the naive similarity (it saturates near 1.0 for every pose sharing the same background). `contrastive` is the cosine after removing the centroid of all prototypes, i.e. the gesture's own signature. `combined = 0.35·raw + 0.65·max(contrastive, 0)`; detection requires `combined ≥ threshold` and a lead of `margin_required` over the runner-up, and a neutral gesture winning always blocks detection. `patch_diff` is the per-patch dissimilarity to the best prototype (row-major over the ViT grid) for spatial explanation.
* `POST /api/keys`: Generates a new scoped Bearer token.
* `GET /api/keys`: Lists active token prefixes and roles.
* `DELETE /api/keys/{prefix}`: Revokes an active token.
* `GET /api/audit`: Returns the rolling 100-entry audit log.

---

## Native Installers & OS Packaging

### macOS (.pkg Installer & LaunchAgent)

```bash
# Build native Apple Silicon installer package
chmod +x packaging/macos/build_pkg.sh
./packaging/macos/build_pkg.sh
```

Installs `jepa` to `/usr/local/bin/jepa` and configures `~/Library/LaunchAgents/com.jepa.daemon.plist` for automatic boot persistence with camera entitlement descriptions in `Info.plist`.

### Linux (.deb & systemd Service)

```bash
# Build Debian package and standalone tarball
chmod +x packaging/linux/build_deb.sh
./packaging/linux/build_deb.sh
```

Provides a `.deb` package and configures `~/.config/systemd/user/jepa.service`:

```bash
systemctl --user daemon-reload
systemctl --user enable --now jepa
```

### Windows (Inno Setup Installer)

Open `packaging/windows/installer.iss` in Inno Setup 6 and click **Compile** to generate `jepa-setup-x86_64.exe` with PATH registration, desktop shortcuts, and an optional Windows Service.
