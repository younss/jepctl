# Architecture

This document explains how `jepctl` is put together and, more importantly, the contracts
that keep it trustworthy. Read it before touching the engine, the media pipeline or the
gesture logic.

## 1. Process layout

```
┌──────────────┐   HTTP/SSE    ┌────────────────────────────────────────────────────┐
│ testbench UI │ ◄───────────► │ axum daemon (tokio)                                │
│ (embedded)   │               │  routes ─► handlers ─► EngineManager ─► candle     │
└──────────────┘               │                   └──► GestureStore               │
        ▲                      │  camera thread ─► RingBuffer (frames + model view) │
   CLI / SDKs                  └────────────────────────────────────────────────────┘
```

- **One process, one active model.** `EngineManager` owns `Option<Box<dyn JepaModelTrait>>`
  behind an `RwLock`; loading swaps it, unloading drops it (and its GPU memory).
- **One camera thread** (`media/capture.rs`) pushes RGB frames into a 16-slot `RingBuffer`
  and broadcasts a sequence number. Consumers (SSE stream, gesture registration) read the
  buffer; they never talk to the camera.
- **State** (`server/handlers.rs::AppState`) is cloned into every handler: engine, catalog,
  auth, audit log, camera supervisor, ring buffer, gesture store, config.

## 2. Engine

### 2.1 Backbone (`engine/vit.rs`)

A single ViT implementation covers every supported checkpoint through `VitVariant`:

| variant | CLS token | LayerScale | default pooling | families |
|---|---|---|---|---|
| `Plain` | no | no | mean of patch tokens | I-JEPA |
| `Cls` | yes | no | CLS token (`mean` for AudioMAE) | HF `ViTModel`, timm ViT, AudioMAE |
| `DinoV2` | yes | yes | CLS token | HF `Dinov2Model` |
| `VJepa2` |: |: | mean of space-time tokens | separate struct, see 2.4 |

`VitBackbone::forward` returns `(patch_tokens [B, N, D], pooled [B, D])`. Patch tokens
exclude the CLS row so `N == grid_h × grid_w` for every variant; the gesture heatmap relies
on this. Inputs may be rectangular and single-channel (`VitConfig::img_w`, `in_chans`).

Positional embeddings are a plain `Tensor` (sin-cos by default, `[1, N(+1), D]`), not a
trainable var, so they do not count toward checkpoint coverage. `adapt_pos_embed` handles
CLS rows and resolution changes (bicubic resampling of the grid, like the reference DINOv2
code).

### 2.2 Weight loading contract (`engine/mod.rs::load_checkpoint_strict`)

```
checkpoint ──► candidate_sources(target) ──► first shape-compatible tensor ──► var.set
                        │
                        └─ HF ViT/I-JEPA, HF DINOv2, HF V-JEPA 2, timm/Meta fused-QKV slices;
                           equal-element-count tensors are reshaped (Conv3d kernel → Linear)
```

- Every trainable parameter of the backbone must be filled. `loaded != expected` is a hard
  error (`JepaError::WeightsIncomplete`) that names the missing tensors in the log and maps
  to HTTP `422`.
- A missing file maps to `ModelNotFound` → HTTP `409` with a "run `jepctl pull`" hint.
- The result is a `WeightReport { loaded, expected, source }` stored on the model and
  exposed by `/api/status`. `source == "random"` exists only for tests
  (`IJepaModel::load_random`, `EngineManager::load_random_for_test`).

**Why so strict:** a ViT with random weights still returns vectors of the right size. At the
API level it is indistinguishable from a working model: until someone spends a day
wondering why gestures are not recognised. That happened; hence the contract.

### 2.3 Preprocessing contract (`types.rs::Preprocessing`)

`ModelManifest::preprocessing()` gives `{ size, normalization }` (explicit in the manifest,
inferred from the model name otherwise). `EngineManager::preprocessing()` returns it for the
active model and **every** producer of an input tensor uses it:

- `POST /api/embed` and `jepctl embed` → `preprocess_image_bytes`
- gestures from `image_base64` → `preprocess_image_bytes`
- camera paths (`embed_current_view`, `jepctl stream`) → `RingBuffer::latest_image_tensor` /
  `to_video_tensor` → `preprocess_dynamic_image`

Centre-crop to a square, bicubic resize to `size`, per-channel `(x/255 − mean)/std`.
The ring buffer also stores a JPEG of the 224 px centre crop of each frame: the
"model view" served by `/api/camera/frame`.

### 2.4 V-JEPA 2 (`engine/vjepa2.rs`)

Not a 2D backbone: its own struct implementing `JepaModelTrait`.

- **Tubelet embedding**: `Conv3d(3, D, (t, p, p), stride (t, p, p))` has no Candle kernel, so
  the input `[B, C, T, H, W]` is unfolded to `[B, (T/t)(H/p)(W/p), C·t·p·p]` and multiplied by
  the kernel reshaped to `[D, C·t·p·p]` (the loader reshapes on equal element count).
- **3D RoPE** inside every attention: head_dim is split into three `2·((hd/3)/2)` slices
  rotated by the token's frame, row and column index; the remainder is left unrotated.
  The reference tiles the sin/cos tables (`[s, s]`) while rotating interleaved pairs
  `(-x₂, x₁)`; we replicate that exactly (`rotation_matches_reference_formula`).
- **Memory**: attention is computed one head at a time. A 16-frame 256 px clip is 2048
  tokens; a full `[B, 16, 2048, 2048]` f32 score tensor per layer, kept alive by the Metal
  allocator, reached 70 GB. Per head the working set is `N²·4 B` ≈ 16 MB.
- Pooled output is the mean over all space-time tokens; the "patch tokens" returned for the
  heatmap are those of the last temporal slot.
- Still images are embedded as `tubelet_size` identical frames.

### 2.5 Audio (`engine/audio.rs`, `media/audio.rs`)

The 2D backbone with `in_chans = 1` and a rectangular grid (`frames/patch × n_mels/patch`).
The front-end follows Kaldi `fbank` as used by AudioMAE/AST: 25 ms Povey window
(Hamming^0.85), 10 ms hop, DC removal, pre-emphasis 0.97, 512-point FFT, 128 mel bins
(1127·ln(1 + f/700)) from 20 Hz to Nyquist, `ln(max(e, ε))`, zero-pad to 1024 frames, then
`(x − mean) / (2·std)` with the AudioSet statistics carried in the manifest.

**Pooling is a manifest parameter.** MAE-pretrained encoders keep a CLS token that the
pre-training objective never trains as a summary; reading it out yields the same vector for
every input (observed: cosine 1.000 between a tone, a chirp and noise). AudioMAE pools by mean.

### 2.6 Clip and audio decoding (`media/video.rs`, `media/audio.rs`)

GIF, animated WebP and WAV are decoded in Rust. MP4/WebM and MP3/FLAC/OGG are decoded by an
external `ffmpeg` (raw RGB / f32 PCM over stdout) because no dependable pure-Rust H.264/MP3
decoder exists; without `ffmpeg` the error names the binary. Clips are uniformly sub-sampled
to `clip_frames()` of the model (16 for V-JEPA 2); image models embed each sampled frame
and average the pooled vectors.

## 3. Gestures (`gestures.rs`)

### 3.1 Data model

```
RegisteredGesture {
  name, model_name, dimension, is_neutral,
  samples:          Vec<unit vector>          // every reference embedding
  prototype:        unit vector               // normalised mean of samples
  patch_prototype:  Option<Vec<Vec<f32>>>     // running mean of per-patch tokens
  thumbnail:        data URI of the model view at last capture
}
```

Gestures are keyed by name and bound to `model_name`; matching only considers gestures of
the active model (embeddings from different models live in different spaces). The store is
written atomically to `~/.jepctl/gestures.json` after each change.

### 3.2 Why registration and matching share one pipeline

The original sandbox registered browser (`getUserMedia`) pixels and matched server
(`nokhwa`) pixels: different device selection, exposure, crop and resize. Even a perfect
model cannot match those. Now both call `embed_current_view()`, which embeds the latest ring
buffer frame with the active model's preprocessing. The UI shows that exact frame so the
user sees what the model sees.

### 3.3 Scoring

Mean-pooled (or CLS) embeddings of a webcam frame are dominated by what never changes:
background, face, torso, lighting. Raw cosine to any prototype is ≈0.99 and useless.

```
c        = centroid of all prototypes (the shared component)
raw      = cos(x, p_i)
contrast = cos(x − c, p_i − c)                       (needs ≥ 2 gestures)
combined = 0.35·raw + 0.65·max(contrast, 0)          if raw ≥ 0.40, else raw/2
detected = best is not neutral
        ∧ combined_best ≥ threshold
        ∧ combined_best − combined_second ≥ margin
```

Every intermediate value is returned in `GestureMatchResult` with a human-readable
`reason`, so the UI (and API users) can see *why* a frame was or was not recognised.
`patch_diff[i] = 1 − cos(x_patch_i, p_patch_i)` against the best candidate gives the
spatial heatmap.

The client adds temporal smoothing (moving average over N frames of `combined`) and applies
its own threshold/margin on the smoothed values; the server decision uses the values passed
in the SSE query. Both decisions and their reasons are displayed side by side.

### 3.4 Region of interest

`types::Roi { x, y, w, h }` (normalised on the raw frame) lives in `AppState::camera_roi`,
is persisted in `settings.json` and travels inside gesture bundles. Every camera path -
`embed_current_view`, `/api/camera/frame`, `jepctl stream`: crops to it *before* the centre
square crop, so the hand can fill the model input. Because prototypes captured with a crop
only match frames cropped the same way, importing a bundle restores its ROI.

### 3.5 Neutral pose

A gesture flagged `is_neutral` competes in scoring (it contributes to the centroid and can
win) but is never reported as a detection. It absorbs "nothing is being shown" frames that
would otherwise be forced onto the nearest real gesture.

## 3b. Robot subsystem (`robot/`)

```
RobotHandle { core: Arc<Mutex<RobotCore>>, telemetry_tx: watch::Sender<RobotTelemetry> }
RobotCore   { backend: Box<dyn RobotBackend>, safety: SafetyGuard, mode, safety_gate,
              joints/gripper (actual), targets (commanded), pending (gated), explorer, gesture_map }
```

- `controller::spawn_control_loop` ticks at 30 Hz: `RobotCore::tick(dt)` ramps the actual
  position toward the targets under `SafetyGuard::ramp` (max 1.5 rad/s, clamped to the
  limits, frozen under E-stop), pushes it to the backend and publishes telemetry. The
  WebSocket handler forwards every telemetry change and accepts joint commands back.
- `RobotCore::submit` validates against the limits and, when the physical backend is
  active and the command is not approved (Mode B), parks it in `pending` instead of
  `targets`. The twin draws `pending` as a ghost; `approve_pending` promotes it.
- Mode A: the SSE stream handler calls `apply_gesture(name)` for every detection it
  emits, so the arm follows exactly what the Gestures tab shows. Gesture moves are
  always gated on the physical arm.
- Learning (`LatentAgent` + `world_model::LatentWorldModel`): the agent alternates
  `AwaitObservation -> Moving`. `spawn_camera_observer` embeds the server camera frame
  each time the arm has settled and calls `RobotCore::observe`, which (1) records the
  completed transition `(z_prev, a, z)` and refits the world model, a ridge regression
  `dz = W [a; 1]` in embedding space with recency weights, (2) measures the energy to the
  goal if one is set, (3) picks the next action: random babbling while fewer than 12
  transitions exist, otherwise the best of 96 candidates evaluated in the model plus 20%
  exploration noise. The step grows when reality beats the prediction and shrinks
  otherwise. The model is persisted every 10 transitions and reloaded at start; a
  change of embedding dimension (other model) clears it. The WebGL twin is never an
  observation source: learning is defined by what the camera sees.
- Serial protocol (`hal::protocol`): Feetech STS3215 / Dynamixel 1.0 style frames
  (`FF FF ID LEN INSTR PARAMS CHK`), sync write of goal positions for all servos in one
  frame, torque enable, present position read back. Tick calibration (`zero_ticks`,
  `direction`, gripper open/closed ticks) is in `HardwareConfig` under `robot_hardware`
  in settings.json. Frame encoding is unit tested; no physical arm was available to run it.
- Simulated camera (`sim_view.rs`): with the virtual backend, `observe_scene` draws the arm
  (same forward kinematics as the WebGL twin, fixed three quarter top view) into the camera
  frame before embedding, over a frozen background by default. Observations are the mean of
  two frames 80 ms apart. The world model is refit locally around the current pose before
  planning (joint space Gaussian weights, 0.6 rad); the planner also considers momentum
  (previous action, reversed and scaled) and a memory candidate: heading toward the
  remembered pose whose observed embedding is closest to the goal. The search ends on
  `converged` (below 2.5 x the noise floor measured at capture) or `plateau` (60 steps
  without improvement) and re-checks the view once a second afterwards.

### Mirror mode

`RobotCore::apply_mirror(scores)` receives the full score list of every stream frame
(not only the winner). Gestures mapped to `GestureAction::Pose` are blended with
`companion::blend_weights`: softmax over `combined` scores with temperature 0.06,
candidates more than 0.2 below the best get no weight, and nothing moves while the
best score is under 0.35. The result is one joint command submitted through the
usual gate, so the physical arm still needs approval. Poses are taught from the UI
with two calls: `POST /api/gestures {from_camera}` then `PUT /api/robot/gesture-map`.

## 3c. Companion (`companion/`), microphone (`media/mic.rs`) and sounds

```
CompanionHandle { core: Arc<Mutex<CompanionCore>>, telemetry_tx: watch::Sender<CompanionTelemetry> }
CompanionCore   { mode, pose (actual), target (manual), mirror_target, animation, cues,
                  attention, hearing, seeing, last_trigger (cooldowns) }
```

- The control loop ticks at 30 Hz: an animation (nod, shake, wave, cheer, dance,
  startle, sleep) overrides the target with a time envelope; otherwise the body
  approaches `mirror_target` (interactive) or `target` (manual), and in interactive
  mode the head pans and tilts toward `attention`. Idle breathing and blinks are
  cosmetic and computed from the tick.
- Two observer tasks (`server/companion_handlers.rs::spawn_observers`) run only in
  interactive mode. Eyes: every 100 ms a 32x24 grayscale thumbnail of the latest
  frame is diffed with the previous one (`motion_centroid`) to update attention; every
  300 ms the camera view is embedded and matched against the gestures of the vision
  model (same `match_gestures` as the sandbox). Ears: every 700 ms the last 1.5 s of
  microphone audio are embedded by the audio model and matched against the sounds.
- `observe_gestures` triggers the cue of the winner (non pose behaviours, 2 s
  cooldown per cue) and blends every pose cue with `blend_poses`; `observe_sound`
  triggers sound cues and startles on loud unknown audio (RMS above 0.35).
- `MicSupervisor` mirrors `CameraSupervisor`: a worker thread owns the `cpal` input
  stream, down-mixes to mono into a 12 s ring at the device rate, and `latest_clip`
  resamples the tail to 16 kHz. `push_samples` feeds the ring in tests.
- Sounds reuse `GestureStore` (`~/.jepctl/sounds.json`), bound to the audio model's
  name; the audio model has its own slot in `EngineManager` so both modalities stay
  loaded. Cues persist in `~/.jepctl/companion.json`.

## 3d. World reconstruction (`server/world_handlers.rs`)

`GET /api/world/frame` embeds the current camera frame and turns the per-patch
tokens into a relief field. `heights` is the foreground signal: a background
prototype is the mean of the patches whose distance from the frame centroid is below
the median (the bulk of a scene is background), and each patch's distance from that
prototype is how much it belongs to the foreground; the grid is then 3x3 smoothed.
This uses JEPA's structure rather than pixel edges, so it tracks the actual object.
The handler also returns `colors` (a fixed, cached `dim x 3` projection of the unit
patch vector), `pixels` (the frame averaged into the grid) and `image` (the aligned
model-view JPEG as a data URI). The UI builds a textured mesh once, and each tick
displaces its vertices by the bilinearly upsampled relief and re-uploads `image` as
the texture, so the real scene appears in 3D. It is neither generative nor a metric
depth model: the real frame given shape by what the encoder perceives.

On top of the encoder, `engine::scene_predictor::WorldScenePredictor` is an online
world model held in `AppState` (after LeWorldModel, arXiv:2603.19312). Each frame it
pools the patch tokens, projects to a low dimension (fixed random projection, JL), and
fits a per-dimension decayed-least-squares AR(1) that predicts the next projected
embedding; the normalised error of last tick's prediction is the **surprise** (baseline
near zero, spikes on the unexpected). A per-patch constant-velocity predictor gives the
`surprise_map` and the `predicted_heights` (expected next relief). A small memory of
unit pooled embeddings gives **recognition** (nearest-neighbour cosine; novel states are
remembered). The encoder is frozen (the catalogue model), only the predictor is fit
online; there is no training run and nothing is persisted. `POST /api/world/reset`
clears it.

## 4. HTTP layer

- `handlers.rs` holds the shared helpers: `ensure_model_loaded` (auto-loads the first
  *downloaded* model, never a random one), `embed_current_view`, `api_error`/`engine_error`
  (error → status mapping: `ModelNotFound → 409`, `InvalidPayload → 400`,
  `WeightsIncomplete → 422`).
- `gesture_handlers.rs` is the gesture REST surface; `resolve_input` normalises the three
  input sources (`from_camera`, `embedding`, `image_base64`).
- The SSE stream emits one event per **new** ring-buffer sequence number; distinct errors are
  emitted once as `event: error` rather than spamming per frame.
- Auth is per handler (`authenticate_request`) with roles `Admin`/`Inference`; the stream
  also accepts `?token=`. CORS is off unless origins are configured (see SECURITY.md).

## 5. Testing strategy

| Layer | Where | What |
|---|---|---|
| scoring, prototypes, persistence | `gestures.rs` tests | pure logic, deterministic vectors |
| backbone variants, name mapping, pos-embed | `engine/vit.rs` tests | tiny configs on CPU, rectangular grids |
| V-JEPA 2 rotary tables and rotation, shapes | `engine/vjepa2.rs` tests | reference formulas on tiny configs |
| audio front-end (WAV, FFT, fbank), clip decoding | `media/audio.rs`, `media/video.rs` tests | synthetic tones, in-memory GIFs |
| preprocessing, ring buffer | `media/*` tests | shapes, crops, normalisation values |
| HTTP behaviour | `server/tests.rs` | full router, random 32-d model, temp `~/.jepctl`, auth on/off |
| UI/HTML consistency | `types.rs` | every static DOM id used by `app.js` exists |

Real checkpoints are verified manually (see CONTRIBUTING → "Adding a model") because CI does
not download gigabytes of weights.

## 6. Known limitations

- No numerical parity test against PyTorch: implementations follow the reference code and
  are checked for determinism and discrimination on synthetic inputs. A contributor with a
  Python environment can close this gap (CONTRIBUTING → verification).
- V-JEPA 2 on CPU takes ≈10 s per 16-frame clip; the live stream then emits one event per
  clip. Metal/CUDA are the intended targets for video.
- Audio has no live capture yet (files only); Audio-JEPA has no loadable public checkpoint.
- The desktop window (`tao` + `wry`) is a thin WebView over the same HTTP UI.
