# Architecture

This document explains how `jepa` is put together and, more importantly, the contracts
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

| variant | CLS token | LayerScale | pooled output | families |
|---|---|---|---|---|
| `Plain` | no | no | mean of patch tokens | I-JEPA, V-JEPA-style |
| `Cls` | yes | no | CLS token | HF `ViTModel`, timm ViT |
| `DinoV2` | yes | yes | CLS token | HF `Dinov2Model` |

`VitBackbone::forward` returns `(patch_tokens [B, N, D], pooled [B, D])`. Patch tokens
exclude the CLS row so `N == grid²` for every variant; the gesture heatmap relies on this.

Positional embeddings are a plain `Tensor` (sin-cos by default, `[1, N(+1), D]`), not a
trainable var, so they do not count toward checkpoint coverage. `adapt_pos_embed` handles
CLS rows and resolution changes (bicubic resampling of the grid, like the reference DINOv2
code).

### 2.2 Weight loading contract (`engine/mod.rs::load_checkpoint_strict`)

```
checkpoint ──► candidate_sources(target) ──► first shape-compatible tensor ──► var.set
                        │
                        └─ HF ViT/I-JEPA names, HF DINOv2 names, timm/Meta fused-QKV slices
```

- Every trainable parameter of the backbone must be filled. `loaded != expected` is a hard
  error (`JepaError::WeightsIncomplete`) that names the missing tensors in the log and maps
  to HTTP `422`.
- A missing file maps to `ModelNotFound` → HTTP `409` with a "run `jepa pull`" hint.
- The result is a `WeightReport { loaded, expected, source }` stored on the model and
  exposed by `/api/status`. `source == "random"` exists only for tests
  (`IJepaModel::load_random`, `EngineManager::load_random_for_test`).

**Why so strict:** a ViT with random weights still returns vectors of the right size. At the
API level it is indistinguishable from a working model — until someone spends a day
wondering why gestures are not recognised. That happened; hence the contract.

### 2.3 Preprocessing contract (`types.rs::Preprocessing`)

`ModelManifest::preprocessing()` gives `{ size, normalization }` (explicit in the manifest,
inferred from the model name otherwise). `EngineManager::preprocessing()` returns it for the
active model and **every** producer of an input tensor uses it:

- `POST /api/embed` and `jepa embed` → `preprocess_image_bytes`
- gestures from `image_base64` → `preprocess_image_bytes`
- camera paths (`embed_current_view`, `jepa stream`) → `RingBuffer::latest_image_tensor` /
  `to_video_tensor` → `preprocess_dynamic_image`

Centre-crop to a square, bicubic resize to `size`, per-channel `(x/255 − mean)/std`.
The ring buffer also stores a JPEG of the 224 px centre crop of each frame — the
"model view" served by `/api/camera/frame`.

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
written atomically to `~/.jepa/gestures.json` after each change.

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

### 3.4 Neutral pose

A gesture flagged `is_neutral` competes in scoring (it contributes to the centroid and can
win) but is never reported as a detection. It absorbs "nothing is being shown" frames that
would otherwise be forced onto the nearest real gesture.

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
| backbone variants, name mapping, pos-embed | `engine/vit.rs` tests | tiny configs on CPU |
| preprocessing, ring buffer | `media/*` tests | shapes, crops, normalisation values |
| HTTP behaviour | `server/tests.rs` | full router, random 32-d model, temp `~/.jepa`, auth on/off |
| UI/HTML consistency | `types.rs` | every static DOM id used by `app.js` exists |

Real checkpoints are verified manually (see CONTRIBUTING → "Adding a model") because CI does
not download gigabytes of weights.

## 6. Known limitations

- V-JEPA: `VJepaModel` is a frame-wise ViT with mean pooling over `T × N` tokens. It cannot
  load Meta's V-JEPA / V-JEPA 2 checkpoints (3D patch embedding, RoPE, different naming) and
  is therefore not in the verified catalog. It is kept for custom Jepafiles and as a starting
  point.
- Gestures use a global embedding of the whole frame; small hands in a big frame have little
  influence. A region-of-interest crop is the natural next step and fits the pipeline
  (crop before `preprocess_dynamic_image`).
- The desktop window (`tao` + `wry`) is a thin WebView over the same HTTP UI.
