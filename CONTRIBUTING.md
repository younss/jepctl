# Contributing to jepctl

Thanks for helping. This document is short on purpose: the code and the tests are the
source of truth, and the rules below exist so that the project stays *honest*: a model
either loads completely or not at all, an API either does what it says or returns an error.

## Ground rules

1. **No silent fallbacks.** A missing checkpoint, an unsupported layout, a camera that is
   not running: all of these are errors with a clear message, never a degraded result.
   If you add a code path that "keeps working" with random weights or stale data, it will
   be asked to fail loudly instead.
2. **One preprocessing pipeline.** Everything that produces an embedding: upload, camera,
   CLI, gesture registration, gesture matching: goes through `EngineManager::preprocessing()`
   and the media helpers. Do not hard-code `224` or ImageNet constants.
3. **Explainable decisions.** Anything that decides (gesture match, anomaly) returns the
   numbers it decided on. Extend `GestureMatchResult` rather than hiding a heuristic.
4. **Tests for behaviour, not for lines.** Unit-test pure logic (`gestures.rs`, `vit.rs`),
   integration-test HTTP behaviour (`src/server/tests.rs`). A bug fix comes with the test
   that would have caught it.

## Workflow

```bash
git clone https://github.com/younss/jepctl && cd jepctl
cargo build                      # CPU; add --features metal|cuda for an accelerator
cargo test
cargo fmt --all
cargo clippy --all-targets -- -D warnings
node --check src/ui/app.js       # if you touched the UI
node --test tests/ui/*.test.cjs   # workspace navigation regressions
```

CI runs exactly these on Linux, macOS and Windows. A PR must be green.

Commit messages: imperative subject line (`engine: refuse partial checkpoints`), body
explaining *why* when it is not obvious. Small, focused PRs are reviewed faster.

## Repository map

| Path | What lives there | Read first |
|---|---|---|
| `src/engine/vit.rs` | backbone variants, weight-name mapping, pos-embed adaptation | `candidate_sources`, `adapt_pos_embed` |
| `src/engine/mod.rs` | `EngineManager`, strict checkpoint loading, `WeightReport` | `load_checkpoint_strict` |
| `src/gestures.rs` | prototypes, contrastive matching, decision trace, persistence | `match_gestures` |
| `src/hub/manifest.rs` | verified catalog, Jepafile schema | `VERIFIED` |
| `src/media/` | preprocessing, camera, ring buffer, "model view" | `preprocess_dynamic_image` |
| `src/server/handlers.rs` | shared helpers (`ensure_model_loaded`, `embed_current_view`), REST, SSE | |
| `src/server/gesture_handlers.rs` | gesture REST | |
| `src/server/tests.rs` | HTTP integration tests with a random in-memory model | |
| `src/ui/` | embedded testbench (vanilla JS/CSS, no build) | `setupGestureSandbox` |

More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Adding a model to the verified catalog

1. Inspect the checkpoint layout without downloading it all:

   ```bash
   python3 - <<'EOF'
   import json, struct, urllib.request
   url = "https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors"
   b = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-2000000"})).read()
   n = struct.unpack("<Q", b[:8])[0]
   for k, v in sorted(json.loads(b[8:8+n]).items()):
       if k != "__metadata__" and (".0." in k or "layer" not in k and "blocks" not in k):
           print(k, v["shape"])
   EOF
   ```

2. Check that every tensor maps onto a backbone parameter through `candidate_sources` in
   `src/engine/vit.rs`. If a new naming scheme is needed, add it there (with a unit test)
   rather than special-casing the model.
3. Add a `Verified` entry in `src/hub/manifest.rs` with explicit `variant`, `normalization`
   (check `preprocessor_config.json` on the Hub: `image_mean: [0.5, …]` means `Inception`)
   and `mlp_ratio`.
4. Pull it, load it, and confirm `GET /api/status` reports `weights.loaded == weights.expected`
   with no "Shape mismatch" warning in the logs. Then embed two different images and one of
   them twice: cosine must be 1.0 for the repeat and clearly lower across images.
5. Add a row to the README table with the numbers you observed.

## Verifying numerics against PyTorch (wanted)

Nothing here depends on Python, so parity with the reference implementations is checked
structurally, not bit-for-bit. If you have `torch` + `transformers`/`timm`, the most valuable
contribution is a script that embeds the same file with both and reports max abs error for
I-JEPA, DINOv2, V-JEPA 2 (`VJEPA2Model(...).encoder` mean over tokens, 16 frames at 256 px)
and AudioMAE (`torchaudio.compliance.kaldi.fbank` front-end). Open an issue with the numbers.

## Adding a model family

A family that is not "ViT with optional CLS/LayerScale" (e.g. V-JEPA's 3D tubelet embedding,
RoPE, register tokens) needs a new struct implementing `JepaModelTrait`
(`src/engine/mod.rs`) and its own loader. Keep the contract: `load()` takes a checkpoint path
and returns a `WeightReport`; anything less than full coverage is an error.

## UI conventions

- No framework, no bundler: `index.html`, `app.js`, `styles.css` are embedded at compile time.
- Every DOM id referenced statically from `app.js` must exist in `index.html`
  (`ui_dom_ids_referenced_by_js_exist_in_html` enforces it).
- The gesture sandbox never uses browser camera pixels for inference: only the server frame.

## Reporting bugs

Use the issue templates. For model problems, include the `weights` field of `/api/status`
and the `RUST_LOG=debug` lines around `Loaded … tensors`.
