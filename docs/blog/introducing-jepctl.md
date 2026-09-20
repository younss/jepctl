# jepctl: run JEPA style encoders locally, and watch a robot learn from what it sees

*Draft, September 2026. Everything below was measured on an Apple M5 Pro with the code at github.com/younss/jepctl.*

Large language models got their local runtime years ago: pull a checkpoint, run it on your machine, call an API. Representation learning encoders never did. If you wanted I-JEPA, V-JEPA 2, DINOv2 or AudioMAE embeddings in an application, you assembled a Python environment, read four different model cards, and hoped the tensor names matched.

jepctl is a single Rust binary that does for these models what Ollama does for LLMs: `jepctl pull facebook/dinov2-small`, `jepctl serve`, and you have a REST and SSE API, a CLI, and a desktop testbench. No Python, no build step for the UI, Metal, CUDA or CPU.

That is the boring half. The interesting half is what you can do once embeddings are cheap and local: teach a camera a few gestures without training anything, and let a robot arm learn a model of its world from what the camera sees.

## What is a JEPA, in one paragraph

A Joint Embedding Predictive Architecture never predicts pixels. It learns to predict the *representation* of a missing part of an image or a clip from the representation of the rest. The result is an encoder whose output vector captures what matters about a scene (objects, layout, motion) and ignores what does not (exact texture, noise). Those vectors are what jepctl gives you: 384 to 1280 floats per image, clip or audio window, from the same API whatever the model.

## Six models that really load

A checkpoint that half loads is worse than one that fails: it still returns vectors of the right size. jepctl refuses any model whose checkpoint does not cover every parameter, and reports the coverage on `/api/status`. The verified catalog today:

| Model | What it is | Output |
|---|---|---|
| facebook/vjepa2-vitl-fpc64-256 | V-JEPA 2, video, 3D tubelets and 3D rotary attention | 1024 d per 16 frame clip |
| facebook/ijepa_vith14_1k and 22k | I-JEPA ViT-H/14 | 1280 d |
| facebook/dinov2-small and base | DINOv2 (CLS + LayerScale) | 384 / 768 d |
| google/vit-base-patch16-224, timm augreg | supervised ViT-B/16 | 768 d |
| gaunernst/...audiomae_as2m | AudioMAE on log-mel spectrograms | 768 d per 10 s |

Each of the four checkpoint layouts (HF ViT, HF DINOv2, HF V-JEPA 2, timm fused QKV) is mapped explicitly. V-JEPA 2 reproduces the reference rotary embedding down to its tiled-sin, interleaved-rotation quirk; attention is computed one head at a time, which brought a 16 frame clip from 70 GB of transient allocations to 2.5 GB and 0.9 s on Metal. AudioMAE taught us something too: its CLS token is untrained and returns the same vector for every sound, so pooling is a manifest parameter and MAE encoders pool by mean.

We say what we did not do: there is no bit for bit comparison with PyTorch, because the repository has no Python. Implementations follow the reference code line by line and are checked for determinism and discrimination. If you run the comparison, open an issue with the numbers.

## Few-shot gestures, with the reasoning on screen

Register three hand poses from your webcam, and a neutral pose. Every frame, the testbench shows not just the winner but *why*: the raw cosine to each prototype, the contrastive score, the blended score, the margin over the runner-up, and a per patch heatmap of where the frame differs from the best prototype.

The contrastive step is the whole trick. A mean pooled embedding of a webcam frame is dominated by what never changes: background, face, torso, light. Raw cosine is 0.99 for every pose. Removing the centroid of all prototypes leaves the gesture's own signature, and suddenly the scores separate. It took an explainability panel to see that; the panel stayed.

Prototypes, thresholds and the camera crop travel in a bundle you can export and import, from the UI, the API or `jepctl gestures export`. Tune on one machine, deploy on many.

## A robot that learns from the camera

This is the part we are most excited about, and the part with the most caveats.

jepctl ships a hardware abstraction layer for a 6 DOF arm with a gripper: a virtual arm rendered in raw WebGL, and a serial backend for Feetech STS servos (the SO-100 family) behind a Cargo feature. A safety guard enforces joint limits and a 1.5 rad/s ramp; an emergency stop locks everything; commands for the physical arm wait behind an approval gate, previewed as an amber ghost on the twin.

Then the JEPA idea is applied to control. The arm moves, the camera frame is embedded, and the transition (embedding before, action, embedding after) trains a latent world model: a locally weighted linear predictor of the *next embedding*, refit after every observation. Give it a goal by capturing the camera view you want, and it plans inside that model: sample 96 candidate actions, predict where each lands in embedding space, execute the closest one to the goal, observe, correct. Nothing is ever predicted in pixel space.

Numbers from the simulator, where the virtual arm is drawn into the camera frame over a frozen background: after 45 seconds of random babbling the model explains about 95 percent of the embedding changes caused by the arm; from a scrambled pose, all six joints return within about 0.1 rad of the goal pose in 15 to 30 seconds. The caveats: a single camera cannot disambiguate every pose, so the search sometimes ends on a plateau at a view-equivalent pose; a live, busy background is noise the model cannot explain away; and no physical arm has been connected yet, so the serial protocol is verified at the frame level only. The tab says all of this while it runs.

## Built for people who integrate, not just click

Every action button in the testbench has an API control that shows the exact request it sends, in curl, JavaScript or Python, with the thresholds you just tuned. The Integration page gives you the base URL, the auth mode and a working example. Telemetry for the robot streams over a WebSocket at 30 Hz. The CLI pipes cleanly (`jepctl tags --json | jq`, logs on stderr).

Security defaults matter for a service that can read your camera: loopback only, bearer tokens with two roles, no CORS unless you opt in per origin, camera and gesture endpoints authenticated.

## Try it

```bash
git clone https://github.com/younss/jepctl && cd jepctl
cargo build --release --features metal      # or --features cuda, or nothing for CPU
./target/release/jepctl pull facebook/dinov2-small
./target/release/jepctl serve --no-auth
```

Open http://127.0.0.1:11435, start the camera, register a neutral pose and two gestures, then open Robot Twin, let it learn for a minute, capture a goal and scramble the pose.

jepctl is Apache 2.0. The roadmap has honest gaps we would love help with: an Audio-JEPA checkpoint in a loadable format, numerical parity tests against PyTorch, larger V-JEPA 2 variants, and the first run of the serial backend on a real arm. Come and break it.
