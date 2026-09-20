# Hacker News submission

Title options (80 characters max, no marketing words):

1. Show HN: jepctl, a local runtime for JEPA style encoders (I-JEPA, V-JEPA 2, DINOv2) in Rust
2. Show HN: jepctl, Ollama for representation learning models, with a robot that learns from a camera
3. Show HN: Run V-JEPA 2 and I-JEPA locally in pure Rust, plus few-shot gestures and a robot twin

Recommended: option 1. It says what it is; the robot is the surprise inside.

URL: https://github.com/younss/jepctl

First comment (post it right after submitting; HN readers look for it):

I built jepctl because there was no equivalent of Ollama for encoders: models whose output is an embedding, not text. It is one Rust binary (candle, no Python): `jepctl pull facebook/dinov2-small`, `jepctl serve`, then a REST/SSE API, a CLI and an embedded desktop testbench. Metal, CUDA or CPU.

Verified catalog: I-JEPA ViT-H/14, V-JEPA 2 ViT-L (3D tubelets + 3D RoPE, encoder only), DINOv2 S/B, two ViT-B/16, AudioMAE. A checkpoint that does not cover every parameter is refused rather than run on random weights; `/api/status` reports the coverage.

Two things on top of raw embeddings:

- Few-shot gestures from a webcam with the decision fully exposed per frame: raw cosine, contrastive score (centroid removed, otherwise every pose scores 0.99), margin, and a per patch heatmap. Prototypes and thresholds export as a bundle for deployment.

- A robot twin (6 DOF arm, WebGL, optional serial backend for Feetech servos) that learns a latent world model from camera observations: transition (embedding, action, next embedding) trains a locally weighted linear predictor in embedding space, and goal seeking plans inside it. In the simulator the model explains about 95 percent of embedding changes after 45 s of babbling and brings the arm back within about 0.1 rad of a goal view in 15 to 30 s.

Honest gaps: no bit for bit parity test against PyTorch yet (no Python in the repo; I would love someone to run it), no Audio-JEPA checkpoint exists in a loadable format so AudioMAE is the audio encoder, and the serial backend is frame-tested only, never run on a real arm.

Things I learned that might interest you: V-JEPA 2's rotary embedding tiles the sin/cos tables while rotating interleaved pairs, and you must reproduce that quirk or the weights are useless; per head attention took a 16 frame clip from 70 GB of transient Metal allocations to 2.5 GB; AudioMAE's CLS token is untrained and returns the same vector for every sound, so pooling has to be mean.

Apache 2.0. Feedback on the world model approach is very welcome.
