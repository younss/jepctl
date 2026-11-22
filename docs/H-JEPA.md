# H-JEPA as a planner for the Robot Twin: study

Status: study only, nothing implemented. Written 2026-11-21 from the paper's code release,
its model cards and the configs of the released checkpoints (sources at the end).

## 1. What H-JEPA is

*H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning* (Zhang,
Terver, Rabbat, LeCun, Balestriero; arXiv 2610.06805; code and weights MIT) stacks
action-conditioned JEPA world models over time scales:

- **Level 1** encodes a frame with a ViT (CLS token, projected) and predicts the next
  latent from the history and the action (causal Transformer predictor, AdaLN action
  conditioning).
- **Level N** encodes every *stride*-th level-(N-1) latent (a latent MLP, no pixels) and
  predicts one of *its* steps ahead, conditioned on a **latent macro-action**: an action
  encoder compresses `stride` lower-level actions into a small vector (8-d on DROID).
- **Planning is top-down and gradient based.** The top level optimises macro-actions
  towards the goal latent; its predicted latents become subgoals for the level below,
  whose rollout cost is measured in the upper level's space; level 1 returns primitive
  actions. Each level runs AdamW on its actions through the differentiable rollout.
- Training is end to end with a SIGReg regulariser against collapse; no reconstruction.

Reported results (paper, not re-measured here): three levels reach 73.3 % planning
success on Visual AntMaze against 18.0 % for the flat LeWorldModel; on Push-T more levels
hurt (two levels 45.3 %, four 0.7 %). On DROID (real Franka videos, open loop) H-JEPA gets
38.4 % Fréchet path fidelity against 33.6 % for flat LeWM. That is an offline path metric,
not closed-loop task success.

## 2. What is released

| Artifact | Format | Notes |
|---|---|---|
| `jepa-world-models/h-jepa` | pickled `HJEPA` modules (`*_object.ckpt`) + `normalizer.pt` | AntMaze, FourRoom, Cube, Push-T, DROID; 3 seeds each; needs the Python code to unpickle |
| `jepacpp/h-jepa-GGUF` | GGUF (f32, f16, q8_0), one seed per config | parity with the authors' PyTorch published per file (f32 cosine 1.0000000); flat, named tensors plus full hyper-parameters in the metadata |

The DROID 2-level file (`droid/droid_hjepa_l2-seed1-f32.gguf`, 399 tensors) declares:

- encoder: ViT-S/16, 256 px squashed (not cropped), ImageNet mean/std, CLS token, learned
  positional table, fused QKV, LayerNorm ε = 1e-12, exact GELU;
- level 1: raw action 7-d (end-effector `xyz` delta, rotation delta, gripper delta per
  step at 5 fps), embed 384, predictor 12 layers × 6 heads, 16-frame context;
- level 2: stride 3, latent-MLP encoder, MLP action encoder 7·3 → 64 → 8, same predictor size.

## 3. Where it would sit in jepctl

Mode C today (`robot/world_model.rs`) fits `dz = W [a; 1]`, a ridge regression in the
embedding space of whichever catalog encoder is loaded, refit online from the camera, and
plans by scoring 96 random candidates. H-JEPA would replace that with a learned, deep,
multi-step model and a gradient planner. The pieces map as follows:

| H-JEPA piece | jepctl today | Effort |
|---|---|---|
| ViT-S/16 encoder | `VitBackbone`, `Cls` variant; `norm_eps` now configurable (1e-12) | small: a GGUF name map |
| GGUF reading | none (safetensors only) | small: `candle_core::quantized::gguf_file` reads GGUF natively |
| causal AdaLN predictor, latent MLP, action encoder | none | medium: about 400 lines, testable against jepa.cpp |
| gradient planner | random shooting | medium: candle has autograd (`Var`, `backward`) |
| 7-d end-effector action | 6 joint angles + gripper | **the hard part**, see 4 |

## 4. The two real obstacles

1. **Action space.** The DROID models act in end-effector pose deltas of a Franka arm. The
   twin is a 6-DOF joint-space arm with forward kinematics only (`sim_view.rs`). Using the
   planner's output needs inverse kinematics and a scale mapping, and the predicted
   dynamics are a Franka's, not the twin's.
2. **Domain gap.** DROID is Franka arms in real labs, seen from DROID's camera placement.
   Neither the virtual arm drawn by `sim_view.rs` nor a hobby arm on a desk looks like
   that. A world model is only useful if its predictions hold on *our* frames; nothing in
   the paper says they do, and the open-loop DROID metric suggests modest accuracy even
   in-domain.

The simulation checkpoints (AntMaze, Cube, Push-T) do not help: their action spaces and
scenes are those environments'.

## 5. Recommendation

Proceed in stages, each with a go/no-go, rather than port everything up front.

1. **Measure transfer first (about a day).** Load only the DROID encoder and the level-1
   predictor from the GGUF, read-only, behind a test. Record short clips of the twin (and
   of a real arm if available) with known joint motions, convert them to approximate
   end-effector deltas through the existing forward kinematics, and measure whether
   level-1 one-step predictions beat the "no motion" baseline on those frames. Validate the
   port itself first: same numbers as jepa.cpp on the same input.
2. **If it transfers:** port level 2, the action encoder and the gradient planner, add an
   IK layer from end-effector deltas to joint targets, and expose it as a fourth Mode C
   planner next to the ridge model. Keep every move behind the existing safety gate
   (Mode B is forced on hardware).
3. **If it does not transfer** (the likely outcome for the virtual arm): keep H-JEPA's idea,
   not its weights. Add a second, coarse level to the existing online learner: macro-actions
   of `k` primitive moves, a ridge model on every `k`-th embedding, and subgoals handed to
   the current planner. That is cheap, needs no new weights, and addresses the plateaus
   Mode C hits on long reaches.

Alternative considered: calling jepa.cpp's `jepa-plan` as an external process, like
`ffmpeg`. Quick to try, but it adds a C++ build to the install and still leaves obstacles
4.1 and 4.2 untouched, so it does not shortcut stage 1.

## Sources

- Paper and code: arXiv 2610.06805, github.com/kevinghst/H-JEPA (`h_jepa/ARCHITECTURE.md`,
  `docs/RESULTS.md`, `config/train/droid_hjepa_l2.yaml`, `config/eval/droid_l2.yaml`,
  `h_jepa/droid_data.py`).
- Weights: huggingface.co/jepa-world-models/h-jepa; GGUF and parity table:
  huggingface.co/jepacpp/h-jepa-GGUF (metadata read from the DROID f32 file header).
