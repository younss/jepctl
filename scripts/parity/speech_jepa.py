#!/usr/bin/env python3
# Usage: jepctl embed clip.wav --model labhamlet/speech-jepa-base --format json > ours.json
#        python speech_jepa.py ~/.jepctl/models/labhamlet/speech-jepa-base/model.safetensors clip.wav ours.json
# The clip must be 16 kHz mono 16-bit PCM, at most 30 s long.
# Requires: torch, safetensors, numpy.
# Independent reference for labhamlet/speech-jepa-base written from the published
# SpeechJEPAModel: conv feature encoder, projection, post-norm RoPE Transformer.
import json, sys, wave
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file

ckpt, wav_path, ours_path = sys.argv[1:4]
W = load_file(ckpt)
with wave.open(wav_path) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
h = torch.from_numpy(x)[None, None]
for i, (k, s) in enumerate(zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])):
    p = f"feature_extractor.conv_layers.{i}."
    h = F.conv1d(h, W[p + "conv.weight"], stride=s)
    if i == 0:
        h = F.group_norm(h, 512, W[p + "layer_norm.weight"], W[p + "layer_norm.bias"], eps=1e-5)
    h = F.gelu(h)
h = h.transpose(1, 2)
ln = lambda t, n, d, eps: F.layer_norm(t, (d,), W[n + ".weight"], W[n + ".bias"], eps=eps)
lin = lambda t, n: F.linear(t, W[n + ".weight"], W[n + ".bias"])
h = ln(lin(ln(h, "feature_projection.layer_norm", 512, 1e-5), "feature_projection.projection"), "feature_projection.post_layer_norm", 768, 1e-5)
S = h.shape[1]
inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
fr = torch.outer(torch.arange(S, dtype=torch.float32), inv)
cos, sin = fr.cos()[None, :, None, :], fr.sin()[None, :, None, :]
def rot(t):
    t = t.reshape(*t.shape[:-1], -1, 2)
    return torch.stack([t[..., 0] * cos - t[..., 1] * sin, t[..., 1] * cos + t[..., 0] * sin], -1).flatten(3)
for i in range(12):
    p = f"encoder.layers.{i}."
    q, k, v = (lin(h, p + f"attention.{n}_proj").view(1, S, 12, 64) for n in "qkv")
    a = F.scaled_dot_product_attention(rot(q).transpose(1, 2), rot(k).transpose(1, 2), v.transpose(1, 2))
    h = ln(h + lin(a.transpose(1, 2).reshape(1, S, 768), p + "attention.out_proj"), p + "layer_norm", 768, 1e-6)
    ff = lin(F.gelu(lin(h, p + "feed_forward.intermediate_dense")), p + "feed_forward.output_dense")
    h = ln(h + ff, p + "final_layer_norm", 768, 1e-6)
ref = h[0].mean(0)
o = torch.tensor(json.load(open(ours_path))["embedding"])
print(f"{wav_path.split('/')[-1]}: {S} frames, pooled cos={F.cosine_similarity(ref, o, dim=0).item():.7f} rel_max_err={((ref - o).abs().max() / ref.abs().max()).item():.2e}")
