#!/usr/bin/env python3
# Usage: jepctl embed clip.wav --model ltuncay/Audio-JEPA-base --format json > ours.json
#        python audio_jepa.py ~/.jepctl/models/ltuncay/Audio-JEPA-base/model.safetensors clip.wav ours.json
# The clip must be 16 kHz mono 16-bit PCM and exactly 10 s long (the model window).
# Requires: torch, torchaudio, safetensors, numpy.
# Independent reference for ltuncay/Audio-JEPA-base written from the published config:
# torchaudio MelSpectrogram + AmplitudeToDB, then a pre-norm ViT read from the checkpoint.
import json, sys, wave
import numpy as np, torch, torch.nn.functional as F, torchaudio
from safetensors.torch import load_file

ckpt, wav_path, ours_path = sys.argv[1:4]
W = load_file(ckpt)
with wave.open(wav_path) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
x = torch.from_numpy(x)[None, None]  # [B, C, T]

mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, win_length=2048, hop_length=625,
                                           n_mels=128, f_min=0.0, f_max=8000.0, power=2.0, normalized=True)
spec = torchaudio.transforms.AmplitudeToDB()(mel(x))  # [1, 1, 128, 257]
spec = spec[..., : spec.shape[-1] // 16 * 16]          # truncate mode -> 256 frames

p = "adapter."
t = F.conv2d(spec, W[p + "patch_embed.patch_embed.proj.weight"], W[p + "patch_embed.patch_embed.proj.bias"], stride=16)
t = t.flatten(2).transpose(1, 2) + W[p + "encoder.pos_embed"]
ln = lambda h, n: F.layer_norm(h, (768,), W[n + ".weight"], W[n + ".bias"], eps=1e-5)
lin = lambda h, n: F.linear(h, W[n + ".weight"], W[n + ".bias"])
for i in range(12):
    b = f"{p}encoder.blocks.{i}."
    h = ln(t, b + "norm1")
    q, k, v = lin(h, b + "attn.qkv").view(1, -1, 3, 12, 64).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, -1, 768)
    t = t + lin(a, b + "attn.proj")
    t = t + lin(F.gelu(lin(ln(t, b + "norm2"), b + "mlp.fc1")), b + "mlp.fc2")
t = ln(t, p + "encoder.norm")[0]
ref = t.mean(0)

ours = json.load(open(ours_path))
o = torch.tensor(ours["embedding"])
cos = F.cosine_similarity(ref, o, dim=0).item()
rel = ((ref - o).abs().max() / ref.abs().max()).item()
print(f"{wav_path.split('/')[-1]}: pooled cos={cos:.7f} rel_max_err={rel:.2e}")
