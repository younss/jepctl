//! V-JEPA 2 video encoder (Meta, 2025) in pure Candle.
//!
//! Architecture, as in `transformers.models.vjepa2` (`VJEPA2Model.encoder`):
//! - 3D tubelet embedding: `Conv3d(3, D, kernel = stride = (tubelet, patch, patch))`,
//!   implemented here as an unfold + linear projection (Candle has no Conv3d).
//! - No CLS token, no positional embedding table: **3D rotary embeddings** inside every
//!   attention layer, over the (frame, height, width) token coordinates.
//! - Pre-norm blocks `x + attn(norm1(x))`, `x + mlp(norm2(x))`, GELU, LayerNorm ε=1e-6.
//! - Final LayerNorm; the pooled representation is the mean over all space-time tokens.
//!
//! The rotary code intentionally mirrors the reference quirk: the sin/cos tables are
//! *tiled* (`[s, s]`) while the half-rotation is *interleaved* (`(-x₂, x₁)` per pair).
//! Changing either would silently break compatibility with the released weights.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder, VarMap};

use crate::engine::vit::map_checkpoint_vars;
use crate::types::{JepaError, ModelManifest, WeightReport};

/// Structural parameters of a V-JEPA 2 encoder.
#[derive(Debug, Clone)]
pub struct VJepa2Config {
    pub img_size: usize,
    pub patch_size: usize,
    pub tubelet_size: usize,
    pub in_chans: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
}

impl VJepa2Config {
    pub fn from_manifest(m: &ModelManifest) -> Self {
        Self {
            img_size: m.image_size,
            patch_size: m.patch_size,
            tubelet_size: m.tubelet_size.unwrap_or(2),
            in_chans: 3,
            embed_dim: m.embed_dim,
            depth: m.num_layers,
            num_heads: m.num_heads,
            mlp_ratio: m.mlp_ratio(),
        }
    }

    pub fn grid_size(&self) -> usize {
        self.img_size / self.patch_size
    }
}

/// Rotary tables for one token layout (number of temporal slots × grid × grid).
struct RopeTables {
    tokens: usize,
    /// One `(cos, sin)` pair per axis (frame, height, width), each `[N, axis_dim]`.
    axes: Vec<(Tensor, Tensor)>,
    axis_dim: usize,
}

struct RopeAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl RopeAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: linear(dim, dim, vb.pp("q_proj"))?,
            k_proj: linear(dim, dim, vb.pp("k_proj"))?,
            v_proj: linear(dim, dim, vb.pp("v_proj"))?,
            out_proj: linear(dim, dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Reference rotation: pairs `(x[2i], x[2i+1]) -> (-x[2i+1], x[2i])`, tables tiled.
    fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, h, n, d) = x.dims4()?;
        let pairs = x.reshape((b, h, n, d / 2, 2))?;
        let y1 = pairs.narrow(4, 0, 1)?;
        let y2 = pairs.narrow(4, 1, 1)?;
        let rotated = Tensor::cat(&[&y2.neg()?, &y1], 4)?.reshape((b, h, n, d))?;
        x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
    }

    fn apply_rope(&self, qk: &Tensor, tables: &RopeTables) -> candle_core::Result<Tensor> {
        let d = tables.axis_dim;
        let mut parts = Vec::with_capacity(4);
        let mut offset = 0;
        for (cos, sin) in &tables.axes {
            let slice = qk.narrow(D::Minus1, offset, d)?;
            parts.push(Self::rotate(&slice, cos, sin)?);
            offset += d;
        }
        if offset < self.head_dim {
            parts.push(qk.narrow(D::Minus1, offset, self.head_dim - offset)?);
        }
        Tensor::cat(&parts, D::Minus1)
    }

    fn forward(&self, x: &Tensor, tables: &RopeTables) -> candle_core::Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let heads = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = self.apply_rope(&heads(self.q_proj.forward(x)?)?, tables)?;
        let k = self.apply_rope(&heads(self.k_proj.forward(x)?)?, tables)?;
        let v = heads(self.v_proj.forward(x)?)?;

        // Attention one head at a time: a clip is thousands of tokens, and a full
        // [B, heads, N, N] score tensor per layer (hundreds of MB each, cached by the
        // Metal allocator) is what blew memory up. Per head the working set is N²·4 B.
        let mut ctx_heads = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let qh = q.narrow(1, h, 1)?;
            let kh = k.narrow(1, h, 1)?.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            let vh = v.narrow(1, h, 1)?;
            let scores = (qh.matmul(&kh)? * self.scale)?;
            let attn = candle_nn::ops::softmax_last_dim(&scores)?;
            ctx_heads.push(attn.matmul(&vh)?); // [B, 1, N, head_dim]
        }
        let ctx = Tensor::cat(&ctx_heads, 1)?.transpose(1, 2)?.reshape((b, n, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&ctx)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: RopeAttention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn new(cfg: &VJepa2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
        Ok(Self {
            norm1: layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?,
            attn: RopeAttention::new(cfg.embed_dim, cfg.num_heads, vb.pp("attn"))?,
            norm2: layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?,
            fc1: linear(cfg.embed_dim, hidden, vb.pp("mlp").pp("fc1"))?,
            fc2: linear(hidden, cfg.embed_dim, vb.pp("mlp").pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, tables: &RopeTables) -> candle_core::Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?, tables)?)?;
        let h = candle_nn::Activation::Gelu.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?)?;
        &x + self.fc2.forward(&h)?
    }
}

/// `(pooled embedding, patch tokens of the last temporal slot, latency in ms)`.
pub type EncoderOutput = (Vec<f32>, Vec<Vec<f32>>, f64);

/// V-JEPA 2 encoder instance.
pub struct VJepa2Model {
    pub manifest: ModelManifest,
    pub cfg: VJepa2Config,
    pub device: Device,
    pub weights: WeightReport,
    patch_proj: Linear,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl VJepa2Model {
    fn build(manifest: ModelManifest, device: &Device) -> Result<(VarMap, Self), JepaError> {
        let cfg = VJepa2Config::from_manifest(&manifest);
        if !cfg.img_size.is_multiple_of(cfg.patch_size) {
            return Err(JepaError::InvalidPayload("image_size must be a multiple of patch_size".into()));
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let patch_in = cfg.in_chans * cfg.tubelet_size * cfg.patch_size * cfg.patch_size;
        let patch_proj = linear(patch_in, cfg.embed_dim, vb.pp("patch_embed").pp("proj"))?;
        let blocks_vb = vb.pp("blocks");
        let blocks =
            (0..cfg.depth).map(|i| Block::new(&cfg, blocks_vb.pp(i))).collect::<candle_core::Result<Vec<_>>>()?;
        let norm = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm"))?;
        let expected = varmap.data().lock().map(|d| d.len()).unwrap_or(0);
        let model = Self {
            manifest,
            cfg,
            device: device.clone(),
            weights: WeightReport { loaded: 0, expected, source: "random".into() },
            patch_proj,
            blocks,
            norm,
        };
        Ok((varmap, model))
    }

    /// Load from a Hugging Face `VJEPA2Model` safetensors file (encoder weights only;
    /// the predictor is ignored). Fails unless every encoder parameter is covered.
    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let name = manifest.name.clone();
        if !weights_path.is_file() {
            return Err(JepaError::ModelNotFound(format!(
                "Weights for '{}' not found at {}. Run `jepa pull {}` first.",
                name,
                weights_path.display(),
                name
            )));
        }
        let (varmap, mut model) = Self::build(manifest, &device)?;
        let mmap = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(weights_path)
                .map_err(|e| JepaError::InferenceError(format!("Could not mmap safetensors: {}", e)))?
        };
        let (loaded, missing) = map_checkpoint_vars(&varmap, &mmap, &device)?;
        let expected = model.weights.expected;
        if loaded != expected {
            let preview: Vec<&str> = missing.iter().take(5).map(String::as_str).collect();
            tracing::error!(
                "Checkpoint {} covers {}/{} parameters of '{}'. Missing (first 5): {:?}",
                weights_path.display(),
                loaded,
                expected,
                name,
                preview
            );
            return Err(JepaError::WeightsIncomplete { loaded, expected });
        }
        tracing::info!("Loaded {}/{} tensors into '{}' from {}", loaded, expected, name, weights_path.display());
        model.weights = WeightReport { loaded, expected, source: "safetensors".into() };
        Ok(model)
    }

    /// Randomly initialised instance for tests.
    #[doc(hidden)]
    pub fn load_random(manifest: ModelManifest, device: Device) -> Result<Self, JepaError> {
        Ok(Self::build(manifest, &device)?.1)
    }

    /// Rotary tables for `depth × grid × grid` tokens, in the reference layout
    /// (frame-major, then row, then column).
    fn rope_tables(&self, depth: usize) -> candle_core::Result<RopeTables> {
        let head_dim = self.cfg.embed_dim / self.cfg.num_heads;
        let axis_dim = 2 * ((head_dim / 3) / 2);
        let g = self.cfg.grid_size();
        let tokens = depth * g * g;
        let half = axis_dim / 2;
        let omega: Vec<f32> = (0..half).map(|j| 1.0 / 10000f32.powf(j as f32 / half as f32)).collect();

        let table = |pos_of: &dyn Fn(usize) -> usize| -> candle_core::Result<(Tensor, Tensor)> {
            let mut cos = Vec::with_capacity(tokens * axis_dim);
            let mut sin = Vec::with_capacity(tokens * axis_dim);
            for idx in 0..tokens {
                let p = pos_of(idx) as f32;
                // Tiled, not interleaved: element j uses omega[j mod half].
                for j in 0..axis_dim {
                    let f = p * omega[j % half];
                    cos.push(f.cos());
                    sin.push(f.sin());
                }
            }
            Ok((
                Tensor::from_vec(cos, (tokens, axis_dim), &self.device)?,
                Tensor::from_vec(sin, (tokens, axis_dim), &self.device)?,
            ))
        };
        let per_frame = g * g;
        let axes = vec![table(&|i| i / per_frame)?, table(&|i| (i % per_frame) / g)?, table(&|i| (i % per_frame) % g)?];
        Ok(RopeTables { tokens, axes, axis_dim })
    }

    /// Tubelet embedding of `[B, C, T, H, W]` → `[B, (T/t)·(H/p)·(W/p), D]` (frame-major).
    fn tubelet_embed(&self, video: &Tensor) -> candle_core::Result<(Tensor, usize)> {
        let (b, c, t, h, w) = video.dims5()?;
        let (ts, p) = (self.cfg.tubelet_size, self.cfg.patch_size);
        let (dt, gh, gw) = (t / ts, h / p, w / p);
        // [B, C, dt, ts, gh, p, gw, p] -> [B, dt, gh, gw, C, ts, p, p] -> [B, N, C·ts·p·p]
        let x = video
            .reshape(vec![b, c, dt, ts, gh, p, gw, p])?
            .permute([0, 2, 4, 6, 1, 3, 5, 7])?
            .contiguous()?
            .reshape((b, dt * gh * gw, c * ts * p * p))?;
        Ok((self.patch_proj.forward(&x)?, dt))
    }

    /// Forward over `[B, C, T, H, W]` (T must be a multiple of the tubelet size).
    /// Returns `(pooled [D], last-temporal-slot patch tokens [grid², D], latency_ms)`.
    pub fn forward_video(&self, video: &Tensor) -> Result<EncoderOutput, JepaError> {
        let start = Instant::now();
        let video = video.to_device(&self.device)?;
        let (_b, _c, t, h, w) = video.dims5()?;
        if t % self.cfg.tubelet_size != 0 || h != self.cfg.img_size || w != self.cfg.img_size {
            return Err(JepaError::InvalidPayload(format!(
                "V-JEPA 2 expects [B, 3, k×{}, {}, {}] inputs, got T={} H={} W={}",
                self.cfg.tubelet_size, self.cfg.img_size, self.cfg.img_size, t, h, w
            )));
        }

        let (mut tokens, depth) = self.tubelet_embed(&video)?;
        let tables = self.rope_tables(depth)?;
        debug_assert_eq!(tables.tokens, tokens.dim(1)?);
        for block in &self.blocks {
            tokens = block.forward(&tokens, &tables)?;
        }
        let normalized = self.norm.forward(&tokens)?; // [B, N, D]

        let pooled: Vec<f32> = normalized.mean(1)?.squeeze(0)?.to_vec1()?;
        let per_frame = self.cfg.grid_size().pow(2);
        let last = normalized.narrow(1, (depth - 1) * per_frame, per_frame)?.squeeze(0)?;
        let patches: Vec<Vec<f32>> = last.to_vec2()?;

        Ok((pooled, patches, start.elapsed().as_secs_f64() * 1000.0))
    }

    /// A still image is embedded as a clip of `tubelet_size` identical frames.
    pub fn forward_image(&self, img: &Tensor) -> Result<EncoderOutput, JepaError> {
        let repeated = img.unsqueeze(2)?.repeat((1, 1, self.cfg.tubelet_size, 1, 1))?;
        self.forward_video(&repeated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelModality;

    fn manifest(frames: usize) -> ModelManifest {
        ModelManifest {
            name: "test/vjepa2-tiny".into(),
            repo_id: "test/vjepa2-tiny".into(),
            architecture: "V-JEPA 2 tiny".into(),
            modality: ModelModality::Video,
            patch_size: 16,
            embed_dim: 48,
            num_layers: 2,
            num_heads: 4,
            image_size: 32,
            frames: Some(frames),
            parameter_count: "tiny".into(),
            disk_size_bytes: 0,
            weights_file: "model.safetensors".into(),
            created_at: chrono::Utc::now(),
            variant: None,
            normalization: None,
            mlp_ratio: None,
            tubelet_size: Some(2),
            input_width: None,
            in_chans: None,
            audio: None,
            pooling: None,
        }
    }

    #[test]
    fn shapes_and_determinism() {
        let m = VJepa2Model::load_random(manifest(4), Device::Cpu).unwrap();
        // 2 + 2 blocks × (2+8+2+4) + 2 = 36 parameters
        assert_eq!(m.weights.expected, 2 + 2 * 16 + 2);
        let clip = Tensor::randn(0f32, 1.0, (1, 3, 4, 32, 32), &Device::Cpu).unwrap();
        let (pooled, patches, _) = m.forward_video(&clip).unwrap();
        assert_eq!(pooled.len(), 48);
        assert_eq!(patches.len(), 4); // 2x2 grid of the last temporal slot
        let (pooled2, _, _) = m.forward_video(&clip).unwrap();
        assert_eq!(pooled, pooled2);
        // Still image path: 2 identical frames.
        let img = Tensor::randn(0f32, 1.0, (1, 3, 32, 32), &Device::Cpu).unwrap();
        let (p, _, _) = m.forward_image(&img).unwrap();
        assert_eq!(p.len(), 48);
    }

    #[test]
    fn rejects_bad_temporal_length() {
        let m = VJepa2Model::load_random(manifest(4), Device::Cpu).unwrap();
        let clip = Tensor::zeros((1, 3, 3, 32, 32), DType::F32, &Device::Cpu).unwrap();
        assert!(m.forward_video(&clip).is_err());
    }

    #[test]
    fn rope_tables_follow_reference_layout() {
        let m = VJepa2Model::load_random(manifest(4), Device::Cpu).unwrap();
        let t = m.rope_tables(2).unwrap();
        // head_dim 12 -> axis_dim = 2 * ((12/3)/2) = 4
        assert_eq!(t.axis_dim, 4);
        assert_eq!(t.tokens, 8);
        let frame_cos: Vec<f32> = t.axes[0].0.flatten_all().unwrap().to_vec1().unwrap();
        // Tokens 0..4 are frame 0 (cos 0 = 1), tokens 4..8 are frame 1.
        assert!(frame_cos[..16].iter().all(|v| (v - 1.0).abs() < 1e-6));
        assert!((frame_cos[16] - 1f32.cos()).abs() < 1e-6);
        // Tiled tables: element j and j + half share the angle.
        assert!((frame_cos[16] - frame_cos[18]).abs() < 1e-6);
        let w_cos: Vec<f32> = t.axes[2].0.flatten_all().unwrap().to_vec1().unwrap();
        // Width position of token 1 is 1, of token 2 (next row) is 0.
        assert!((w_cos[4] - 1f32.cos()).abs() < 1e-6);
        assert!((w_cos[8] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_matches_reference_formula() {
        // x = [1, 2, 3, 4], cos = c, sin = s (tiled) → x*c + [-2, 1, -4, 3]*s
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &dev).unwrap();
        let cos = Tensor::from_vec(vec![0.5f32, 0.6, 0.5, 0.6], (1, 4), &dev).unwrap();
        let sin = Tensor::from_vec(vec![0.1f32, 0.2, 0.1, 0.2], (1, 4), &dev).unwrap();
        let out: Vec<f32> = RopeAttention::rotate(&x, &cos, &sin).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let expect = [1.0 * 0.5 + -2.0 * 0.1, 2.0 * 0.6 + 1.0 * 0.2, 3.0 * 0.5 + -4.0 * 0.1, 4.0 * 0.6 + 3.0 * 0.2];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{out:?} vs {expect:?}");
        }
    }
}
