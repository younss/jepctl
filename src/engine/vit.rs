//! Vision Transformer (ViT) backbone implemented in pure Candle.
//!
//! One backbone serves every supported checkpoint family; the differences are
//! captured by [`VitVariant`]:
//!
//! | variant  | CLS token | LayerScale | pooled output | checkpoints                    |
//! |----------|-----------|------------|---------------|--------------------------------|
//! | `Plain`  | no        | no         | mean of patches | I-JEPA (HF), V-JEPA-style    |
//! | `Cls`    | yes       | no         | CLS token     | HF `ViTModel`, timm ViT        |
//! | `DinoV2` | yes       | yes        | CLS token     | HF `Dinov2Model`               |
//!
//! [`load_safetensors_into_backbone`] maps the four tensor namings encountered in
//! the wild (HF ViT/I-JEPA, HF DINOv2, timm/Meta with fused QKV) onto the backbone
//! and reports exactly which parameters were covered.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::types::JepaError;

/// Structural variant of the ViT backbone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VitVariant {
    /// Patch tokens only, mean-pooled (I-JEPA, V-JEPA).
    #[default]
    Plain,
    /// Learned CLS token, no LayerScale (HF `ViTModel`, timm ViT).
    Cls,
    /// CLS token + LayerScale on both residual branches (DINOv2).
    DinoV2,
    /// 3D tubelet embedding + rotary attention, no CLS (V-JEPA 2). Not a 2D backbone:
    /// handled by [`crate::engine::vjepa2::VJepa2Model`].
    VJepa2,
}

impl VitVariant {
    pub fn has_cls(self) -> bool {
        !matches!(self, VitVariant::Plain)
    }

    pub fn has_layer_scale(self) -> bool {
        matches!(self, VitVariant::DinoV2)
    }
}

/// How the pooled representation is read out of the final tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Mean of the patch tokens (I-JEPA, MAE-style encoders such as AudioMAE).
    Mean,
    /// The CLS token (supervised ViT, DINOv2).
    Cls,
}

/// Everything needed to instantiate a backbone.
#[derive(Debug, Clone)]
pub struct VitConfig {
    /// Input height (rows). Equal to `img_w` for images; e.g. 1024 mel frames for audio.
    pub img_size: usize,
    /// Input width (columns). Same as `img_size` unless set explicitly.
    pub img_w: Option<usize>,
    pub patch_size: usize,
    pub in_chans: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub variant: VitVariant,
    pub pooling: Pooling,
}

/// 2D Patch embedding module using convolution
#[derive(Debug)]
pub struct PatchEmbed {
    proj: Conv2d,
    pub patch_size: usize,
    pub embed_dim: usize,
    /// Patch grid rows (height / patch).
    pub grid_size: usize,
    /// Patch grid columns (width / patch).
    pub grid_w: usize,
    pub num_patches: usize,
}

impl PatchEmbed {
    pub fn new(
        img_h: usize,
        img_w: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig { stride: patch_size, padding: 0, dilation: 1, groups: 1, ..Default::default() };
        let proj = conv2d(in_chans, embed_dim, patch_size, cfg, vb.pp("proj"))?;
        let grid_size = img_h / patch_size;
        let grid_w = img_w / patch_size;

        Ok(Self { proj, patch_size, embed_dim, grid_size, grid_w, num_patches: grid_size * grid_w })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, H, W] -> [B, num_patches, embed_dim]
        let feat = self.proj.forward(x)?; // [B, D, gh, gw]
        feat.flatten(2, 3)?.transpose(1, 2)
    }
}

/// Multi-Head Self-Attention
#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj: linear(embed_dim, embed_dim, vb.pp("q_proj"))?,
            k_proj: linear(embed_dim, embed_dim, vb.pp("k_proj"))?,
            v_proj: linear(embed_dim, embed_dim, vb.pp("v_proj"))?,
            out_proj: linear(embed_dim, embed_dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = split_heads(self.q_proj.forward(x)?)?;
        let k = split_heads(self.k_proj.forward(x)?)?;
        let v = split_heads(self.v_proj.forward(x)?)?;

        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(&v)?; // [B, heads, N, head_dim]
        let context = context.transpose(1, 2)?.reshape((b, n, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&context)
    }
}

/// Feedforward MLP module
#[derive(Debug)]
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(embed_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(embed_dim, hidden_dim, vb.pp("fc1"))?,
            fc2: linear(hidden_dim, embed_dim, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_nn::Activation::Gelu.forward(&self.fc1.forward(x)?)?;
        self.fc2.forward(&h)
    }
}

/// Pre-norm Transformer encoder block with optional LayerScale.
#[derive(Debug)]
pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    /// LayerScale gains (DINOv2): `ls1` scales the attention branch, `ls2` the MLP branch.
    ls1: Option<Tensor>,
    ls2: Option<Tensor>,
}

impl Block {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_ratio: f64, layer_scale: bool, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (embed_dim as f64 * mlp_ratio) as usize;
        let (ls1, ls2) = if layer_scale {
            (
                Some(vb.get_with_hints(embed_dim, "ls1", candle_nn::Init::Const(1.0))?),
                Some(vb.get_with_hints(embed_dim, "ls2", candle_nn::Init::Const(1.0))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            norm1: layer_norm(embed_dim, 1e-6, vb.pp("norm1"))?,
            attn: Attention::new(embed_dim, num_heads, vb.pp("attn"))?,
            norm2: layer_norm(embed_dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(embed_dim, hidden_dim, vb.pp("mlp"))?,
            ls1,
            ls2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut a = self.attn.forward(&self.norm1.forward(x)?)?;
        if let Some(ls) = &self.ls1 {
            a = a.broadcast_mul(ls)?;
        }
        let x = (x + a)?;
        let mut m = self.mlp.forward(&self.norm2.forward(&x)?)?;
        if let Some(ls) = &self.ls2 {
            m = m.broadcast_mul(ls)?;
        }
        x + m
    }
}

/// Full Vision Transformer Backbone
#[derive(Debug)]
pub struct VitBackbone {
    pub patch_embed: PatchEmbed,
    pub blocks: Vec<Block>,
    pub norm: LayerNorm,
    pub embed_dim: usize,
    pub variant: VitVariant,
    pub pooling: Pooling,
    /// Learned CLS token `[1, 1, D]` for CLS variants.
    pub cls_token: Option<Tensor>,
    /// Positional embedding `[1, N(+1), D]`; sin-cos by default, replaced by the checkpoint when present.
    pub pos_embed: Tensor,
}

impl VitBackbone {
    pub fn new(cfg: &VitConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            cfg.img_size,
            cfg.img_w.unwrap_or(cfg.img_size),
            cfg.patch_size,
            cfg.in_chans,
            cfg.embed_dim,
            vb.pp("patch_embed"),
        )?;
        let blocks_vb = vb.pp("blocks");
        let blocks = (0..cfg.depth)
            .map(|i| {
                Block::new(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.variant.has_layer_scale(), blocks_vb.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm"))?;

        let cls_token = if cfg.variant.has_cls() {
            Some(vb.get_with_hints((1, 1, cfg.embed_dim), "cls_token", candle_nn::Init::Const(0.0))?)
        } else {
            None
        };

        let mut pos_embed =
            generate_sincos_pos_embed(patch_embed.grid_size, patch_embed.grid_w, cfg.embed_dim, vb.device())?;
        if cfg.variant.has_cls() {
            let zero = Tensor::zeros((1, 1, cfg.embed_dim), pos_embed.dtype(), vb.device())?;
            pos_embed = Tensor::cat(&[&zero, &pos_embed], 1)?;
        }

        Ok(Self {
            patch_embed,
            blocks,
            norm,
            embed_dim: cfg.embed_dim,
            variant: cfg.variant,
            pooling: cfg.pooling,
            cls_token,
            pos_embed,
        })
    }

    /// Forward pass returning `(patch tokens [B, N, D], pooled [B, D])`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = x.dim(0)?;
        let mut tokens = self.patch_embed.forward(x)?; // [B, N, D]

        if let Some(cls) = &self.cls_token {
            let cls = cls.expand((b, 1, self.embed_dim))?;
            tokens = Tensor::cat(&[&cls, &tokens], 1)?;
        }
        tokens = tokens.broadcast_add(&self.pos_embed)?;

        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        let normalized = self.norm.forward(&tokens)?;

        let (cls, patches) = if self.cls_token.is_some() {
            let n = normalized.dim(1)?;
            (Some(normalized.narrow(1, 0, 1)?.squeeze(1)?), normalized.narrow(1, 1, n - 1)?.contiguous()?)
        } else {
            (None, normalized)
        };
        let pooled = match (self.pooling, cls) {
            (Pooling::Cls, Some(cls)) => cls,
            // Mean pooling, or CLS requested on a backbone without one.
            _ => patches.mean(1)?,
        };
        Ok((patches, pooled))
    }
}

/// Canonical 2D sine-cosine positional embedding `[1, gh·gw, D]` (MAE / I-JEPA style).
fn generate_sincos_pos_embed(grid_h: usize, grid_w: usize, embed_dim: usize, device: &Device) -> Result<Tensor> {
    let num_patches = grid_h * grid_w;
    let mut data = Vec::with_capacity(num_patches * embed_dim);
    let omega_len = embed_dim / 4;

    for h in 0..grid_h {
        for w in 0..grid_w {
            let mut token = Vec::with_capacity(embed_dim);
            for i in 0..omega_len {
                let omega = 1.0 / 10000f64.powf(i as f64 / omega_len as f64);
                token.push((h as f64 * omega).sin() as f32);
                token.push((h as f64 * omega).cos() as f32);
                token.push((w as f64 * omega).sin() as f32);
                token.push((w as f64 * omega).cos() as f32);
            }
            token.resize(embed_dim, 0.0);
            data.extend(token);
        }
    }
    Tensor::from_vec(data, (1, num_patches, embed_dim), device)
}

/// Outcome of mapping a checkpoint onto the backbone.
#[derive(Debug, Clone)]
pub struct LoadOutcome {
    /// Parameters successfully copied from the checkpoint.
    pub loaded: usize,
    /// Parameters the backbone owns (all of them must be loaded for a usable model).
    pub expected: usize,
    /// Backbone parameter names that had no counterpart in the checkpoint.
    pub missing: Vec<String>,
    /// Whether positional embeddings were taken from the checkpoint.
    pub pos_embed_loaded: bool,
}

/// Where a backbone parameter may come from inside a checkpoint.
enum Source {
    Named(String),
    /// Row-slice `index` (0 = q, 1 = k, 2 = v) of a fused `qkv` tensor.
    FusedQkv {
        name: String,
        index: usize,
    },
}

/// All checkpoint names that may hold a given backbone parameter.
fn candidate_sources(target: &str) -> Vec<Source> {
    let prefixes = ["", "vit.", "dinov2.", "model.", "encoder."];
    let mut out = vec![Source::Named(target.to_string())];
    let mut push_named = |name: String| {
        for p in prefixes {
            out.push(Source::Named(format!("{p}{name}")));
        }
    };

    match target {
        "patch_embed.proj.weight" => {
            push_named("embeddings.patch_embeddings.projection.weight".into());
            push_named("embeddings.patch_embeddings.proj.weight".into());
        }
        "patch_embed.proj.bias" => {
            push_named("embeddings.patch_embeddings.projection.bias".into());
            push_named("embeddings.patch_embeddings.proj.bias".into());
        }
        "norm.weight" => push_named("layernorm.weight".into()),
        "norm.bias" => push_named("layernorm.bias".into()),
        "cls_token" => push_named("embeddings.cls_token".into()),
        _ => {
            if let Some(rest) = target.strip_prefix("blocks.") {
                if let Some((idx, sub)) = rest.split_once('.') {
                    // HF ViT / I-JEPA naming
                    let hf_vit = match sub {
                        "norm1.weight" => Some("layernorm_before.weight"),
                        "norm1.bias" => Some("layernorm_before.bias"),
                        "norm2.weight" => Some("layernorm_after.weight"),
                        "norm2.bias" => Some("layernorm_after.bias"),
                        "mlp.fc1.weight" => Some("intermediate.dense.weight"),
                        "mlp.fc1.bias" => Some("intermediate.dense.bias"),
                        "mlp.fc2.weight" => Some("output.dense.weight"),
                        "mlp.fc2.bias" => Some("output.dense.bias"),
                        _ => None,
                    };
                    // HF V-JEPA 2: `attention.{query,key,value,proj}` directly under the layer
                    let hf_vjepa2 = match sub {
                        "attn.q_proj.weight" => Some("attention.query.weight"),
                        "attn.q_proj.bias" => Some("attention.query.bias"),
                        "attn.k_proj.weight" => Some("attention.key.weight"),
                        "attn.k_proj.bias" => Some("attention.key.bias"),
                        "attn.v_proj.weight" => Some("attention.value.weight"),
                        "attn.v_proj.bias" => Some("attention.value.bias"),
                        "attn.out_proj.weight" => Some("attention.proj.weight"),
                        "attn.out_proj.bias" => Some("attention.proj.bias"),
                        _ => None,
                    };
                    // Shared by HF ViT and HF DINOv2
                    let hf_attn = match sub {
                        "attn.q_proj.weight" => Some("attention.attention.query.weight"),
                        "attn.q_proj.bias" => Some("attention.attention.query.bias"),
                        "attn.k_proj.weight" => Some("attention.attention.key.weight"),
                        "attn.k_proj.bias" => Some("attention.attention.key.bias"),
                        "attn.v_proj.weight" => Some("attention.attention.value.weight"),
                        "attn.v_proj.bias" => Some("attention.attention.value.bias"),
                        "attn.out_proj.weight" => Some("attention.output.dense.weight"),
                        "attn.out_proj.bias" => Some("attention.output.dense.bias"),
                        _ => None,
                    };
                    // HF DINOv2 naming (norm1/norm2/mlp keep timm names; LayerScale is separate)
                    let hf_dino = match sub {
                        "ls1" => Some("layer_scale1.lambda1"),
                        "ls2" => Some("layer_scale2.lambda1"),
                        "norm1.weight" | "norm1.bias" | "norm2.weight" | "norm2.bias" | "mlp.fc1.weight"
                        | "mlp.fc1.bias" | "mlp.fc2.weight" | "mlp.fc2.bias" => Some(sub),
                        _ => None,
                    };
                    for name in [hf_vit, hf_attn, hf_dino, hf_vjepa2].into_iter().flatten() {
                        push_named(format!("encoder.layer.{idx}.{name}"));
                    }
                    // timm / Meta naming: fused qkv, `attn.proj`, `ls*.gamma`
                    let fused = |what: &str, index: usize| Source::FusedQkv {
                        name: format!("blocks.{idx}.attn.qkv.{what}"),
                        index,
                    };
                    match sub {
                        "attn.q_proj.weight" => out.push(fused("weight", 0)),
                        "attn.k_proj.weight" => out.push(fused("weight", 1)),
                        "attn.v_proj.weight" => out.push(fused("weight", 2)),
                        "attn.q_proj.bias" => out.push(fused("bias", 0)),
                        "attn.k_proj.bias" => out.push(fused("bias", 1)),
                        "attn.v_proj.bias" => out.push(fused("bias", 2)),
                        "attn.out_proj.weight" => out.push(Source::Named(format!("blocks.{idx}.attn.proj.weight"))),
                        "attn.out_proj.bias" => out.push(Source::Named(format!("blocks.{idx}.attn.proj.bias"))),
                        "ls1" => out.push(Source::Named(format!("blocks.{idx}.ls1.gamma"))),
                        "ls2" => out.push(Source::Named(format!("blocks.{idx}.ls2.gamma"))),
                        _ => {}
                    }
                }
            }
        }
    }
    out
}

/// Bicubic (Catmull-Rom) resampling of a `[1, S*S, D]` positional grid to `[1, T*T, D]`.
fn resample_pos_grid(pos: &Tensor, src: usize, dst: usize, device: &Device) -> Result<Tensor> {
    let (_, _, d) = pos.dims3()?;
    let data: Vec<f32> = pos.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; dst * dst * d];

    let cubic = |t: f32| -> f32 {
        // Catmull-Rom kernel (a = -0.5)
        let t = t.abs();
        if t < 1.0 {
            1.5 * t * t * t - 2.5 * t * t + 1.0
        } else if t < 2.0 {
            -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
        } else {
            0.0
        }
    };
    let scale = src as f32 / dst as f32;

    for ty in 0..dst {
        let sy = (ty as f32 + 0.5) * scale - 0.5;
        let y0 = sy.floor() as isize;
        for tx in 0..dst {
            let sx = (tx as f32 + 0.5) * scale - 0.5;
            let x0 = sx.floor() as isize;
            let o = (ty * dst + tx) * d;
            let mut wsum = 0f32;
            for dy in -1..=2isize {
                let yy = (y0 + dy).clamp(0, src as isize - 1) as usize;
                let wy = cubic(sy - (y0 + dy) as f32);
                for dx in -1..=2isize {
                    let xx = (x0 + dx).clamp(0, src as isize - 1) as usize;
                    let w = wy * cubic(sx - (x0 + dx) as f32);
                    if w == 0.0 {
                        continue;
                    }
                    wsum += w;
                    let i = (yy * src + xx) * d;
                    for c in 0..d {
                        out[o + c] += w * data[i + c];
                    }
                }
            }
            if wsum.abs() > 1e-6 {
                for c in 0..d {
                    out[o + c] /= wsum;
                }
            }
        }
    }
    Tensor::from_vec(out, (1, dst * dst, d), device)
}

/// Adapt a checkpoint positional embedding `[1, M(+1), D]` to the backbone's grid,
/// interpolating between resolutions and adding/removing the CLS row as required.
fn adapt_pos_embed(
    ckpt: &Tensor,
    grid_h: usize,
    grid_w: usize,
    has_cls: bool,
    device: &Device,
) -> std::result::Result<Tensor, JepaError> {
    let (_, m, _d) = ckpt.dims3()?;
    let target = grid_h * grid_w;
    let is_square = |n: usize| ((n as f64).sqrt() as usize).pow(2) == n;

    // Exact token-count match (any rectangle): take the rows as they are.
    let (ckpt_cls, src_grid) = if m == target {
        (false, None)
    } else if m == target + 1 {
        (true, None)
    } else if is_square(m) {
        (false, Some((m as f64).sqrt() as usize))
    } else if m >= 1 && is_square(m - 1) {
        (true, Some(((m - 1) as f64).sqrt() as usize))
    } else {
        return Err(JepaError::InferenceError(format!(
            "Positional embedding with {m} rows matches neither the {grid_h}x{grid_w} grid nor a square grid"
        )));
    };

    let (cls_row, patch_rows) = if ckpt_cls {
        (Some(ckpt.narrow(1, 0, 1)?), ckpt.narrow(1, 1, m - 1)?.contiguous()?)
    } else {
        (None, ckpt.clone())
    };

    let patch_rows = match src_grid {
        None => patch_rows,
        Some(src) if grid_h == grid_w => {
            tracing::info!("Resampling positional embedding grid {}x{} -> {}x{}", src, src, grid_h, grid_w);
            resample_pos_grid(&patch_rows, src, grid_h, device)?
        }
        Some(src) => {
            return Err(JepaError::InferenceError(format!(
                "Cannot resample a {src}x{src} positional grid to a non-square {grid_h}x{grid_w} grid"
            )))
        }
    };

    if !has_cls {
        return Ok(patch_rows);
    }
    let cls_row = match cls_row {
        Some(c) => c,
        None => Tensor::zeros((1, 1, patch_rows.dim(2)?), patch_rows.dtype(), device)?,
    };
    Ok(Tensor::cat(&[&cls_row, &patch_rows], 1)?)
}

/// Load and map safetensors weights across common Hugging Face, timm and Meta ViT formats.
pub fn load_safetensors_into_backbone(
    varmap: &candle_nn::VarMap,
    backbone: &mut VitBackbone,
    path: &std::path::Path,
    device: &Device,
) -> std::result::Result<LoadOutcome, JepaError> {
    let mmap = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(path)
            .map_err(|e| JepaError::InferenceError(format!("Could not mmap safetensors: {}", e)))?
    };

    let (loaded, missing) = map_checkpoint_vars(varmap, &mmap, device)?;

    let pos_names = [
        "pos_embed",
        "embeddings.position_embeddings",
        "vit.embeddings.position_embeddings",
        "dinov2.embeddings.position_embeddings",
    ];
    let mut pos_embed_loaded = false;
    if let Some(ckpt_pos) = pos_names.iter().find_map(|n| mmap.load(n, device).ok()) {
        match adapt_pos_embed(
            &ckpt_pos,
            backbone.patch_embed.grid_size,
            backbone.patch_embed.grid_w,
            backbone.cls_token.is_some(),
            device,
        ) {
            Ok(pos) if pos.shape() == backbone.pos_embed.shape() => {
                backbone.pos_embed = pos;
                pos_embed_loaded = true;
            }
            Ok(pos) => tracing::warn!(
                "Adapted positional embedding {:?} does not match {:?}",
                pos.shape(),
                backbone.pos_embed.shape()
            ),
            Err(e) => tracing::warn!("Could not adapt positional embedding: {}", e),
        }
    }

    let expected = varmap.data().lock().map(|d| d.len()).unwrap_or(0);
    Ok(LoadOutcome { loaded, expected, missing, pos_embed_loaded })
}

/// Copy every backbone parameter of `varmap` from the checkpoint, trying all known
/// namings. A tensor with the same element count but another shape (e.g. a Conv3d
/// kernel feeding a Linear) is reshaped. Returns `(loaded, missing names)`.
pub fn map_checkpoint_vars(
    varmap: &candle_nn::VarMap,
    mmap: &candle_core::safetensors::MmapedSafetensors,
    device: &Device,
) -> std::result::Result<(usize, Vec<String>), JepaError> {
    let mut loaded = 0;
    let mut missing = Vec::new();
    let vars = varmap.data().lock().map_err(|_| JepaError::InferenceError("VarMap lock poisoned".into()))?;

    for (target, var) in vars.iter() {
        let mut found = false;
        for source in candidate_sources(target) {
            let tensor = match &source {
                Source::Named(name) => mmap.load(name, device).ok(),
                Source::FusedQkv { name, index } => mmap.load(name, device).ok().and_then(|t| {
                    let rows = t.dim(0).ok()? / 3;
                    t.narrow(0, index * rows, rows).ok()?.contiguous().ok()
                }),
            };
            let Some(tensor) = tensor else { continue };
            let tensor = if var.shape() == tensor.shape() {
                tensor
            } else if var.shape().elem_count() == tensor.shape().elem_count() {
                tracing::debug!("Reshaping '{}' {:?} -> {:?}", target, tensor.shape(), var.shape());
                tensor.reshape(var.shape())?
            } else {
                tracing::warn!(
                    "Shape mismatch for '{}': backbone {:?}, checkpoint {:?}",
                    target,
                    var.shape(),
                    tensor.shape()
                );
                continue;
            };
            var.set(&tensor)?;
            loaded += 1;
            found = true;
            break;
        }
        if !found {
            missing.push(target.clone());
        }
    }
    Ok((loaded, missing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn cfg(variant: VitVariant) -> VitConfig {
        VitConfig {
            img_size: 32,
            img_w: None,
            patch_size: 16,
            in_chans: 3,
            embed_dim: 8,
            depth: 1,
            num_heads: 2,
            mlp_ratio: 4.0,
            variant,
            pooling: if variant.has_cls() { Pooling::Cls } else { Pooling::Mean },
        }
    }

    #[test]
    fn variants_produce_expected_shapes() {
        for variant in [VitVariant::Plain, VitVariant::Cls, VitVariant::DinoV2] {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
            let bb = VitBackbone::new(&cfg(variant), vb).unwrap();
            let x = Tensor::ones((2, 3, 32, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
            let (patches, pooled) = bb.forward(&x).unwrap();
            assert_eq!(patches.dims(), &[2, 4, 8], "{variant:?}");
            assert_eq!(pooled.dims(), &[2, 8], "{variant:?}");
            let n_vars = varmap.data().lock().unwrap().len();
            // patch_embed (2) + final norm (2) + one block (16) [+ cls] [+ ls1, ls2]
            let expected = match variant {
                VitVariant::Plain => 2 + 2 + 16,
                VitVariant::Cls => 2 + 2 + 16 + 1,
                VitVariant::DinoV2 => 2 + 2 + 16 + 1 + 2,
                VitVariant::VJepa2 => unreachable!("not a 2D backbone"),
            };
            assert_eq!(n_vars, expected, "{variant:?}");
        }
    }

    #[test]
    fn rectangular_single_channel_backbone() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let mut c = cfg(VitVariant::Cls);
        c.img_size = 64;
        c.img_w = Some(32);
        c.in_chans = 1;
        let bb = VitBackbone::new(&c, vb).unwrap();
        assert_eq!((bb.patch_embed.grid_size, bb.patch_embed.grid_w, bb.patch_embed.num_patches), (4, 2, 8));
        assert_eq!(bb.pos_embed.dims(), &[1, 9, 8]);
        let x = Tensor::ones((1, 1, 64, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
        let (patches, pooled) = bb.forward(&x).unwrap();
        assert_eq!(patches.dims(), &[1, 8, 8]);
        assert_eq!(pooled.dims(), &[1, 8]);
    }

    #[test]
    fn fused_qkv_and_hf_names_are_candidates() {
        let names: Vec<String> = candidate_sources("blocks.3.attn.k_proj.weight")
            .into_iter()
            .map(|s| match s {
                Source::Named(n) => n,
                Source::FusedQkv { name, index } => format!("{name}#{index}"),
            })
            .collect();
        assert!(names.contains(&"encoder.layer.3.attention.attention.key.weight".to_string()));
        assert!(names.contains(&"vit.encoder.layer.3.attention.attention.key.weight".to_string()));
        assert!(names.contains(&"blocks.3.attn.qkv.weight#1".to_string()));
        let ls: Vec<String> = candidate_sources("blocks.0.ls1")
            .into_iter()
            .filter_map(|s| if let Source::Named(n) = s { Some(n) } else { None })
            .collect();
        assert!(ls.contains(&"encoder.layer.0.layer_scale1.lambda1".to_string()));
    }

    #[test]
    fn pos_embed_adapts_cls_and_resolution() {
        let dev = Device::Cpu;
        // Checkpoint: CLS + 4x4 grid; backbone: no CLS, 2x2 grid.
        let ckpt = Tensor::arange(0f32, 17.0 * 3.0, &dev).unwrap().reshape((1, 17, 3)).unwrap();
        let out = adapt_pos_embed(&ckpt, 2, 2, false, &dev).unwrap();
        assert_eq!(out.dims(), &[1, 4, 3]);
        // Rectangular grid with an exact CLS + token count match (AudioMAE-style 4x2 + cls).
        let ckpt = Tensor::ones((1, 9, 3), candle_core::DType::F32, &dev).unwrap();
        let out = adapt_pos_embed(&ckpt, 4, 2, true, &dev).unwrap();
        assert_eq!(out.dims(), &[1, 9, 3]);
        assert!(adapt_pos_embed(&ckpt, 4, 3, true, &dev).is_err());
        // Checkpoint without CLS, backbone with CLS at the same grid: a zero row is prepended.
        let ckpt = Tensor::ones((1, 4, 3), candle_core::DType::F32, &dev).unwrap();
        let out = adapt_pos_embed(&ckpt, 2, 2, true, &dev).unwrap();
        assert_eq!(out.dims(), &[1, 5, 3]);
        let first: Vec<f32> = out.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(first.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn resample_is_identity_on_constant_grid() {
        let dev = Device::Cpu;
        let src = Tensor::full(2.5f32, (1, 9, 2), &dev).unwrap();
        let out = resample_pos_grid(&src, 3, 5, &dev).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v.len(), 50);
        assert!(v.iter().all(|x| (x - 2.5).abs() < 1e-4));
    }
}
