//! Vision Transformer (ViT) backbone implemented in pure Candle.
//! Supports I-JEPA and V-JEPA spatio-temporal representations.

use candle_core::{D, Device, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};
use crate::types::JepaError;

/// 2D Patch embedding module using convolution
#[derive(Debug)]
pub struct PatchEmbed {
    proj: Conv2d,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_patches: usize,
}

impl PatchEmbed {
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let proj = conv2d(in_chans, embed_dim, patch_size, cfg, vb.pp("proj"))?;
        let grid_size = img_size / patch_size;
        let num_patches = grid_size * grid_size;

        Ok(Self {
            proj,
            patch_size,
            embed_dim,
            num_patches,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, H, W]
        let _b = x.dim(0)?;
        let feat = self.proj.forward(x)?; // [B, embed_dim, grid_h, grid_w]
        let flattened = feat.flatten(2, 3)?; // [B, embed_dim, num_patches]
        let tokens = flattened.transpose(1, 2)?; // [B, num_patches, embed_dim]
        Ok(tokens)
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
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B, heads, N, head_dim]
        let k = k
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

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
        let fc1 = linear(embed_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_nn::Activation::Gelu.forward(&self.fc1.forward(x)?)?;
        self.fc2.forward(&h)
    }
}

/// Vision Transformer Encoder Block with Pre-Norm
#[derive(Debug)]
pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_ratio: f64, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(embed_dim, 1e-6, vb.pp("norm1"))?;
        let attn = Attention::new(embed_dim, num_heads, vb.pp("attn"))?;
        let norm2 = layer_norm(embed_dim, 1e-6, vb.pp("norm2"))?;
        let hidden_dim = (embed_dim as f64 * mlp_ratio) as usize;
        let mlp = Mlp::new(embed_dim, hidden_dim, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = (residual + self.attn.forward(&self.norm1.forward(x)?)?)?;
        let residual = &x;
        let x = (residual + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

/// Full Vision Transformer Backbone
#[derive(Debug)]
pub struct VitBackbone {
    pub patch_embed: PatchEmbed,
    pub blocks: Vec<Block>,
    pub norm: LayerNorm,
    pub embed_dim: usize,
    pub pos_embed: Option<Tensor>,
}

impl VitBackbone {
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(img_size, patch_size, in_chans, embed_dim, vb.pp("patch_embed"))?;
        let mut blocks = Vec::with_capacity(depth);
        let blocks_vb = vb.pp("blocks");
        for i in 0..depth {
            blocks.push(Block::new(embed_dim, num_heads, mlp_ratio, blocks_vb.pp(i))?);
        }
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;

        // 2D Sine-Cosine Positional Embedding initialization
        let num_patches = patch_embed.num_patches;
        let pos_tensor = Self::generate_sincos_pos_embed(num_patches, embed_dim, vb.device())?;

        Ok(Self {
            patch_embed,
            blocks,
            norm,
            embed_dim,
            pos_embed: Some(pos_tensor),
        })
    }

    /// Forward pass returning patch token representations and pooled vector
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [B, 3, H, W]
        let mut tokens = self.patch_embed.forward(x)?; // [B, N, D]

        if let Some(ref pos) = self.pos_embed {
            tokens = tokens.broadcast_add(pos)?;
        }

        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }

        let normalized_tokens = self.norm.forward(&tokens)?; // [B, N, D]

        // Global Average Pooling across patch tokens
        let pooled = normalized_tokens.mean(1)?; // [B, D]

        Ok((normalized_tokens, pooled))
    }

    /// Generates canonical 2D Sine-Cosine Positional Embeddings
    fn generate_sincos_pos_embed(num_patches: usize, embed_dim: usize, device: &Device) -> Result<Tensor> {
        let grid_size = (num_patches as f64).sqrt() as usize;
        let mut pos_embed_data = Vec::with_capacity(num_patches * embed_dim);

        let half_dim = embed_dim / 2;
        let omega_len = half_dim / 2;

        for h in 0..grid_size {
            for w in 0..grid_size {
                let mut token_pos = Vec::with_capacity(embed_dim);
                for i in 0..omega_len {
                    let omega = 1.0 / (10000.0f64.powf((i as f64) / (omega_len as f64)));
                    token_pos.push((h as f64 * omega).sin() as f32);
                    token_pos.push((h as f64 * omega).cos() as f32);
                    token_pos.push((w as f64 * omega).sin() as f32);
                    token_pos.push((w as f64 * omega).cos() as f32);
                }
                while token_pos.len() < embed_dim {
                    token_pos.push(0.0f32);
                }
                pos_embed_data.extend(token_pos);
            }
        }

        Tensor::from_vec(pos_embed_data, (1, num_patches, embed_dim), device)
    }
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

/// Load and map safetensors weights across common Hugging Face and Meta ViT formats.
///
/// Supported layouts: this crate's native names (`blocks.N.attn.q_proj`, ...),
/// HF `ViTModel` / `IJepaModel` (`encoder.layer.N.attention.attention.query`, with or
/// without a `vit.` prefix). Checkpoints with a CLS token have their positional
/// embedding sliced to the patch tokens only.
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

    let mut loaded_tensors = 0;
    let mut missing = Vec::new();
    let vars = varmap.data().lock().unwrap();
    let expected = vars.len();

    for (target_name, var) in vars.iter() {
        let mut candidates = vec![target_name.clone()];

        if target_name == "patch_embed.proj.weight" {
            candidates.push("embeddings.patch_embeddings.projection.weight".to_string());
            candidates.push("vit.embeddings.patch_embeddings.projection.weight".to_string());
        } else if target_name == "patch_embed.proj.bias" {
            candidates.push("embeddings.patch_embeddings.projection.bias".to_string());
            candidates.push("vit.embeddings.patch_embeddings.projection.bias".to_string());
        } else if target_name == "norm.weight" {
            candidates.push("layernorm.weight".to_string());
            candidates.push("vit.layernorm.weight".to_string());
        } else if target_name == "norm.bias" {
            candidates.push("layernorm.bias".to_string());
            candidates.push("vit.layernorm.bias".to_string());
        } else if let Some(stripped) = target_name.strip_prefix("blocks.") {
            if let Some(dot_idx) = stripped.find('.') {
                let block_idx = &stripped[..dot_idx];
                let sub = &stripped[dot_idx + 1..];

                let hf_sub = match sub {
                    "norm1.weight" => Some("layernorm_before.weight"),
                    "norm1.bias" => Some("layernorm_before.bias"),
                    "attn.q_proj.weight" => Some("attention.attention.query.weight"),
                    "attn.q_proj.bias" => Some("attention.attention.query.bias"),
                    "attn.k_proj.weight" => Some("attention.attention.key.weight"),
                    "attn.k_proj.bias" => Some("attention.attention.key.bias"),
                    "attn.v_proj.weight" => Some("attention.attention.value.weight"),
                    "attn.v_proj.bias" => Some("attention.attention.value.bias"),
                    "attn.out_proj.weight" => Some("attention.output.dense.weight"),
                    "attn.out_proj.bias" => Some("attention.output.dense.bias"),
                    "norm2.weight" => Some("layernorm_after.weight"),
                    "norm2.bias" => Some("layernorm_after.bias"),
                    "mlp.fc1.weight" => Some("intermediate.dense.weight"),
                    "mlp.fc1.bias" => Some("intermediate.dense.bias"),
                    "mlp.fc2.weight" => Some("output.dense.weight"),
                    "mlp.fc2.bias" => Some("output.dense.bias"),
                    _ => None,
                };

                if let Some(hs) = hf_sub {
                    candidates.push(format!("encoder.layer.{}.{}", block_idx, hs));
                    candidates.push(format!("vit.encoder.layer.{}.{}", block_idx, hs));
                }
            }
        }

        let mut found = false;
        for c in candidates {
            if let Ok(tensor) = mmap.load(&c, device) {
                if var.shape() == tensor.shape() {
                    var.set(&tensor)?;
                    loaded_tensors += 1;
                    found = true;
                    break;
                } else {
                    tracing::warn!(
                        "Shape mismatch for '{}' -> '{}': backbone {:?}, checkpoint {:?}",
                        target_name, c, var.shape(), tensor.shape()
                    );
                }
            }
        }
        if !found {
            missing.push(target_name.clone());
        }
    }

    // Positional embeddings: HF checkpoints may carry a leading CLS position.
    let mut pos_embed_loaded = false;
    if let Ok(pos_tensor) = mmap
        .load("embeddings.position_embeddings", device)
        .or_else(|_| mmap.load("vit.embeddings.position_embeddings", device))
        .or_else(|_| mmap.load("pos_embed", device))
    {
        if let Some(ref mut current_pos) = backbone.pos_embed {
            let (_, want_n, want_d) = current_pos.dims3()?;
            match pos_tensor.dims3() {
                Ok((1, n, d)) if n == want_n && d == want_d => {
                    *current_pos = pos_tensor;
                    pos_embed_loaded = true;
                }
                Ok((1, n, d)) if n == want_n + 1 && d == want_d => {
                    *current_pos = pos_tensor.narrow(1, 1, want_n)?.contiguous()?;
                    pos_embed_loaded = true;
                }
                Ok(dims) => {
                    tracing::warn!("Positional embedding shape {:?} incompatible with backbone {:?}", dims, (1, want_n, want_d));
                }
                Err(e) => tracing::warn!("Unreadable positional embedding: {}", e),
            }
        }
    }

    Ok(LoadOutcome {
        loaded: loaded_tensors,
        expected,
        missing,
        pos_embed_loaded,
    })
}
