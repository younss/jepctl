//! Image preprocessing: magic byte validation, bicubic resizing, and ImageNet normalization.

use std::io::Cursor;
use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, ImageReader};

use crate::types::JepaError;

/// ImageNet canonical mean constants for RGB channels
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet canonical standard deviation constants for RGB channels
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Sniff raw magic bytes to detect media format safely without trusting user-provided MIME
pub fn sniff_media_format(buffer: &[u8]) -> Result<&'static str, JepaError> {
    if buffer.len() < 12 {
        return Err(JepaError::InvalidPayload("Payload buffer too short to determine format".into()));
    }

    // PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    if buffer.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok("png");
    }

    // JPEG magic bytes: FF D8 FF
    if buffer.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok("jpeg");
    }

    // WebP magic bytes: RIFF....WEBP
    if buffer.starts_with(b"RIFF") && &buffer[8..12] == b"WEBP" {
        return Ok("webp");
    }

    // MP4/QuickTime magic bytes: ....ftyp
    if buffer.len() >= 8 && &buffer[4..8] == b"ftyp" {
        return Ok("mp4");
    }

    // Matroska/WebM magic bytes: 1A 45 DF A3
    if buffer.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Ok("webm");
    }

    Err(JepaError::InvalidPayload(
        "Unsupported media format. Expected PNG, JPEG, WebP, MP4, or WebM".into(),
    ))
}

/// Decode raw image bytes, resize to target dimensions with bicubic interpolation,
/// and apply ImageNet normalization to produce a Candle Tensor of shape [1, 3, target_h, target_w].
pub fn preprocess_image_bytes(
    buffer: &[u8],
    target_w: u32,
    target_h: u32,
    device: &Device,
) -> Result<Tensor, JepaError> {
    // 1. Verify format via magic bytes
    let _ = sniff_media_format(buffer)?;

    // 2. Decode image using image crate
    let reader = ImageReader::new(Cursor::new(buffer))
        .with_guessed_format()
        .map_err(|e| JepaError::ImageProcessing(e.to_string()))?;

    let img = reader.decode().map_err(|e| JepaError::ImageProcessing(e.to_string()))?;

    // 3. Resize using CatmullRom (bicubic filter)
    preprocess_dynamic_image(&img, target_w, target_h, device)
}

/// Preprocess DynamicImage into normalized Candle tensor [1, 3, H, W]
pub fn preprocess_dynamic_image(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    device: &Device,
) -> Result<Tensor, JepaError> {
    let (w, h) = (img.width(), img.height());
    let min_dim = w.min(h);
    let sx = (w - min_dim) / 2;
    let sy = (h - min_dim) / 2;
    let cropped = img.crop_imm(sx, sy, min_dim, min_dim);
    let resized = cropped.resize_exact(target_w, target_h, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let num_pixels = (target_w * target_h) as usize;
    let mut channel_r = Vec::with_capacity(num_pixels);
    let mut channel_g = Vec::with_capacity(num_pixels);
    let mut channel_b = Vec::with_capacity(num_pixels);

    for pixel in rgb.pixels() {
        // Normalize: (pixel / 255.0 - mean) / std
        let r = ((pixel[0] as f32 / 255.0) - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        let g = ((pixel[1] as f32 / 255.0) - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        let b = ((pixel[2] as f32 / 255.0) - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

        channel_r.push(r);
        channel_g.push(g);
        channel_b.push(b);
    }

    let mut planar = Vec::with_capacity(3 * num_pixels);
    planar.extend(channel_r);
    planar.extend(channel_g);
    planar.extend(channel_b);

    let tensor = Tensor::from_vec(planar, (1, 3, target_h as usize, target_w as usize), device)?;
    Ok(tensor)
}
