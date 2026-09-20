//! Image preprocessing: magic byte validation, bicubic resizing, and ImageNet normalization.

use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, ImageReader};
use std::io::Cursor;

use crate::types::{JepaError, Preprocessing};

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

    Err(JepaError::InvalidPayload("Unsupported media format. Expected PNG, JPEG, WebP, MP4, or WebM".into()))
}

/// Decode raw image bytes, centre-crop, resize with bicubic interpolation and
/// normalise into a Candle tensor of shape `[1, 3, size, size]`.
pub fn preprocess_image_bytes(buffer: &[u8], prep: &Preprocessing, device: &Device) -> Result<Tensor, JepaError> {
    // 1. Verify format via magic bytes
    let _ = sniff_media_format(buffer)?;

    // 2. Decode image using image crate
    let reader = ImageReader::new(Cursor::new(buffer))
        .with_guessed_format()
        .map_err(|e| JepaError::ImageProcessing(e.to_string()))?;

    let img = reader.decode().map_err(|e| JepaError::ImageProcessing(e.to_string()))?;

    // 3. Resize using CatmullRom (bicubic filter)
    preprocess_dynamic_image(&img, prep, device)
}

/// Preprocess a decoded image into a normalised Candle tensor `[1, 3, size, size]`.
pub fn preprocess_dynamic_image(
    img: &DynamicImage,
    prep: &Preprocessing,
    device: &Device,
) -> Result<Tensor, JepaError> {
    let (target_w, target_h) = (prep.size, prep.size);
    let mean = prep.normalization.mean();
    let std = prep.normalization.std();
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
        let r = ((pixel[0] as f32 / 255.0) - mean[0]) / std[0];
        let g = ((pixel[1] as f32 / 255.0) - mean[1]) / std[1];
        let b = ((pixel[2] as f32 / 255.0) - mean[2]) / std[2];

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Normalization;

    #[test]
    fn sniff_detects_formats() {
        assert_eq!(sniff_media_format(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0]).unwrap(), "png");
        assert_eq!(sniff_media_format(&[0xFF, 0xD8, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(), "jpeg");
        assert_eq!(sniff_media_format(b"RIFF\0\0\0\0WEBPVP8 ").unwrap(), "webp");
        assert!(sniff_media_format(b"hello world!").is_err());
        assert!(sniff_media_format(&[1, 2]).is_err());
    }

    #[test]
    fn normalization_follows_preprocessing() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(10, 20, image::Rgb([128, 128, 128])));
        let inc = Preprocessing { size: 4, normalization: Normalization::Inception };
        let t = preprocess_dynamic_image(&img, &inc, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 4, 4]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // (128/255 - 0.5) / 0.5 ≈ 0.0039
        assert!(v.iter().all(|x| (x - 0.00392).abs() < 1e-3));

        let imnet = Preprocessing { size: 4, normalization: Normalization::ImageNet };
        let t = preprocess_dynamic_image(&img, &imnet, &Device::Cpu).unwrap();
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - (128.0 / 255.0 - 0.485) / 0.229).abs() < 1e-4);
    }
}
