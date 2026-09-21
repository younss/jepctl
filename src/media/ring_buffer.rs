//! 16-frame circular sliding window buffer for V-JEPA spatio-temporal inference.

use base64::prelude::*;
use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageFormat, RgbImage, imageops::FilterType};
use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::RING_BUFFER_CAPACITY;
use crate::media::image::preprocess_dynamic_image;
use crate::types::{JepaError, Preprocessing, Roi};

/// Side length of the "model view": the centre-cropped square every frame is
/// reduced to before it is embedded.
pub const MODEL_VIEW_SIZE: u32 = 224;

/// Individual captured frame container
#[derive(Clone)]
pub struct FrameEntry {
    pub rgb_image: RgbImage,
    pub thumbnail_base64: String,
    /// JPEG of exactly what the model receives (centre crop, 224x224) before
    /// normalisation. Served to the UI so users can see what is being embedded.
    pub model_view_jpeg: Arc<Vec<u8>>,
    pub timestamp_ms: u64,
    /// Monotonic sequence number assigned at push time.
    pub sequence: u64,
}

/// Crop a frame to a normalised region of interest (pixel-clamped, never empty).
pub fn crop_roi(image: &RgbImage, roi: &Roi) -> RgbImage {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let x0 = (roi.x * w).round().clamp(0.0, w - 1.0) as u32;
    let y0 = (roi.y * h).round().clamp(0.0, h - 1.0) as u32;
    let cw = ((roi.w * w).round() as u32).clamp(1, image.width() - x0);
    let ch = ((roi.h * h).round() as u32).clamp(1, image.height() - y0);
    DynamicImage::ImageRgb8(image.clone()).crop_imm(x0, y0, cw, ch).to_rgb8()
}

/// What the model sees: optional ROI crop, then centre square crop, then resize.
pub fn model_input_image(image: &RgbImage, roi: Option<&Roi>) -> RgbImage {
    match roi {
        Some(r) => crop_roi(image, r),
        None => image.clone(),
    }
}

/// Centre-crop a frame to a square and resize it to the model's input size.
pub fn to_model_view(image: &RgbImage, size: u32) -> RgbImage {
    let (w, h) = (image.width(), image.height());
    let min_dim = w.min(h).max(1);
    let sx = (w - min_dim) / 2;
    let sy = (h - min_dim) / 2;
    DynamicImage::ImageRgb8(image.clone())
        .crop_imm(sx, sy, min_dim, min_dim)
        .resize_exact(size, size, FilterType::CatmullRom)
        .to_rgb8()
}

/// Circular sliding window ring buffer
pub struct RingBuffer {
    capacity: usize,
    frames: VecDeque<FrameEntry>,
    next_sequence: u64,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: if capacity == 0 { RING_BUFFER_CAPACITY } else { capacity },
            frames: VecDeque::with_capacity(capacity),
            next_sequence: 0,
        }
    }

    /// Insert a new frame into circular buffer, automatically popping the oldest
    pub fn push_frame(&mut self, image: RgbImage, timestamp_ms: u64) {
        // Generate small thumbnail for UI scrubber
        let dyn_img = DynamicImage::ImageRgb8(image.clone());
        let thumb = dyn_img.resize_exact(96, 54, FilterType::Nearest);
        let mut thumb_bytes = Cursor::new(Vec::new());
        let _ = thumb.write_to(&mut thumb_bytes, ImageFormat::Jpeg);
        let thumb_base64 = format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(thumb_bytes.into_inner()));

        let model_view = to_model_view(&image, MODEL_VIEW_SIZE);
        let mut view_bytes = Cursor::new(Vec::new());
        let _ = DynamicImage::ImageRgb8(model_view).write_to(&mut view_bytes, ImageFormat::Jpeg);

        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }

        self.next_sequence += 1;
        self.frames.push_back(FrameEntry {
            rgb_image: image,
            thumbnail_base64: thumb_base64,
            model_view_jpeg: Arc::new(view_bytes.into_inner()),
            timestamp_ms,
            sequence: self.next_sequence,
        });
    }

    /// Most recently pushed frame.
    pub fn latest(&self) -> Option<&FrameEntry> {
        self.frames.back()
    }

    /// Preprocess only the latest frame into an image tensor [1, 3, H, W].
    pub fn latest_image_tensor(
        &self,
        prep: &Preprocessing,
        roi: Option<&Roi>,
        device: &Device,
    ) -> Result<Tensor, JepaError> {
        let entry = self.latest().ok_or_else(|| JepaError::InvalidPayload("Ring buffer is empty".into()))?;
        preprocess_dynamic_image(&DynamicImage::ImageRgb8(model_input_image(&entry.rgb_image, roi)), prep, device)
    }

    /// JPEG of exactly what the model receives from the latest frame (ROI applied,
    /// centre crop, `size`×`size`), plus its sequence number.
    pub fn latest_model_view_jpeg(&self, size: u32, roi: Option<&Roi>) -> Option<(Vec<u8>, u64)> {
        let entry = self.latest()?;
        if roi.is_none() {
            return Some((entry.model_view_jpeg.as_ref().clone(), entry.sequence));
        }
        let view = to_model_view(&model_input_image(&entry.rgb_image, roi), size);
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(view).write_to(&mut bytes, ImageFormat::Jpeg).ok()?;
        Some((bytes.into_inner(), entry.sequence))
    }

    /// Small JPEG of the full latest frame (for the ROI editor).
    pub fn latest_full_frame_jpeg(&self, max_side: u32) -> Option<(Vec<u8>, u64, u32, u32)> {
        let entry = self.latest()?;
        let img = DynamicImage::ImageRgb8(entry.rgb_image.clone());
        let (w, h) = (img.width(), img.height());
        let scaled = if w.max(h) > max_side { img.resize(max_side, max_side, FilterType::Triangle) } else { img };
        let mut bytes = Cursor::new(Vec::new());
        scaled.write_to(&mut bytes, ImageFormat::Jpeg).ok()?;
        Some((bytes.into_inner(), entry.sequence, w, h))
    }

    /// Retrieve the number of frames currently in the buffer
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Retrieve list of thumbnail data URIs for UI visual scrubber
    pub fn get_thumbnails(&self) -> Vec<String> {
        self.frames.iter().map(|f| f.thumbnail_base64.clone()).collect()
    }

    /// Construct 5D spatio-temporal video tensor [1, 3, T, H, W] for V-JEPA
    pub fn to_video_tensor(
        &self,
        prep: &Preprocessing,
        roi: Option<&Roi>,
        device: &Device,
    ) -> Result<Tensor, JepaError> {
        if self.frames.is_empty() {
            return Err(JepaError::InvalidPayload("Ring buffer is empty".into()));
        }

        let mut frame_tensors = Vec::with_capacity(self.capacity);

        // Collect existing frames
        for entry in &self.frames {
            let dyn_img = DynamicImage::ImageRgb8(model_input_image(&entry.rgb_image, roi));
            let tensor = preprocess_dynamic_image(&dyn_img, prep, device)?; // [1, 3, H, W]
            frame_tensors.push(tensor);
        }

        // Pad with latest frame if buffer is not yet full
        while frame_tensors.len() < self.capacity {
            let Some(last) = frame_tensors.last().cloned() else { break };
            frame_tensors.push(last);
        }

        // Stack across temporal dimension T: list of [1, 3, H, W] -> [1, 3, T, H, W]
        // First stack frames to [T, 1, 3, H, W]
        let stacked = Tensor::stack(&frame_tensors, 0)?; // [T, 1, 3, H, W]
        let squeezed = stacked.squeeze(1)?; // [T, 3, H, W]
        let permuted = squeezed.transpose(0, 1)?; // [3, T, H, W]
        let video_5d = permuted.unsqueeze(0)?; // [1, 3, T, H, W]

        Ok(video_5d)
    }

    /// Clear all frames
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

pub type SharedRingBuffer = Arc<RwLock<RingBuffer>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_view_is_square_centre_crop() {
        let img = RgbImage::from_fn(640, 360, |x, _| {
            if !(140..500).contains(&x) { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 255, 0]) }
        });
        let view = to_model_view(&img, 224);
        assert_eq!((view.width(), view.height()), (224, 224));
        // Red side bands are cropped away: every pixel is green.
        assert!(view.pixels().all(|p| p[1] > 200 && p[0] < 50));
    }

    #[test]
    fn roi_crop_selects_the_region() {
        // Left half red, right half green.
        let img =
            RgbImage::from_fn(200, 100, |x, _| if x < 100 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 255, 0]) });
        let right = crop_roi(&img, &Roi { x: 0.5, y: 0.0, w: 0.5, h: 1.0 });
        assert_eq!((right.width(), right.height()), (100, 100));
        assert!(right.pixels().all(|p| p[1] == 255));
        // Out-of-range boxes are clamped, never empty.
        let tiny = crop_roi(&img, &Roi { x: 0.99, y: 0.99, w: 0.5, h: 0.5 });
        assert!(tiny.width() >= 1 && tiny.height() >= 1);

        let mut rb = RingBuffer::new(2);
        rb.push_frame(img, 0);
        let (jpeg, seq) = rb.latest_model_view_jpeg(32, Some(&Roi { x: 0.5, y: 0.0, w: 0.5, h: 1.0 })).unwrap();
        assert_eq!(seq, 1);
        let decoded = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!((decoded.width(), decoded.height()), (32, 32));
        assert!(decoded.pixels().all(|p| p[1] > 200 && p[0] < 60));
        assert!(rb.latest_full_frame_jpeg(64).is_some());
    }

    #[test]
    fn ring_buffer_keeps_capacity_and_sequence() {
        let mut rb = RingBuffer::new(3);
        for i in 0..5 {
            rb.push_frame(RgbImage::new(8, 8), i);
        }
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.latest().unwrap().sequence, 5);
        assert_eq!(rb.latest().unwrap().timestamp_ms, 4);
        assert!(!rb.latest().unwrap().model_view_jpeg.is_empty());

        let prep = Preprocessing { size: 16, ..Default::default() };
        let t = rb.to_video_tensor(&prep, None, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 3, 16, 16]);
        let img = rb.latest_image_tensor(&prep, None, &Device::Cpu).unwrap();
        assert_eq!(img.dims(), &[1, 3, 16, 16]);
    }
}
