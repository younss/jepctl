//! Cross-platform camera capture runner with nokhwa and synthetic fallback.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use image::{Rgb, RgbImage};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use tokio::sync::broadcast;

use crate::media::ring_buffer::SharedRingBuffer;
use crate::types::{CameraDeviceInfo, JepaError};

/// Query available system camera devices
pub fn list_camera_devices() -> Vec<CameraDeviceInfo> {
    let mut devices = Vec::new();

    // Query physical camera devices via nokhwa
    match nokhwa::query(ApiBackend::Auto) {
        Ok(cams) => {
            for cam in cams {
                let idx = match cam.index() {
                    CameraIndex::Index(i) => *i as usize,
                    CameraIndex::String(s) => s.parse().unwrap_or(0),
                };
                devices.push(CameraDeviceInfo {
                    index: idx,
                    name: cam.human_name(),
                });
            }
        }
        Err(e) => {
            tracing::warn!("Failed to query camera devices via nokhwa: {}. Providing synthetic test camera.", e);
        }
    }

    // Always include a synthetic test bench camera for headless/CI/testing environments
    if devices.is_empty() {
        devices.push(CameraDeviceInfo {
            index: 0,
            name: "Virtual Synthetic JEPA Test Camera".to_string(),
        });
    }

    devices
}

/// Camera streaming supervisor
pub struct CameraSupervisor {
    is_running: Arc<AtomicBool>,
    ring_buffer: SharedRingBuffer,
    frame_tx: broadcast::Sender<u64>,
}

impl CameraSupervisor {
    pub fn new(ring_buffer: SharedRingBuffer) -> Self {
        let (frame_tx, _) = broadcast::channel(32);
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            ring_buffer,
            frame_tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<u64> {
        self.frame_tx.subscribe()
    }

    pub fn is_active(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        tracing::info!("Requested camera stream stop.");
    }

    /// Start capturing frames at target fps and pushing into ring buffer
    pub fn start(&self, camera_index: usize, target_fps: u64) -> Result<(), JepaError> {
        if self.is_running.load(Ordering::Relaxed) {
            self.stop();
            std::thread::sleep(Duration::from_millis(150));
        }

        self.is_running.store(true, Ordering::Relaxed);
        let running_flag = self.is_running.clone();
        let ring_buffer = self.ring_buffer.clone();
        let frame_tx = self.frame_tx.clone();
        let fps = if target_fps == 0 { 10 } else { target_fps };
        let frame_interval = Duration::from_millis(1000 / fps);

        std::thread::spawn(move || {
            tracing::info!("Starting camera capture worker thread (device: {}, target fps: {})...", camera_index, fps);

            let index = CameraIndex::Index(camera_index as u32);
            let mut cam_result = Camera::new(
                index.clone(),
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
            );
            if cam_result.is_err() {
                cam_result = Camera::new(
                    index,
                    RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
                );
            }
            let mut use_synthetic = false;

            if let Ok(ref mut cam) = cam_result {
                if let Err(e) = cam.open_stream() {
                    tracing::warn!("Could not open camera stream: {}. Switching to synthetic pattern.", e);
                    use_synthetic = true;
                }
            } else {
                tracing::warn!("Could not initialize camera: {}. Switching to synthetic pattern.", cam_result.as_ref().err().unwrap());
                use_synthetic = true;
            }

            let mut frame_count: u64 = 0;
            let start_time = Instant::now();

            while running_flag.load(Ordering::Relaxed) {
                let loop_start = Instant::now();
                let now_ms = start_time.elapsed().as_millis() as u64;

                let frame_image: Option<RgbImage> = if !use_synthetic {
                    if let Ok(ref mut cam) = cam_result {
                        match cam.frame() {
                            Ok(frame) => match frame.decode_image::<RgbFormat>() {
                                Ok(img) => Some(img),
                                Err(e) => {
                                    tracing::warn!("Frame decode error: {}", e);
                                    None
                                }
                            },
                            Err(e) => {
                                tracing::warn!("Frame capture error: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    Some(generate_synthetic_frame(frame_count))
                };

                if let Some(rgb) = frame_image {
                    frame_count += 1;
                    {
                        let mut lock = ring_buffer.blocking_write();
                        lock.push_frame(rgb, now_ms);
                    }
                    let _ = frame_tx.send(frame_count);
                }

                let elapsed = loop_start.elapsed();
                if elapsed < frame_interval {
                    std::thread::sleep(frame_interval - elapsed);
                }
            }

            if !use_synthetic {
                if let Ok(ref mut cam) = cam_result {
                    let _ = cam.stop_stream();
                }
            }

            tracing::info!("Camera capture worker thread stopped.");
        });

        Ok(())
    }
}

/// Generate dynamic synthetic RGB test pattern (224x224) for headless testing
fn generate_synthetic_frame(frame_num: u64) -> RgbImage {
    let width = 224;
    let height = 224;
    let mut img = RgbImage::new(width, height);

    let t = (frame_num as f32) * 0.1;
    let center_x = (width as f32 / 2.0) + (t.cos() * 40.0);
    let center_y = (height as f32 / 2.0) + (t.sin() * 40.0);

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            let r = ((dist * 2.0).sin() * 127.0 + 128.0) as u8;
            let g = (((x as f32 / 2.0) + t * 5.0).sin() * 127.0 + 128.0) as u8;
            let b = (((y as f32 / 2.0) - t * 5.0).cos() * 127.0 + 128.0) as u8;

            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    img
}
