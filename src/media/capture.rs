//! Cross-platform camera capture runner with nokhwa and synthetic fallback.

use image::{Rgb, RgbImage};
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
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
                devices.push(CameraDeviceInfo { index: idx, name: cam.human_name() });
            }
        }
        Err(e) => {
            tracing::warn!("Failed to query camera devices via nokhwa: {}. Providing synthetic test camera.", e);
        }
    }

    // Always include a synthetic test bench camera for headless/CI/testing environments
    if devices.is_empty() {
        devices.push(CameraDeviceInfo { index: 0, name: "Virtual synthetic test camera".to_string() });
    }

    devices
}

/// How long the device may take to open before falling back to the synthetic pattern.
/// On macOS a first launch waits for the camera permission prompt; a terminal that was
/// never granted access can block indefinitely.
pub const OPEN_TIMEOUT: Duration = Duration::from_secs(6);

/// Source of the frames currently produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraSource {
    Off,
    Opening,
    Device,
    /// The device could not be opened; frames are a generated test pattern.
    Synthetic,
}

/// Camera streaming supervisor
pub struct CameraSupervisor {
    is_running: Arc<AtomicBool>,
    ring_buffer: SharedRingBuffer,
    frame_tx: broadcast::Sender<u64>,
    frames: Arc<std::sync::atomic::AtomicU64>,
    source: Arc<std::sync::Mutex<CameraSource>>,
    last_error: Arc<std::sync::Mutex<Option<String>>>,
}

/// Snapshot of the capture state for `/api/status`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CameraHealth {
    pub active: bool,
    pub source: CameraSource,
    pub frames: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CameraSupervisor {
    pub fn new(ring_buffer: SharedRingBuffer) -> Self {
        let (frame_tx, _) = broadcast::channel(32);
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            ring_buffer,
            frame_tx,
            frames: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            source: Arc::new(std::sync::Mutex::new(CameraSource::Off)),
            last_error: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn health(&self) -> CameraHealth {
        CameraHealth {
            active: self.is_active(),
            source: *self.source.lock().unwrap_or_else(|e| e.into_inner()),
            frames: self.frames.load(Ordering::Relaxed),
            error: self.last_error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
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
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = CameraSource::Off;
        tracing::info!("Requested camera stream stop.");
    }

    /// Start capturing frames at target fps and pushing into ring buffer
    pub fn start(&self, camera_index: usize, target_fps: u64) -> Result<(), JepaError> {
        if self.is_running.load(Ordering::Relaxed) {
            self.stop();
            std::thread::sleep(Duration::from_millis(150));
        }

        self.is_running.store(true, Ordering::Relaxed);
        self.frames.store(0, Ordering::Relaxed);
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = CameraSource::Opening;
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let running_flag = self.is_running.clone();
        let ring_buffer = self.ring_buffer.clone();
        let frame_tx = self.frame_tx.clone();
        let frames = self.frames.clone();
        let source = self.source.clone();
        let last_error = self.last_error.clone();
        let fps = if target_fps == 0 { 10 } else { target_fps };
        let frame_interval = Duration::from_millis(1000 / fps);

        // Watchdog: if the device has not produced a frame within OPEN_TIMEOUT (a
        // permission prompt or a driver that never answers), a synthetic producer takes
        // over and the worker below drops the device when it finally opens. `nokhwa`
        // cameras are not `Send`, so the open call cannot be moved to a helper thread.
        {
            let running = running_flag.clone();
            let frames = frames.clone();
            let source = source.clone();
            let last_error = last_error.clone();
            let ring_buffer = ring_buffer.clone();
            let frame_tx = frame_tx.clone();
            std::thread::spawn(move || {
                std::thread::sleep(OPEN_TIMEOUT);
                if !running.load(Ordering::Relaxed) || frames.load(Ordering::Relaxed) > 0 {
                    return;
                }
                let still_opening = *source.lock().unwrap_or_else(|e| e.into_inner()) == CameraSource::Opening;
                if !still_opening {
                    return;
                }
                let msg = format!(
                    "Camera {} produced no frame within {} s (on macOS: allow camera access for the app or terminal that launched jepctl in System Settings > Privacy & Security > Camera). Using the synthetic test pattern.",
                    camera_index,
                    OPEN_TIMEOUT.as_secs()
                );
                tracing::warn!("{}", msg);
                *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(msg);
                *source.lock().unwrap_or_else(|e| e.into_inner()) = CameraSource::Synthetic;
                run_producer(None, running, ring_buffer, frame_tx, frames, frame_interval);
            });
        }

        std::thread::spawn(move || {
            tracing::info!("Starting camera capture worker thread (device: {}, target fps: {})...", camera_index, fps);
            let index = CameraIndex::Index(camera_index as u32);
            let mut cam_result = Camera::new(
                index.clone(),
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
            );
            if cam_result.is_err() {
                cam_result = Camera::new(index, RequestedFormat::new::<RgbFormat>(RequestedFormatType::None));
            }
            let opened = cam_result.and_then(|mut c| c.open_stream().map(|_| c));

            // The watchdog may already have taken over.
            let taken_over = *source.lock().unwrap_or_else(|e| e.into_inner()) == CameraSource::Synthetic;
            let camera = match opened {
                Ok(cam) if !taken_over => {
                    *source.lock().unwrap_or_else(|e| e.into_inner()) = CameraSource::Device;
                    Some(cam)
                }
                Ok(_) => {
                    tracing::warn!(
                        "Camera {} opened after the timeout; the synthetic producer keeps running.",
                        camera_index
                    );
                    return;
                }
                Err(e) => {
                    if taken_over {
                        return;
                    }
                    let detail = e.to_string();
                    let hint = if detail.contains("Lock Rejected")
                        || detail.contains("busy")
                        || detail.contains("in use")
                    {
                        " The device is held by another application or another jepctl instance: stop it and restart the camera."
                    } else {
                        ""
                    };
                    let msg = format!(
                        "Could not open camera {}: {}.{} Using the synthetic test pattern.",
                        camera_index, detail, hint
                    );
                    tracing::warn!("{}", msg);
                    *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(msg);
                    *source.lock().unwrap_or_else(|e| e.into_inner()) = CameraSource::Synthetic;
                    None
                }
            };
            run_producer(camera, running_flag, ring_buffer, frame_tx, frames, frame_interval);
        });

        Ok(())
    }
}

/// Frame loop shared by the device and the synthetic fallback.
fn run_producer(
    mut camera: Option<Camera>,
    running: Arc<AtomicBool>,
    ring_buffer: SharedRingBuffer,
    frame_tx: broadcast::Sender<u64>,
    frames: Arc<std::sync::atomic::AtomicU64>,
    frame_interval: Duration,
) {
    let mut frame_count: u64 = 0;
    let start_time = Instant::now();
    while running.load(Ordering::Relaxed) {
        let loop_start = Instant::now();
        let now_ms = start_time.elapsed().as_millis() as u64;
        let frame_image: Option<RgbImage> = match camera.as_mut() {
            Some(cam) => match cam.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
                Ok(img) => Some(img),
                Err(e) => {
                    tracing::warn!("Frame capture error: {}", e);
                    None
                }
            },
            None => Some(generate_synthetic_frame(frame_count)),
        };
        if let Some(rgb) = frame_image {
            frame_count += 1;
            frames.store(frame_count, Ordering::Relaxed);
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
    if let Some(mut cam) = camera {
        let _ = cam.stop_stream();
    }
    tracing::info!("Camera producer stopped.");
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
