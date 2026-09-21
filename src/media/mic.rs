//! Microphone capture: the audio counterpart of the camera supervisor.
//!
//! A worker thread owns the `cpal` input stream (streams are not `Send` on every
//! platform) and appends down-mixed mono samples into a shared ring of the last
//! [`BUFFER_SECONDS`] seconds. Consumers ask for the most recent `n` seconds as an
//! [`AudioClip`] at 16 kHz, ready for the audio model front-end.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};

use crate::media::audio::{AudioClip, resample};
use crate::types::JepaError;

/// Seconds of audio kept in memory (AudioMAE's window is 10.24 s).
pub const BUFFER_SECONDS: f32 = 12.0;

/// Sample rate delivered to consumers.
pub const TARGET_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MicSource {
    Off,
    Opening,
    Device,
}

/// Snapshot for `/api/status` and `/api/mic/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicHealth {
    pub active: bool,
    pub source: MicSource,
    /// Seconds currently buffered (up to [`BUFFER_SECONDS`]).
    pub buffered_seconds: f32,
    /// RMS level of the last 100 ms, 0 to 1.
    pub level: f32,
    pub device_name: Option<String>,
    pub sample_rate: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicDeviceInfo {
    pub index: usize,
    pub name: String,
    pub is_default: bool,
}

struct Ring {
    samples: VecDeque<f32>,
    rate: u32,
    capacity: usize,
}

impl Ring {
    fn push(&mut self, chunk: &[f32]) {
        for &s in chunk {
            if self.samples.len() >= self.capacity {
                self.samples.pop_front();
            }
            self.samples.push_back(s);
        }
    }
}

/// Microphone streaming supervisor.
pub struct MicSupervisor {
    is_running: Arc<AtomicBool>,
    ring: Arc<Mutex<Ring>>,
    chunks: Arc<AtomicU64>,
    source: Arc<Mutex<MicSource>>,
    device_name: Arc<Mutex<Option<String>>>,
    last_error: Arc<Mutex<Option<String>>>,
    /// Bumped once per captured chunk so listeners can wake up.
    chunk_tx: tokio::sync::broadcast::Sender<u64>,
}

impl Default for MicSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl MicSupervisor {
    pub fn new() -> Self {
        let (chunk_tx, _) = tokio::sync::broadcast::channel(32);
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            ring: Arc::new(Mutex::new(Ring {
                samples: VecDeque::new(),
                rate: TARGET_RATE,
                capacity: (BUFFER_SECONDS * TARGET_RATE as f32) as usize,
            })),
            chunks: Arc::new(AtomicU64::new(0)),
            source: Arc::new(Mutex::new(MicSource::Off)),
            device_name: Arc::new(Mutex::new(None)),
            last_error: Arc::new(Mutex::new(None)),
            chunk_tx,
        }
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<u64> {
        self.chunk_tx.subscribe()
    }

    pub fn is_active(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    pub fn health(&self) -> MicHealth {
        let ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let rate = ring.rate.max(1);
        let window = (rate as f32 * 0.1) as usize;
        let tail = ring.samples.len().saturating_sub(window);
        let level = if ring.samples.len() > tail {
            let sum: f32 = ring.samples.iter().skip(tail).map(|s| s * s).sum();
            (sum / (ring.samples.len() - tail) as f32).sqrt().min(1.0)
        } else {
            0.0
        };
        MicHealth {
            active: self.is_active(),
            source: *self.source.lock().unwrap_or_else(|e| e.into_inner()),
            buffered_seconds: ring.samples.len() as f32 / rate as f32,
            level,
            device_name: self.device_name.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            sample_rate: ring.rate,
            error: self.last_error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }

    /// The most recent `seconds` of audio, resampled to 16 kHz. `None` when nothing
    /// has been captured yet.
    pub fn latest_clip(&self, seconds: f32) -> Option<AudioClip> {
        let ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        if ring.samples.is_empty() {
            return None;
        }
        let want = ((seconds.max(0.1)) * ring.rate as f32) as usize;
        let start = ring.samples.len().saturating_sub(want);
        let samples: Vec<f32> = ring.samples.iter().skip(start).copied().collect();
        let clip = AudioClip { samples, sample_rate: ring.rate };
        Some(resample(&clip, TARGET_RATE))
    }

    /// Peak-normalised envelope of the last `seconds`, `points` values in `[0, 1]`
    /// (max absolute sample per bucket). Empty when nothing was captured.
    pub fn waveform(&self, seconds: f32, points: usize) -> Vec<f32> {
        let ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        if ring.samples.is_empty() || points == 0 {
            return Vec::new();
        }
        let want = ((seconds.max(0.1)) * ring.rate as f32) as usize;
        let start = ring.samples.len().saturating_sub(want);
        let tail: Vec<f32> = ring.samples.iter().skip(start).copied().collect();
        envelope(&tail, points)
    }

    /// Feed samples directly (tests and headless use).
    #[doc(hidden)]
    pub fn push_samples(&self, samples: &[f32], rate: u32) {
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        if ring.rate != rate {
            ring.samples.clear();
            ring.rate = rate;
            ring.capacity = (BUFFER_SECONDS * rate as f32) as usize;
        }
        ring.push(samples);
        drop(ring);
        let n = self.chunks.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.chunk_tx.send(n);
    }

    /// Pretend the device is open (tests feed samples with [`Self::push_samples`]).
    #[doc(hidden)]
    pub fn force_active_for_test(&self, active: bool) {
        self.is_running.store(active, Ordering::Relaxed);
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) =
            if active { MicSource::Device } else { MicSource::Off };
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = MicSource::Off;
        tracing::info!("Requested microphone stop.");
    }

    /// Open the input device (`None` for the system default) on a worker thread.
    pub fn start(&self, device_index: Option<usize>) -> Result<(), JepaError> {
        if self.is_active() {
            self.stop();
            std::thread::sleep(Duration::from_millis(100));
        }
        self.is_running.store(true, Ordering::Relaxed);
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = MicSource::Opening;
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = None;

        let running = self.is_running.clone();
        let ring = self.ring.clone();
        let chunks = self.chunks.clone();
        let source = self.source.clone();
        let device_name = self.device_name.clone();
        let last_error = self.last_error.clone();
        let chunk_tx = self.chunk_tx.clone();

        std::thread::spawn(move || {
            let fail = |msg: String| {
                tracing::warn!("{}", msg);
                *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(msg);
                *source.lock().unwrap_or_else(|e| e.into_inner()) = MicSource::Off;
                running.store(false, Ordering::Relaxed);
            };
            let host = cpal::default_host();
            let device = match device_index {
                Some(i) => host.input_devices().ok().and_then(|mut d| d.nth(i)),
                None => host.default_input_device(),
            };
            let Some(device) = device else {
                fail("No microphone found (on macOS: allow microphone access for the app or terminal that launched jepctl in System Settings > Privacy & Security > Microphone).".into());
                return;
            };
            let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "microphone".into());
            let config = match device.default_input_config() {
                Ok(c) => c,
                Err(e) => {
                    fail(format!("Microphone '{name}' has no usable input configuration: {e}"));
                    return;
                }
            };
            let channels = config.channels() as usize;
            let rate = config.sample_rate();
            {
                let mut r = ring.lock().unwrap_or_else(|e| e.into_inner());
                r.samples.clear();
                r.rate = rate;
                r.capacity = (BUFFER_SECONDS * rate as f32) as usize;
            }
            let ring_cb = ring.clone();
            let chunks_cb = chunks.clone();
            let tx_cb = chunk_tx.clone();
            let err_cb = last_error.clone();
            let stream = device.build_input_stream::<f32, _, _>(
                config.into(),
                move |data: &[f32], _| {
                    let mono: Vec<f32> = data
                        .chunks(channels.max(1))
                        .map(|frame| frame.iter().sum::<f32>() / frame.len().max(1) as f32)
                        .collect();
                    ring_cb.lock().unwrap_or_else(|e| e.into_inner()).push(&mono);
                    let n = chunks_cb.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = tx_cb.send(n);
                },
                move |e| {
                    *err_cb.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("Microphone stream error: {e}"));
                },
                None,
            );
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    fail(format!("Could not open microphone '{name}': {e}"));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                fail(format!("Could not start microphone '{name}': {e}"));
                return;
            }
            *device_name.lock().unwrap_or_else(|e| e.into_inner()) = Some(name.clone());
            *source.lock().unwrap_or_else(|e| e.into_inner()) = MicSource::Device;
            tracing::info!("Microphone '{}' capturing at {} Hz, {} channel(s).", name, rate, channels);
            while running.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
            }
            drop(stream);
            tracing::info!("Microphone worker stopped.");
        });
        Ok(())
    }
}

/// Max absolute value per bucket, scaled so the loudest bucket is 1 (all zeros when silent).
pub fn envelope(samples: &[f32], points: usize) -> Vec<f32> {
    if samples.is_empty() || points == 0 {
        return Vec::new();
    }
    let bucket = samples.len().div_ceil(points).max(1);
    let mut out: Vec<f32> = samples.chunks(bucket).map(|c| c.iter().fold(0.0f32, |m, s| m.max(s.abs()))).collect();
    let peak = out.iter().cloned().fold(0.0f32, f32::max);
    if peak > 1e-4 {
        for v in out.iter_mut() {
            *v /= peak;
        }
    }
    out
}

/// Input devices known to the audio host.
pub fn list_mic_devices() -> Vec<MicDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.description().ok()).map(|d| d.name().to_string());
    host.input_devices()
        .map(|devices| {
            devices
                .enumerate()
                .map(|(index, d)| {
                    let name =
                        d.description().map(|d| d.name().to_string()).unwrap_or_else(|_| format!("Input {index}"));
                    let is_default = default_name.as_deref() == Some(name.as_str());
                    MicDeviceInfo { index, name, is_default }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_the_tail_and_resamples() {
        let mic = MicSupervisor::new();
        assert!(mic.latest_clip(1.0).is_none());
        let tone: Vec<f32> = (0..48_000).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        mic.push_samples(&tone, 48_000);
        let clip = mic.latest_clip(0.5).unwrap();
        assert_eq!(clip.sample_rate, TARGET_RATE);
        assert!((clip.samples.len() as i64 - 8_000).abs() <= 2, "{}", clip.samples.len());
        let h = mic.health();
        assert!((h.buffered_seconds - 1.0).abs() < 0.01);
        let wf = mic.waveform(0.5, 50);
        assert_eq!(wf.len(), 50);
        assert!(wf.iter().cloned().fold(0.0f32, f32::max) > 0.99);
        assert!(h.level > 0.2 && h.level < 0.6, "{}", h.level);
        // Capacity bounds the ring.
        for _ in 0..20 {
            mic.push_samples(&tone, 48_000);
        }
        assert!(mic.health().buffered_seconds <= BUFFER_SECONDS + 0.01);
    }
}
