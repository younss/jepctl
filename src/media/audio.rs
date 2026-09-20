//! Audio decoding and the log-mel front-end used by AudioMAE / AST style encoders.
//!
//! - WAV (PCM 8/16/24/32-bit and 32-bit float) is parsed here without dependencies.
//! - MP3 / FLAC / OGG go through the external `ffmpeg` binary when available.
//! - The filterbank follows Kaldi `compute-fbank-feats` conventions as used by
//!   `torchaudio.compliance.kaldi.fbank` in the reference training code: 25 ms Povey
//!   window, 10 ms hop, DC removal, pre-emphasis 0.97, 512-point FFT, 128 mel bins
//!   between 20 Hz and Nyquist, natural log with a floor of `f32::EPSILON`, then
//!   dataset normalisation `(fbank - mean) / (2 * std)` after zero-padding to the
//!   model's frame count.
//!
//! Numerical parity with torchaudio has been checked structurally (same steps and
//! constants), not bit-for-bit: see docs/ARCHITECTURE.md.

use std::path::Path;
use std::process::Command;

use candle_core::{Device, Tensor};

use crate::types::{AudioSpec, JepaError};

/// Mono PCM samples in `[-1, 1]` at a known rate.
#[derive(Debug, Clone)]
pub struct AudioClip {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Whether the sniffed format is an audio container this module can decode.
pub fn is_audio_format(format: &str) -> bool {
    matches!(format, "wav" | "mp3" | "flac" | "ogg")
}

/// Sniff audio containers (complements `media::image::sniff_media_format`).
pub fn sniff_audio_format(buffer: &[u8]) -> Option<&'static str> {
    if buffer.len() >= 12 && buffer.starts_with(b"RIFF") && &buffer[8..12] == b"WAVE" {
        Some("wav")
    } else if buffer.starts_with(b"ID3") || (buffer.len() >= 2 && buffer[0] == 0xFF && (buffer[1] & 0xE0) == 0xE0) {
        Some("mp3")
    } else if buffer.starts_with(b"fLaC") {
        Some("flac")
    } else if buffer.starts_with(b"OggS") {
        Some("ogg")
    } else {
        None
    }
}

/// Decode an audio file held in memory.
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<AudioClip, JepaError> {
    match sniff_audio_format(bytes) {
        Some("wav") => decode_wav(bytes),
        Some(fmt) => {
            let tmp = std::env::temp_dir().join(format!("jepa_audio_{}.{}", uuid::Uuid::new_v4(), fmt));
            std::fs::write(&tmp, bytes)?;
            let r = decode_with_ffmpeg(&tmp);
            let _ = std::fs::remove_file(&tmp);
            r
        }
        None => Err(JepaError::InvalidPayload("Not a supported audio format (WAV, MP3, FLAC, OGG)".into())),
    }
}

/// Decode an audio file from disk.
pub fn decode_audio_path(path: &Path) -> Result<AudioClip, JepaError> {
    let bytes = std::fs::read(path)?;
    match sniff_audio_format(&bytes) {
        Some("wav") => decode_wav(&bytes),
        Some(_) => decode_with_ffmpeg(path),
        None => Err(JepaError::InvalidPayload("Not a supported audio format (WAV, MP3, FLAC, OGG)".into())),
    }
}

fn decode_with_ffmpeg(path: &Path) -> Result<AudioClip, JepaError> {
    let Some(bin) = crate::media::video::ffmpeg_binary() else {
        return Err(JepaError::InvalidPayload(
            "Decoding MP3/FLAC/OGG needs the `ffmpeg` binary on PATH (or JEPA_FFMPEG). WAV is decoded natively.".into(),
        ));
    };
    let output = Command::new(bin)
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
        .output()
        .map_err(|e| JepaError::ImageProcessing(format!("Could not run ffmpeg: {e}")))?;
    if !output.status.success() {
        return Err(JepaError::ImageProcessing(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let samples: Vec<f32> = output.stdout.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    if samples.is_empty() {
        return Err(JepaError::InvalidPayload("ffmpeg produced no audio samples".into()));
    }
    Ok(AudioClip { samples, sample_rate: 16_000 })
}

/// Minimal RIFF/WAVE reader (PCM integer, IEEE float, WAVE_FORMAT_EXTENSIBLE).
pub fn decode_wav(bytes: &[u8]) -> Result<AudioClip, JepaError> {
    let bad = |m: &str| JepaError::InvalidPayload(format!("Invalid WAV: {m}"));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("missing RIFF/WAVE header"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);

    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err(bad("fmt chunk too short"));
            }
            format = u16_at(body_start);
            channels = u16_at(body_start + 2);
            rate = u32_at(body_start + 4);
            bits = u16_at(body_start + 14);
            if format == 0xFFFE && body.len() >= 26 {
                // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the sub-format GUID.
                format = u16_at(body_start + 24);
            }
        } else if id == b"data" {
            data = Some(body);
        }
        pos = body_start + size + (size & 1);
    }
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    if channels == 0 || rate == 0 {
        return Err(bad("no fmt chunk"));
    }
    let ch = channels as usize;
    let frames: Vec<f32> = match (format, bits) {
        (1, 8) => data.iter().map(|b| (*b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => data.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c) as f32 / 32768.0).collect(),
        (1, 24) => data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0)
            .collect(),
        (1, 32) => data.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as f32 / 2_147_483_648.0).collect(),
        (3, 32) => data.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        (f, b) => return Err(bad(&format!("unsupported format {f} / {b} bits"))),
    };
    // Down-mix to mono.
    let samples: Vec<f32> = frames.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    if samples.is_empty() {
        return Err(bad("no samples"));
    }
    Ok(AudioClip { samples, sample_rate: rate })
}

/// Linear-interpolation resampling (adequate for speech/ambient features at 16 kHz).
pub fn resample(clip: &AudioClip, target_rate: u32) -> AudioClip {
    if clip.sample_rate == target_rate || clip.samples.len() < 2 {
        return AudioClip { samples: clip.samples.clone(), sample_rate: target_rate };
    }
    let ratio = clip.sample_rate as f64 / target_rate as f64;
    let out_len = ((clip.samples.len() as f64) / ratio).floor() as usize;
    let samples = (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos.floor() as usize;
            let frac = (pos - i0 as f64) as f32;
            let a = clip.samples[i0];
            let b = clip.samples[(i0 + 1).min(clip.samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect();
    AudioClip { samples, sample_rate: target_rate }
}

/// In-place iterative radix-2 FFT over interleaved complex `(re, im)` pairs.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

fn hz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

/// Kaldi-style log-mel filterbank `[frames, n_mels]` for 16 kHz mono audio.
pub fn log_mel_fbank(samples: &[f32], sample_rate: u32, n_mels: usize) -> Vec<Vec<f32>> {
    let frame_len = (sample_rate as f32 * 0.025).round() as usize; // 400 @ 16 kHz
    let hop = (sample_rate as f32 * 0.010).round() as usize; // 160 @ 16 kHz
    let n_fft = frame_len.next_power_of_two(); // 512
    if samples.len() < frame_len {
        return Vec::new();
    }
    let num_frames = 1 + (samples.len() - frame_len) / hop;

    // Povey window = Hamming^0.85 (Kaldi default)
    let window: Vec<f32> = (0..frame_len)
        .map(|n| (0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (frame_len as f32 - 1.0)).cos()).powf(0.85))
        .collect();

    // Mel filterbank over FFT bins 0..n_fft/2 (Kaldi: low 20 Hz, high = Nyquist)
    let nyquist = sample_rate as f32 / 2.0;
    let (mel_lo, mel_hi) = (hz_to_mel(20.0), hz_to_mel(nyquist));
    let mel_step = (mel_hi - mel_lo) / (n_mels as f32 + 1.0);
    let bin_hz = sample_rate as f32 / n_fft as f32;
    let half = n_fft / 2;
    let filters: Vec<Vec<(usize, f32)>> = (0..n_mels)
        .map(|m| {
            let left = mel_lo + m as f32 * mel_step;
            let center = left + mel_step;
            let right = center + mel_step;
            (0..=half)
                .filter_map(|k| {
                    let mel = hz_to_mel(k as f32 * bin_hz);
                    let w = if mel > left && mel < center {
                        (mel - left) / (center - left)
                    } else if mel >= center && mel < right {
                        (right - mel) / (right - center)
                    } else {
                        0.0
                    };
                    (w > 0.0).then_some((k, w))
                })
                .collect()
        })
        .collect();
    let _ = mel_to_hz; // kept for symmetry / debugging

    let mut re = vec![0f32; n_fft];
    let mut im = vec![0f32; n_fft];
    let mut out = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let frame = &samples[f * hop..f * hop + frame_len];
        let mean = frame.iter().sum::<f32>() / frame_len as f32;
        // DC removal, pre-emphasis, window
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for n in 0..frame_len {
            let x = frame[n] - mean;
            let prev = if n == 0 { x } else { frame[n - 1] - mean };
            re[n] = (x - 0.97 * prev) * window[n];
        }
        fft(&mut re, &mut im);
        let power: Vec<f32> = (0..=half).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
        let row: Vec<f32> = filters
            .iter()
            .map(|taps| taps.iter().map(|(k, w)| power[*k] * w).sum::<f32>().max(f32::EPSILON).ln())
            .collect();
        out.push(row);
    }
    out
}

/// Full front-end: resample, filterbank, zero-pad/crop to `spec.frames`, normalise,
/// and return `[1, 1, frames, n_mels]`.
pub fn clip_to_spectrogram_tensor(clip: &AudioClip, spec: &AudioSpec, device: &Device) -> Result<Tensor, JepaError> {
    let clip = resample(clip, spec.sample_rate);
    let fbank = log_mel_fbank(&clip.samples, clip.sample_rate, spec.n_mels);
    if fbank.is_empty() {
        return Err(JepaError::InvalidPayload("Audio is shorter than one 25 ms frame".into()));
    }
    let mut data = vec![0f32; spec.frames * spec.n_mels];
    for (t, row) in fbank.iter().take(spec.frames).enumerate() {
        data[t * spec.n_mels..(t + 1) * spec.n_mels].copy_from_slice(row);
    }
    for v in data.iter_mut() {
        *v = (*v - spec.mean) / (2.0 * spec.std);
    }
    Ok(Tensor::from_vec(data, (1, 1, spec.frames, spec.n_mels), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sine_wav(freq: f32, secs: f32, rate: u32, bits: u16) -> Vec<u8> {
        let n = (secs * rate as f32) as usize;
        let mut data = Vec::new();
        for i in 0..n {
            let v = (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.5;
            match bits {
                16 => data.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes()),
                32 => data.extend_from_slice(&v.to_le_bytes()),
                _ => unreachable!(),
            }
        }
        let fmt_tag: u16 = if bits == 32 { 3 } else { 1 };
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&fmt_tag.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * u32::from(bits) / 8).to_le_bytes());
        wav.extend_from_slice(&(bits / 8).to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }

    #[test]
    fn wav_pcm16_and_float_decode() {
        let clip = decode_wav(&sine_wav(440.0, 0.5, 16_000, 16)).unwrap();
        assert_eq!(clip.sample_rate, 16_000);
        assert_eq!(clip.samples.len(), 8000);
        assert!(clip.samples.iter().cloned().fold(0f32, f32::max) > 0.45);
        let clip = decode_wav(&sine_wav(440.0, 0.1, 44_100, 32)).unwrap();
        assert_eq!(clip.sample_rate, 44_100);
        assert_eq!(resample(&clip, 16_000).samples.len(), 1600);
        assert!(decode_wav(b"RIFF\0\0\0\0WAVEjunk").is_err());
        assert_eq!(sniff_audio_format(b"RIFF....WAVE"), Some("wav"));
        assert_eq!(sniff_audio_format(b"ID3\x03"), Some("mp3"));
        assert_eq!(sniff_audio_format(b"fLaC"), Some("flac"));
    }

    #[test]
    fn fft_matches_naive_dft() {
        let n = 8;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
        let (mut re, mut im) = (x.clone(), vec![0f32; n]);
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut sr, mut si) = (0f32, 0f32);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                sr += v * a.cos();
                si += v * a.sin();
            }
            assert!((re[k] - sr).abs() < 1e-4 && (im[k] - si).abs() < 1e-4);
        }
    }

    #[test]
    fn fbank_peaks_at_the_tone_and_tensor_has_model_shape() {
        let clip = decode_wav(&sine_wav(1000.0, 1.0, 16_000, 16)).unwrap();
        let fb = log_mel_fbank(&clip.samples, 16_000, 128);
        assert_eq!(fb.len(), 1 + (16_000 - 400) / 160);
        // The loudest mel bin of a 1 kHz tone sits in the lower-middle of the range.
        let row = &fb[10];
        let peak = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!(peak > 30 && peak < 70, "peak bin {peak}");

        let spec = AudioSpec::default();
        let t = clip_to_spectrogram_tensor(&clip, &spec, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 1, 1024, 128]);
        // Padding rows are (0 - mean) / (2 std), not raw zeros.
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let pad = v[1023 * 128];
        assert!((pad - (0.0 - spec.mean) / (2.0 * spec.std)).abs() < 1e-5);
    }
}
