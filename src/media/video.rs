//! Video file decoding into frames.
//!
//! Two paths, chosen by container:
//! - **GIF and animated WebP** are decoded in pure Rust by the `image` crate.
//! - **MP4 / WebM / MOV / MKV** are decoded by an external `ffmpeg` binary when one is
//!   on `PATH`. There is no dependable pure-Rust H.264/VP9 decoder, so rather than
//!   shipping a half-working one the runtime says exactly what is missing.
//!
//! Frames are then uniformly sub-sampled to the clip length the model expects.

use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, ImageDecoder, RgbImage};

use crate::media::image::sniff_media_format;
use crate::types::JepaError;

/// Upper bound on frames kept in memory from one file (before sub-sampling).
pub const MAX_DECODED_FRAMES: usize = 512;

/// Upper bound on decoded pixels kept in memory from one animation (about 768 MB of
/// RGB): frames past this budget are dropped, sampling covers what was kept.
pub const MAX_DECODED_PIXELS: u64 = 256 * 1024 * 1024;

/// Whether `bytes` are a container this module can turn into frames.
pub fn is_video_format(format: &str) -> bool {
    matches!(format, "gif" | "webp" | "mp4" | "webm")
}

/// Decode a clip from memory. `max_frames` caps decoding (GIF/WebP) or the ffmpeg
/// output. `fps` is the sampling rate requested from ffmpeg (ignored for GIF/WebP,
/// whose own timing is kept).
pub fn decode_clip_bytes(bytes: &[u8], max_frames: usize, fps: f32) -> Result<Vec<RgbImage>, JepaError> {
    let format = sniff_media_format(bytes)?;
    match format {
        "gif" => {
            let mut dec = GifDecoder::new(Cursor::new(bytes)).map_err(|e| JepaError::ImageProcessing(e.to_string()))?;
            dec.set_limits(crate::media::image::decode_limits())
                .map_err(|e| JepaError::ImageProcessing(e.to_string()))?;
            decode_animation(dec, max_frames)
        }
        "webp" => {
            let mut dec =
                WebPDecoder::new(Cursor::new(bytes)).map_err(|e| JepaError::ImageProcessing(e.to_string()))?;
            dec.set_limits(crate::media::image::decode_limits())
                .map_err(|e| JepaError::ImageProcessing(e.to_string()))?;
            if dec.has_animation() {
                decode_animation(dec, max_frames)
            } else {
                Err(JepaError::InvalidPayload("WebP file is a still image, not an animation".into()))
            }
        }
        "mp4" | "webm" => {
            let tmp = std::env::temp_dir().join(format!("jepa_clip_{}.{}", uuid::Uuid::new_v4(), format));
            std::fs::write(&tmp, bytes)?;
            let result = decode_with_ffmpeg(&tmp, max_frames, fps);
            let _ = std::fs::remove_file(&tmp);
            result
        }
        other => Err(JepaError::InvalidPayload(format!("'{other}' is not a video format"))),
    }
}

/// Decode a clip from a file on disk.
pub fn decode_clip_path(path: &Path, max_frames: usize, fps: f32) -> Result<Vec<RgbImage>, JepaError> {
    let mut head = vec![0u8; 64];
    let n = {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        f.read(&mut head)?
    };
    head.truncate(n);
    let format = sniff_media_format(&head)?;
    match format {
        "gif" | "webp" => decode_clip_bytes(&std::fs::read(path)?, max_frames, fps),
        "mp4" | "webm" => decode_with_ffmpeg(path, max_frames, fps),
        other => Err(JepaError::InvalidPayload(format!("'{other}' is not a video format"))),
    }
}

fn decode_animation<'a, D: AnimationDecoder<'a>>(decoder: D, max_frames: usize) -> Result<Vec<RgbImage>, JepaError> {
    let mut out = Vec::new();
    let mut pixels: u64 = 0;
    for frame in decoder.into_frames() {
        let frame = frame.map_err(|e| JepaError::ImageProcessing(e.to_string()))?;
        let rgb = image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8();
        pixels += u64::from(rgb.width()) * u64::from(rgb.height());
        out.push(rgb);
        if out.len() >= max_frames.min(MAX_DECODED_FRAMES) || pixels >= MAX_DECODED_PIXELS {
            break;
        }
    }
    if out.is_empty() {
        return Err(JepaError::InvalidPayload("Animation contains no frames".into()));
    }
    Ok(out)
}

/// Locate an ffmpeg binary (`JEPA_FFMPEG` overrides `PATH`).
pub fn ffmpeg_binary() -> Option<String> {
    if let Ok(p) = std::env::var("JEPA_FFMPEG")
        && Path::new(&p).is_file()
    {
        return Some(p);
    }
    let ok = Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false);
    ok.then(|| "ffmpeg".to_string())
}

fn decode_with_ffmpeg(path: &Path, max_frames: usize, fps: f32) -> Result<Vec<RgbImage>, JepaError> {
    let Some(bin) = ffmpeg_binary() else {
        return Err(JepaError::InvalidPayload(
            "Decoding MP4/WebM needs the `ffmpeg` binary on PATH (or JEPA_FFMPEG=/path/to/ffmpeg). \
             GIF and animated WebP are decoded natively."
                .into(),
        ));
    };
    // Decode at a fixed small size: the model input is a centre crop anyway and this
    // keeps memory bounded regardless of the source resolution.
    const SIDE: u32 = 320;
    let max_frames = max_frames.clamp(1, MAX_DECODED_FRAMES);
    let output = Command::new(&bin)
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("fps={fps},scale={SIDE}:{SIDE}:force_original_aspect_ratio=increase,crop={SIDE}:{SIDE}"),
            "-frames:v",
            &max_frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .map_err(|e| JepaError::ImageProcessing(format!("Could not run ffmpeg: {e}")))?;
    if !output.status.success() {
        return Err(JepaError::ImageProcessing(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let frame_len = (SIDE * SIDE * 3) as usize;
    let frames: Vec<RgbImage> =
        output.stdout.chunks_exact(frame_len).filter_map(|c| RgbImage::from_raw(SIDE, SIDE, c.to_vec())).collect();
    if frames.is_empty() {
        return Err(JepaError::InvalidPayload("ffmpeg produced no frames".into()));
    }
    Ok(frames)
}

/// Pick `count` frames spread uniformly over the clip (repeating when the clip is shorter).
pub fn sample_uniform(frames: &[RgbImage], count: usize) -> Vec<RgbImage> {
    if frames.is_empty() || count == 0 {
        return Vec::new();
    }
    (0..count).map(|i| frames[(i * frames.len()) / count].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};

    pub(crate) fn demo_gif(frames: usize, side: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            for i in 0..frames {
                let img = RgbaImage::from_fn(side, side, |x, _| {
                    let on = ((x / 8) as usize + i).is_multiple_of(2);
                    if on { Rgba([255, 255, 255, 255]) } else { Rgba([0, 0, 0, 255]) }
                });
                enc.encode_frame(Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1))).unwrap();
            }
        }
        buf
    }

    #[test]
    fn gif_is_decoded_natively_and_sampled() {
        let gif = demo_gif(6, 32);
        assert_eq!(sniff_media_format(&gif).unwrap(), "gif");
        let frames = decode_clip_bytes(&gif, 100, 10.0).unwrap();
        assert_eq!(frames.len(), 6);
        assert_eq!((frames[0].width(), frames[0].height()), (32, 32));
        assert_eq!(sample_uniform(&frames, 4).len(), 4);
        assert_eq!(sample_uniform(&frames, 16).len(), 16);
        assert_eq!(decode_clip_bytes(&gif, 2, 10.0).unwrap().len(), 2);
    }

    #[test]
    fn oversized_images_are_refused_before_allocation() {
        // A PNG header claiming 100000 x 100000 pixels: refused by the limits, not by OOM.
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, b'I', b'H', b'D', b'R'];
        png.extend_from_slice(&100_000u32.to_be_bytes());
        png.extend_from_slice(&100_000u32.to_be_bytes());
        png.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
        let err = crate::media::image::preprocess_image_bytes(&png, &Default::default(), &candle_core::Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(!err.is_empty());
        let frames = decode_clip_bytes(&demo_gif(3, 16), 100, 10.0).unwrap();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn containers_need_ffmpeg_and_say_so() {
        let fake_mp4 = [b"\0\0\0\x18ftypisom".as_slice(), &[0u8; 32]].concat();
        if ffmpeg_binary().is_none() {
            let err = decode_clip_bytes(&fake_mp4, 8, 4.0).unwrap_err().to_string();
            assert!(err.contains("ffmpeg"), "{err}");
        }
        assert!(decode_clip_bytes(b"not a video at all, really", 8, 4.0).is_err());
    }
}
