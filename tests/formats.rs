//! Verifies every extension advertised in `api/formats.rs` can actually be
//! decoded by the paths the app uses: `image::open` for images and
//! `AudioData::from_file` (ffmpeg) for audio.
//!
//! Assets live in `tests/assets/formats/{image,audio}` and are (re)generated
//! with `py tests/assets/formats/generate.py`.

use boquilahub::api::audio::AudioData;
use boquilahub::api::formats::{AUDIO_FORMATS, IMAGE_FORMATS};

#[test]
fn every_listed_format_has_an_asset() {
    for ext in IMAGE_FORMATS {
        let path = format!("tests/assets/formats/image/img.{ext}");
        assert!(
            std::path::Path::new(&path).exists(),
            "missing asset for image format '{ext}' - regenerate tests/assets/formats or drop the extension"
        );
    }
    for ext in AUDIO_FORMATS {
        let path = format!("tests/assets/formats/audio/audio.{ext}");
        assert!(
            std::path::Path::new(&path).exists(),
            "missing asset for audio format '{ext}' - regenerate tests/assets/formats or drop the extension"
        );
    }
}

#[test]
fn image_formats_decode() {
    for ext in IMAGE_FORMATS {
        let path = format!("tests/assets/formats/image/img.{ext}");
        let img = image::open(&path).unwrap_or_else(|e| panic!("img.{ext}: {e}"));
        let rgb: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = img.to_rgb8();
        assert!(
            rgb.width() > 0 && rgb.height() > 0,
            "img.{ext}: decoded to an empty buffer"
        );
    }
}

#[test]
fn audio_formats_decode() {
    for ext in AUDIO_FORMATS {
        let path = format!("tests/assets/formats/audio/audio.{ext}");
        let audio = AudioData::from_file(&path).unwrap_or_else(|e| panic!("audio.{ext}: {e}"));
        assert!(
            !audio.samples.is_empty(),
            "audio.{ext}: decoded to zero samples"
        );
        assert!(audio.duration() > 0.0, "audio.{ext}: zero duration");
    }
}
