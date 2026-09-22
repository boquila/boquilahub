// We cherck that every format in 'api/formats.rs' can be loaded

use boquilahub::api::abstractions::{AIOutputs, PredVideo};
use boquilahub::api::audio::AudioData;
use boquilahub::api::formats::{AUDIO_FORMATS, IMAGE_FORMATS, VIDEO_FORMATS};
use boquilahub::api::video_file::{playback_stream, VideofileProcessor};

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
    for ext in VIDEO_FORMATS {
        let path = format!("tests/assets/formats/video/video.{ext}");
        assert!(
            std::path::Path::new(&path).exists(),
            "missing asset for video format '{ext}' - generate tests/assets/formats or drop the extension"
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

#[test]
fn video_formats_decode() {
    for ext in VIDEO_FORMATS {
        let path = format!("tests/assets/formats/video/video.{ext}");
        let probe = VideofileProcessor::probe(&path).unwrap_or_else(|e| panic!("video.{ext}: {e}"));
        assert!(
            probe.width > 0 && probe.height > 0,
            "video.{ext}: zero dimensions"
        );
        assert!(
            !probe.first_frame.as_raw().is_empty(),
            "video.{ext}: empty first frame"
        );
    }
}

#[test]
fn video_playback_stream_seeks_and_scales() {
    let path = "tests/assets/formats/video/video.mp4";
    let probe = VideofileProcessor::probe(path).expect("probe playback fixture");
    let target = (probe.fps * 10.0) as u64;
    let receiver = playback_stream(path.into(), target, probe.fps, 320, 180);

    let first = receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("decode a frame after seeking");
    assert!(first.index >= target);
    assert!(first.img.width() <= 320);
    assert!(first.img.height() <= 180);

    let second = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("decode the following frame");
    assert!(second.index > first.index);
}

#[test]
fn video_probe_estimates_missing_frame_count_from_duration() {
    let probe = VideofileProcessor::probe("tests/assets/formats/video/video.webm")
        .expect("probe WebM fixture");
    assert!(probe.n_frames > 0);
}

#[test]
fn video_metadata_does_not_allocate_per_frame_predictions() {
    let mut video = PredVideo::new_simple("no-sidecar-test.mp4".into());
    video.hydrate(3840, 2160, 60.0, 10_000_000);
    assert!(video.frames.is_empty());

    video.record(90, AIOutputs::ObjectDetection(Vec::new()));
    assert_eq!(video.frames.len(), 91);
    assert_eq!(video.processed_count(), 1);

    video.record(90, AIOutputs::ObjectDetection(Vec::new()));
    assert_eq!(video.processed_count(), 1);
    video.reset();
    assert!(video.frames.is_empty());
    assert_eq!(video.processed_count(), 0);
}
