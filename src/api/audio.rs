use serde::{Deserialize, Serialize};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::format::sample::Type as SampleType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn load_audio(path: impl AsRef<std::path::Path>) -> Result<AudioData, ffmpeg::Error> {
    ffmpeg::init()?;
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    let mut ictx = ffmpeg::format::input(&path.as_ref())?;

    let stream = ictx.streams().best(ffmpeg::media::Type::Audio)
        .ok_or(ffmpeg::Error::StreamNotFound)?;
    let stream_index = stream.index();

    let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
        .decoder().audio()?;

    let mut resampler = decoder.resampler(
        Sample::F32(SampleType::Packed),
        decoder.channel_layout(),
        decoder.rate(),
    )?;

    let mut samples: Vec<f32> = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index { continue; }
        decoder.send_packet(&packet)?;
        let mut decoded = ffmpeg::frame::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = ffmpeg::frame::Audio::empty();
            resampler.run(&decoded, &mut resampled)?;
            samples.extend(
                resampled.data(0).chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            );
        }
    }

    decoder.send_eof()?;
    let mut decoded = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let mut resampled = ffmpeg::frame::Audio::empty();
        resampler.run(&decoded, &mut resampled)?;
        samples.extend(
            resampled.data(0).chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        );
    }

    Ok(AudioData {
        samples,
        sample_rate: decoder.rate(),
        channels: decoder.channels(),
    })
}

impl AudioData {
    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        let samples_per_channel = self.samples.len() / self.channels.max(1) as usize;
        let seconds = samples_per_channel as f64 / self.sample_rate as f64;
        seconds
    }

    /// Returns (min_amplitude, max_amplitude, rms_amplitude).
    pub fn amplitude_stats(&self) -> (f32, f32, f32) {
        let min_amplitude = self.samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_amplitude = self.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_square = self.samples.iter().map(|s| s * s).sum::<f32>() / self.samples.len().max(1) as f32;
        let rms_amplitude = mean_square.sqrt();
        (min_amplitude, max_amplitude, rms_amplitude)
    }

    /// Average sample value (should be near 0.0 for normal audio).
    pub fn dc_offset(&self) -> f32 {
        if self.samples.is_empty() { return 0.0; }
        let sum = self.samples.iter().sum::<f32>();
        let average = sum / self.samples.len() as f32;
        average
    }

    /// First n samples.
    pub fn preview(&self, n: usize) -> &[f32] {
        let count = n.min(self.samples.len());
        &self.samples[..count]
    }
}

#[test]
fn smoke() {
    let a = load_audio("assets/test/audio.mp3").unwrap();
    let (min, max, rms) = a.amplitude_stats();

    println!("samples={} rate={} channels={} duration={:.3}s", a.samples.len(), a.sample_rate, a.channels, a.duration());
    println!("amp min={:.4} max={:.4} rms={:.4}", min, max, rms);
    println!("dc_offset={:.6}", a.dc_offset());
    println!("preview={:?}", a.preview(16));
}