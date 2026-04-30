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

impl AudioData {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ffmpeg::Error> {
        ffmpeg::init()?;
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
        let mut ictx = ffmpeg::format::input(&path.as_ref())?;
    
        let stream = ictx.streams().best(ffmpeg::media::Type::Audio)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let stream_index = stream.index();
    
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder().audio()?;
    
        // HARDCODED: MODEL EXPECTS 48 KHZ (SEE MD_AUDIOBIRDS_V1 SPEC)
        // TODO: MAKE CONFIGURABLE VIA MODEL METADATA WHEN MULTIPLE AUDIO MODELS EXIST
        let target_rate = 48000;
        let mut resampler = decoder.resampler(
            Sample::F32(SampleType::Packed),
            decoder.channel_layout(),
            target_rate,
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
    
        Ok(Self {
            samples,
            sample_rate: target_rate,
            channels: decoder.channels(),
        })
    }
    
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

    /// Mix all channels down to mono by averaging.
    pub fn to_mono(&self) -> Self {
        if self.channels <= 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let frames = self.samples.len() / ch;
        let mut mono = Vec::with_capacity(frames);
        for i in 0..frames {
            let sum: f32 = self.samples[i * ch..(i + 1) * ch].iter().sum();
            mono.push(sum / ch as f32);
        }
        Self {
            samples: mono,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Pad (or truncate) to exactly `secs` duration with silence.
    pub fn padded_to(&self, secs: f64) -> Self {
        let ch = self.channels.max(1) as usize;
        let target_samples = (secs * self.sample_rate as f64).ceil() as usize * ch;
        if self.samples.len() >= target_samples {
            return Self {
                samples: self.samples[..target_samples].to_vec(),
                sample_rate: self.sample_rate,
                channels: self.channels,
            };
        }
        let mut padded = self.samples.clone();
        padded.resize(target_samples, 0.0);
        Self {
            samples: padded,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Iterate over overlapping fixed-length chunks of audio.
    ///
    /// `chunk_secs` is the duration of each emitted chunk.
    /// `hop_secs` is how far the window advances between chunks.
    ///
    /// The iterator only yields full chunks; if the audio ends before a
    /// full chunk would finish, that partial tail is silently dropped.
    ///
    /// # Example
    ///
    /// ```rust
    /// for chunk in audio.chunks(5.0, 1.0) {
    ///     model.process(&chunk);
    /// }
    /// ```
    ///
    /// A 10-second file with 5-second windows and 1-second hops yields
    /// chunks covering seconds `[0-5)`, `[1-6)`, `[2-7)`, `[3-8)`, `[4-9)`.
    pub fn chunks(&self, chunk_secs: f64, hop_secs: f64) -> impl Iterator<Item = AudioData> + '_ {
        let ch = self.channels.max(1) as usize;
        let chunk_frames = (chunk_secs * self.sample_rate as f64).ceil() as usize;
        let hop_frames = (hop_secs * self.sample_rate as f64).ceil() as usize;
        let total_frames = self.samples.len() / ch;

        (0..)
            .map(move |i| i * hop_frames)
            .take_while(move |&start| start + chunk_frames <= total_frames)
            .map(move |start| {
                let s = start * ch;
                let e = s + chunk_frames * ch;
                AudioData {
                    samples: self.samples[s..e].to_vec(),
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                }
            })
    }
}

#[test]
fn smoke() {
    let a = AudioData::from_file("assets/test/audio.mp3").unwrap();
    let (min, max, rms) = a.amplitude_stats();

    println!("samples={} rate={} channels={} duration={:.3}s", a.samples.len(), a.sample_rate, a.channels, a.duration());
    println!("amp min={:.4} max={:.4} rms={:.4}", min, max, rms);
    println!("dc_offset={:.6}", a.dc_offset());
    println!("preview={:?}", a.preview(16));
}

#[test]
fn chunks_iterator() {
    // 10 seconds of mono audio @ 1000 Hz => 10000 samples
    let audio = AudioData {
        samples: (0..10000).map(|i| i as f32).collect(),
        sample_rate: 1000,
        channels: 1,
    };

    let mut count = 0;

    let _ = super::bq::BQModel::from_file_and_allocate("models/MD_AudioBirds_V1.bq", super::bq::GlobalBQ::First, None, None);
    for chunk in audio.chunks(5.0, 1.0) {
        // Each chunk is just another AudioData — pass it by reference
        assert_eq!(chunk.sample_rate, 1000);
        assert_eq!(chunk.channels, 1);
        assert_eq!(chunk.samples.len(), 5000, "each chunk is 5s = 5000 frames");

        let outputs = super::bq::process_audio(&chunk);
        println!("{:?}",outputs);
        
        let expected_first = (count * 1000) as f32;
        assert!((chunk.samples[0] - expected_first).abs() < f32::EPSILON);

        count += 1;
    }

    assert_eq!(count, 6, "10s audio with 5s window / 1s hop => 6 full chunks");
}