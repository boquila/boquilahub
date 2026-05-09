use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::sample::Type as SampleType;
use ffmpeg_next::format::Sample;
use serde::{Deserialize, Serialize};

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

        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let stream_index = stream.index();

        let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .audio()?;

        let source_rate = decoder.rate();
        let mut resampler = decoder.resampler(
            Sample::F32(SampleType::Packed),
            decoder.channel_layout(),
            source_rate,
        )?;

        let mut samples: Vec<f32> = Vec::new();

        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg::frame::Audio::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut resampled = ffmpeg::frame::Audio::empty();
                resampler.run(&decoded, &mut resampled)?;
                samples.extend(
                    resampled
                        .data(0)
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
        }

        decoder.send_eof()?;
        let mut decoded = ffmpeg::frame::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = ffmpeg::frame::Audio::empty();
            resampler.run(&decoded, &mut resampled)?;
            samples.extend(
                resampled
                    .data(0)
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }

        Ok(Self {
            samples,
            sample_rate: source_rate,
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
        let max_amplitude = self
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_square =
            self.samples.iter().map(|s| s * s).sum::<f32>() / self.samples.len().max(1) as f32;
        let rms_amplitude = mean_square.sqrt();
        (min_amplitude, max_amplitude, rms_amplitude)
    }

    /// Average sample value (should be near 0.0 for normal audio).
    pub fn dc_offset(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum = self.samples.iter().sum::<f32>();
        let average = sum / self.samples.len() as f32;
        average
    }

    /// First n samples.
    pub fn preview(&self, n: usize) -> &[f32] {
        let count = n.min(self.samples.len());
        &self.samples[..count]
    }

    pub fn resample(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate || self.samples.is_empty() {
            return self.clone();
        }
        let ch = self.channels.max(1) as usize;
        let input_frames = self.samples.len() / ch;
        let output_frames = ((input_frames as f64) * (target_rate as f64)
            / (self.sample_rate as f64))
            .ceil() as usize;
        let ratio = (input_frames as f64) / (output_frames as f64);
        let mut out = Vec::with_capacity(output_frames * ch);
        for i in 0..output_frames {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f64;
            if idx + 1 < input_frames {
                for c in 0..ch {
                    let a = self.samples[idx * ch + c];
                    let b = self.samples[(idx + 1) * ch + c];
                    out.push(a + (b - a) * frac as f32);
                }
            } else {
                for c in 0..ch {
                    out.push(self.samples[idx * ch + c]);
                }
            }
        }
        Self {
            samples: out,
            sample_rate: target_rate,
            channels: self.channels,
        }
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

    /// Extract a time range `[start_secs, start_secs + duration_secs)` from the audio.
    /// Shorter than requested range is zero-padded.
    pub fn slice(&self, start_secs: f64, duration_secs: f64) -> Self {
        let ch = self.channels.max(1) as usize;
        let start_sample = (start_secs * self.sample_rate as f64).round() as usize * ch;
        let len_samples = (duration_secs * self.sample_rate as f64).ceil() as usize * ch;

        if start_sample >= self.samples.len() {
            return Self {
                samples: vec![0.0f32; len_samples],
                sample_rate: self.sample_rate,
                channels: self.channels,
            };
        }

        let end_sample = (start_sample + len_samples).min(self.samples.len());
        let mut samples = self.samples[start_sample..end_sample].to_vec();
        samples.resize(len_samples, 0.0);
        Self {
            samples,
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
    /// ```
    /// use boquilahub::api::audio::AudioData;
    /// let audio = AudioData { samples: vec![0.0; 48000], sample_rate: 16000, channels: 1 };
    /// for chunk in audio.chunks(5.0, 1.0) {
    ///     let (min, max, rms) = chunk.amplitude_stats();
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