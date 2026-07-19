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

        let mut samples: Vec<f32> = Vec::new();
        // Built lazily from the first decoded frame's actual props. Some
        // containers (notably PCM WAVs) leave the decoder's channel_layout /
        // sample_format unset pre-decode; initialising the resampler from
        // those guesses then bails with "Input changed" on the first frame.
        let mut resampler: Option<ffmpeg::software::resampling::Context> = None;

        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet)?;
            Self::collect_decoded(&mut decoder, &mut resampler, &mut samples, source_rate)?;
        }

        decoder.send_eof()?;
        Self::collect_decoded(&mut decoder, &mut resampler, &mut samples, source_rate)?;

        Ok(Self {
            samples,
            sample_rate: source_rate,
            channels: decoder.channels(),
        })
    }

    fn collect_decoded(
        decoder: &mut ffmpeg::decoder::Audio,
        resampler: &mut Option<ffmpeg::software::resampling::Context>,
        samples: &mut Vec<f32>,
        target_rate: u32,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = ffmpeg::frame::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let canonical_layout = ffmpeg::ChannelLayout::default(decoder.channels() as i32);
            // libswresample compares the frame's layout to the configured one
            // via `av_channel_layout_compare`, which is sensitive to
            // order=UNSPEC vs NATIVE even when the channel count matches.
            // Pin the frame's layout to the canonical NATIVE-order default so
            // we never see a spurious "Input changed" on the first frame.
            decoded.set_channel_layout(canonical_layout);
            if resampler.is_none() {
                *resampler = Some(ffmpeg::software::resampling::Context::get(
                    decoded.format(),
                    canonical_layout,
                    decoded.rate(),
                    Sample::F32(SampleType::Packed),
                    canonical_layout,
                    target_rate,
                )?);
            }
            let r = resampler.as_mut().unwrap();
            let mut resampled = ffmpeg::frame::Audio::empty();
            r.run(&decoded, &mut resampled)?;
            // data(0) returns the full SIMD-aligned linesize buffer, which is
            // longer than the actual sample data. Slice it to the real length
            // (samples × channels × 4 bytes for packed f32) to avoid reading
            // uninitialised padding as audio.
            let real_bytes = resampled.samples() * decoder.channels() as usize * 4;
            samples.extend(
                resampled.data(0)[..real_bytes]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        Ok(())
    }

    // ffmpeg-next only opens input by path, so round-trip bytes through a temp file.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ffmpeg::Error> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("bq_audio_{}_{n}", std::process::id()));
        std::fs::write(&path, data).map_err(|_| ffmpeg::Error::External)?;
        let result = Self::from_file(&path);
        let _ = std::fs::remove_file(&path);
        result
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
    /// ```no_run
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
