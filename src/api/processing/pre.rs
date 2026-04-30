use crate::api::abstractions::XYXY;
use crate::api::audio::AudioData;
use fast_image_resize::{self as fir};
use image::{ImageBuffer, Rgb};
use ndarray::{s, Array, Array2, Ix4};
use realfft::RealFftPlanner;

const SCALE: f32 = 1.0 / 255.0;

pub enum TensorFormat {
    NCHW, // Batch, Channel, Height, Width
    NHWC, // Batch, Height, Width, Channel
}

fn fast_resize(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    new_width: u32,
    new_height: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // Create source image view
    let src_image =
        fir::images::Image::from_vec_u8(width, height, img.as_raw().clone(), fir::PixelType::U8x3)
            .unwrap();

    // Create destination image
    let mut dst_image = fir::images::Image::new(new_width, new_height, fir::PixelType::U8x3);

    let mut resizer = fir::Resizer::new();
    let options = fir::ResizeOptions::new().resize_alg(fast_image_resize::ResizeAlg::Nearest);

    resizer
        .resize(&src_image, &mut dst_image, &options)
        .unwrap();

    // Convert back to ImageBuffer
    ImageBuffer::from_raw(new_width, new_height, dst_image.into_vec()).unwrap()
}

pub fn imgbuf_to_input_array(
    batch_size: usize,
    input_depth: usize,
    input_height: u32,
    input_width: u32,
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    format: &TensorFormat,
) -> (Array<f32, Ix4>, u32, u32) {
    let (img_width, img_height) = img.dimensions();

    let resized = fast_resize(img, input_width, input_height);

    let (h, w) = (input_height as usize, input_width as usize);
    let mut input = match format {
        TensorFormat::NCHW => Array::zeros((batch_size, input_depth, h, w)),
        TensorFormat::NHWC => Array::zeros((batch_size, h, w, input_depth)),
    };

    let input_slice = input.as_slice_mut().unwrap();

    for (x, y, pixel) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        let [r, g, b] = pixel.0;
        let (r, g, b) = (r as f32 * SCALE, g as f32 * SCALE, b as f32 * SCALE);

        match format {
            TensorFormat::NCHW => {
                // Layout: [batch, channel, height, width]
                let idx = y * w + x;
                input_slice[idx] = r;
                input_slice[h * w + idx] = g;
                input_slice[2 * h * w + idx] = b;
            }
            TensorFormat::NHWC => {
                // Layout: [batch, height, width, channel]
                let idx = (y * w + x) * 3;
                input_slice[idx] = r;
                input_slice[idx + 1] = g;
                input_slice[idx + 2] = b;
            }
        }
    }
    (input, img_width, img_height)
}

pub fn slice_image(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    bbox: &XYXY,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (img_width, img_height) = img.dimensions();

    let x1 = (bbox.x1.max(0.0) as u32).min(img_width);
    let y1 = (bbox.y1.max(0.0) as u32).min(img_height);
    let x2 = (bbox.x2.max(0.0) as u32).min(img_width);
    let y2 = (bbox.y2.max(0.0) as u32).min(img_height);

    let width = x2 - x1;
    let height = y2 - y1;

    let mut sliced = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x1 + x, y1 + y);
            sliced.put_pixel(x, y, *pixel);
        }
    }

    sliced
}

// ── Audio preprocessing: AudioData → mel spectrogram → NCHW tensor ──

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Build a mel filterbank matrix of shape `[n_mels, n_fft/2 + 1]`.
fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr as f32 / 2.0);
    let n_points = n_mels + 2;

    let mel_pts: Vec<f32> = (0..n_points)
        .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
        .collect();

    let freq_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_bins: Vec<usize> = freq_pts
        .iter()
        .map(|&f| (f / sr as f32 * n_fft as f32).round() as usize)
        .map(|b| b.min(n_fft / 2))
        .collect();

    let n_freqs = n_fft / 2 + 1;
    let mut mel_fb = Array2::zeros((n_mels, n_freqs));

    for i in 0..n_mels {
        let left = fft_bins[i];
        let center = fft_bins[i + 1];
        let right = fft_bins[i + 2];

        for j in left..=right {
            if j > n_fft / 2 {
                break;
            }
            let weight = if j < center {
                (j - left) as f32 / ((center - left).max(1) as f32)
            } else if j == center {
                1.0
            } else {
                (right - j) as f32 / ((right - center).max(1) as f32)
            };
            mel_fb[[i, j]] = weight;
        }
    }

    mel_fb
}

/// Short-time Fourier transform with center padding (librosa style).
/// Returns power spectrogram of shape `[n_fft/2 + 1, n_frames]`.
fn stft(signal: &[f32], n_fft: usize, hop_length: usize) -> Array2<f32> {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; signal.len() + 2 * pad];
    padded[pad..pad + signal.len()].copy_from_slice(signal);

    let n_frames = signal.len() / hop_length + 1; // floor division + 1
    let n_freqs = n_fft / 2 + 1;
    let mut spec = Array2::zeros((n_freqs, n_frames));

    let mut windowed = vec![0.0f32; n_fft];
    let mut spectrum = r2c.make_output_vec();

    for i in 0..n_frames {
        let start = i * hop_length;
        for j in 0..n_fft {
            let hann =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * j as f32 / (n_fft - 1) as f32).cos();
            windowed[j] = padded[start + j] * hann;
        }

        r2c.process(&mut windowed, &mut spectrum).unwrap();

        for j in 0..n_freqs {
            let power = spectrum[j].re * spectrum[j].re + spectrum[j].im * spectrum[j].im;
            spec[[j, i]] = power;
        }
    }

    spec
}

/// Convert power spectrogram to decibels and clamp dynamic range.
/// Matches librosa `power_to_db(ref=np.max)`: peak is 0 dB, floor is `-top_db`.
fn power_to_db(spec: &Array2<f32>, top_db: f32) -> Array2<f32> {
    let max_power = spec.iter().fold(0.0f32, |a, &b| a.max(b));
    let ref_power = max_power.max(1e-10);
    let mut db = spec.mapv(|v| 10.0 * (v / ref_power).max(1e-10).log10());
    let min_db = -top_db;
    db.mapv_inplace(|v| v.max(min_db));
    db
}

/// Compute a single mel spectrogram from a mono signal.
fn mel_spectrogram(
    signal: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    top_db: f32,
) -> Array2<f32> {
    let spec = stft(signal, n_fft, hop_length);
    let mel_fb = mel_filterbank(sr, n_fft, n_mels);
    let mel_spec = mel_fb.dot(&spec);
    power_to_db(&mel_spec, top_db)
}

/// Transform `AudioData` into a batched NCHW tensor for audio models.
///
/// - Slices audio into `window_secs` chunks with `hop_secs` stride.
/// - Shorter audio is zero-padded to exactly one window.
/// - Computes a mel spectrogram per window.
/// - Stacks into `[batch, 1, n_mels, time_steps]`; shorter spectrograms are
///   zero-padded on the time axis to match the longest in the batch.
pub fn audio_to_input_array(
    audio: &AudioData,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    top_db: f32,
    window_secs: f64,
    hop_secs: f64,
) -> Array<f32, Ix4> {
    let mono = audio.to_mono();

    let windows: Vec<AudioData> = if mono.duration() < window_secs {
        vec![mono.padded_to(window_secs)]
    } else {
        mono.chunks(window_secs, hop_secs).collect()
    };

    let batch = windows.len();
    let specs: Vec<Array2<f32>> = windows
        .iter()
        .map(|w| mel_spectrogram(&w.samples, w.sample_rate, n_fft, hop_length, n_mels, top_db))
        .collect();

    let max_time = specs.iter().map(|s| s.ncols()).max().unwrap_or(0);

    let mut input = Array::zeros((batch, 1, n_mels, max_time));
    for (i, spec) in specs.iter().enumerate() {
        let n_times = spec.ncols();
        input.slice_mut(s![i, 0, .., ..n_times]).assign(spec);
    }

    input
}

#[test]
fn mel_spectrogram_shape() {
    // 5 seconds of silence @ 48 kHz mono
    let audio = AudioData {
        samples: vec![0.0f32; 240_000],
        sample_rate: 48_000,
        channels: 1,
    };

    let input = audio_to_input_array(&audio, 2048, 512, 224, 80.0, 5.0, 1.0);

    assert_eq!(input.shape(), &[1, 1, 224, 469]);
}
