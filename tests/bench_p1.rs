//! Benchmarks for five P1 changes proposed in `thoughts/plan.MD` (now landed).
//!
//! Each benchmark times the *pre-change implementation* ("old"; verbatim
//! mirrors of the code as of commit `b4d3f28`) against the landed
//! implementation ("new") side by side, on identical synthetic data, in the
//! same process. The `mirrors_are_faithful_to_compute_mel` test proves the
//! old mirrors still match the real (landed) public pipeline bit-for-bit.
//! Correctness of every "new" variant is asserted bit-for-bit (or exactly,
//! where the value is a label) by the non-ignored tests below.
//!
//! Covered:
//!   1. Sidecar JSON: `serde_json::from_reader(File)` vs `BufReader` (plan:
//!      "Buffer sidecar JSON without doubling large video memory").
//!      Variants also include `fs::read` + `from_slice` (informational only:
//!      the plan rejects it for large sidecars due to the memory spike).
//!   2. `.bq` metadata: `to_vec()` + `String::from_utf8` + `from_str` vs
//!      `serde_json::from_slice` (plan: "Parse embedded model metadata without
//!      two temporary allocations").
//!   3. STFT: Hann window recomputed per frame vs hoisted out of the frame
//!      loop (plan: "Hoist the Hann window out of the STFT frame loop").
//!   4. `power_to_db`: two array passes (`mapv` + `mapv_inplace`) vs one
//!      fused pass (plan: "Fuse the power_to_db clamp").
//!   5. Top-1 softmax: full `Vec<Prob>` softmax + `max_by` vs two-pass
//!      allocation-free argmax + denominator (plan: "Compute only the top
//!      softmax class where only the top is retained"; Perch-shaped:
//!      10,000 classes).
//!
//! Run (correctness only, fast):
//!   cargo test --test bench_p1
//!
//! Run benchmarks, debug:
//!   cargo test --test bench_p1 -- --ignored --nocapture --test-threads=1
//!
//! Run benchmarks, release:
//!   cargo test --release --test bench_p1 -- --ignored --nocapture --test-threads=1
//!
//! `--test-threads=1` matters: the timing tests must not compete with each
//! other for cores. File benchmarks read from the OS page cache after the
//! warmup iteration; that is intentional (both variants see the same cache)
//! and isolates the syscall/CPU difference rather than disk speed.

use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use boquilahub::api::abstractions::{
    AIMetadataRaw, AIOutputs, PredVideo, Prob, ProbSugar, XYXY, XYXYc, sidecar_predictions_path,
};
use boquilahub::api::audio::AudioData;
use boquilahub::api::processing::pre::compute_mel;
use ndarray::Array2;
use realfft::RealFftPlanner;
use serde_json::json;

// ────────────────────────────── harness ──────────────────────────────

/// Deterministic xorshift64 PRNG so every run measures identical data.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }
}

fn bench_dir() -> PathBuf {
    let d = std::env::temp_dir().join("boquilahub_bench_p1");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Time each variant (`iters` runs after 1 warmup) and print a table.
/// The first variant is the baseline ("old"); ratios are old/new.
#[allow(clippy::type_complexity)]
fn bench_variants<'a>(
    title: &str,
    detail: &str,
    iters: usize,
    bytes: Option<u64>,
    variants: &[(&str, Box<dyn Fn() + 'a>)],
) {
    println!("\n=== {title} ===");
    println!("  {detail}");
    println!("  iters per variant: {iters} (+1 warmup)");
    let mut results: Vec<(&str, f64)> = Vec::new();
    for (name, f) in variants {
        f(); // warmup (also warms the OS page cache for file benches)
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        results.push((name, us));
    }
    let base = results[0].1;
    for (name, us) in &results {
        let ratio = base / us;
        let throughput = bytes.map(|b| b as f64 / 1e6 / (us / 1e6));
        match throughput {
            Some(mb) => println!("  {name:<42} {us:>12.1} µs/op  {ratio:>6.2}x  {mb:>8.1} MB/s"),
            None => println!("  {name:<42} {us:>12.1} µs/op  {ratio:>6.2}x"),
        }
    }
}

fn assert_bit_equal(a: &Array2<f32>, b: &Array2<f32>, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shapes differ");
    let mut bad = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            bad += 1;
            if first.is_none() {
                first = Some((i, x.to_bits(), y.to_bits()));
            }
        }
    }
    assert!(
        bad == 0,
        "{what}: {}/{} elements differ bit-for-bit, first at index {first:?}",
        bad,
        a.len()
    );
}

// ─────────────────────── synthetic data generation ───────────────────────

/// 5 s of bird-call-ish audio: three tones + noise, 32 kHz mono
/// (Perch-like sample rate; ResNet18 models resample to this range too).
const SR: u32 = 32_000;
const N_FFT: usize = 2048;
const HOP: usize = 512;
const N_MELS: usize = 128;
const TOP_DB: f32 = 80.0;

fn synthetic_signal() -> Vec<f32> {
    let n = (SR as usize) * 5;
    let mut rng = Rng::new(0x5EED_0001);
    let mut samples = vec![0.0f32; n];
    for (i, s) in samples.iter_mut().enumerate() {
        let t = i as f32 / SR as f32;
        *s = 0.5 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
            + 0.25 * (2.0 * std::f32::consts::PI * 3400.0 * t).sin()
            + 0.10 * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
            + 0.05 * rng.next_f32();
    }
    samples
}

fn synthetic_classes(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("species_{i:05}_label")).collect()
}

/// One hour of detections: 25 fps, step 5 → 18,000 analyzed frames × 2 boxes.
/// Serializes to several MB, matching a "long video sidecar".
fn synthetic_video_predictions() -> PredVideo {
    const FPS: u64 = 25;
    const SECONDS: u64 = 3600;
    const STEP: u64 = 5;
    let n_frames = FPS * SECONDS;
    let mut rng = Rng::new(0x5EED_0002);
    let mut frames = Vec::with_capacity(n_frames as usize);
    for i in 0..n_frames {
        if i % STEP == 0 {
            let dets: Vec<XYXYc> = (0..2)
                .map(|k| {
                    let (x, y) = (rng.next_f32(), rng.next_f32());
                    XYXYc::new(
                        XYXY::new(
                            x * 500.0,
                            y * 300.0,
                            x * 500.0 + 120.0,
                            y * 300.0 + 90.0,
                            0.30 + 0.69 * (0.5 + 0.5 * rng.next_f32()),
                            (i % 40) as u32 + k,
                        ),
                        format!("species_{:05}_label", (i + k as u64) % 10_000),
                    )
                })
                .collect();
            frames.push(Some(AIOutputs::ObjectDetection(dets)));
        } else {
            frames.push(None);
        }
    }
    PredVideo {
        file_path: "D:/recordings/long.mp4".into(),
        width: 1920,
        height: 1080,
        fps: FPS as f64,
        n_frames,
        step: STEP as u32,
        frames,
        wasprocessed: true,
    }
}

/// Small image sidecar: a 20-class classification result (~2 KB of JSON).
fn synthetic_image_predictions() -> AIOutputs {
    let mut rng = Rng::new(0x5EED_0003);
    AIOutputs::Classification(
        (0..20)
            .map(|c| {
                Prob::new(
                    format!("species_{c:05}_label"),
                    0.5 + 0.49 * (0.5 + 0.5 * rng.next_f32()),
                    c as u32,
                )
            })
            .collect(),
    )
}

/// Perch-shaped `.bq` file: BQMODEL magic + version + JSON metadata with
/// 10,000 classes + a 4 MB pseudo-ONNX payload (untouched by header parsing).
fn synthetic_bq_file() -> Vec<u8> {
    const ONNX_LEN: usize = 4 << 20;
    let meta = json!({
        "task": "audio_classification",
        "architecture": "perch",
        "post_processing": [],
        "classes": synthetic_classes(10_000),
        "modality": "audio",
        "audio_config": {
            "sample_rate": 32000,
            "window_size": 5.0,
            "stride": 1.0,
            "n_fft": 2048,
            "hop_length": 512,
            "top_db": 80.0,
        },
    });
    let json_bytes = serde_json::to_vec(&meta).unwrap();

    let mut content = Vec::with_capacity(12 + json_bytes.len() + 4 + ONNX_LEN);
    content.extend_from_slice(b"BQMODEL");
    content.push(1);
    content.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    content.extend_from_slice(&json_bytes);
    content.extend_from_slice(&(ONNX_LEN as u32).to_le_bytes());
    let mut rng = Rng::new(0x5EED_0004);
    content.extend((0..ONNX_LEN).map(|_| rng.next_u64() as u8));
    content
}

// ───────────────── mirrors of current production code ("old") ─────────────────
// Verbatim copies of the functions under change so both variants run in the
// same process on the same data. Fidelity is proven by
// `mirrors_are_faithful_to_compute_mel`, which compares the old-mirror
// pipeline against the real public `compute_mel` bit-for-bit.

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank_mirror(sr: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
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

/// Mirror of `src/api/processing/pre.rs` `stft` (Hann recomputed per sample
/// of every frame) — the code the plan proposes to change.
#[allow(clippy::needless_range_loop)]
fn stft_old(signal: &[f32], n_fft: usize, hop_length: usize) -> Array2<f32> {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; signal.len() + 2 * pad];
    padded[pad..pad + signal.len()].copy_from_slice(signal);

    let n_frames = signal.len() / hop_length + 1;
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

/// Proposed `stft`: identical coefficients, computed once per call.
#[allow(clippy::needless_range_loop)]
fn stft_new(signal: &[f32], n_fft: usize, hop_length: usize) -> Array2<f32> {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let window: Vec<f32> = (0..n_fft)
        .map(|j| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * j as f32 / (n_fft - 1) as f32).cos())
        .collect();

    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; signal.len() + 2 * pad];
    padded[pad..pad + signal.len()].copy_from_slice(signal);

    let n_frames = signal.len() / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;
    let mut spec = Array2::zeros((n_freqs, n_frames));

    let mut windowed = vec![0.0f32; n_fft];
    let mut spectrum = r2c.make_output_vec();

    for i in 0..n_frames {
        let start = i * hop_length;
        for j in 0..n_fft {
            windowed[j] = padded[start + j] * window[j];
        }

        r2c.process(&mut windowed, &mut spectrum).unwrap();

        for j in 0..n_freqs {
            let power = spectrum[j].re * spectrum[j].re + spectrum[j].im * spectrum[j].im;
            spec[[j, i]] = power;
        }
    }

    spec
}

/// Mirror of `power_to_db`: `mapv` then a second clamping `mapv_inplace`.
fn power_to_db_old(spec: &Array2<f32>, top_db: f32) -> Array2<f32> {
    let max_power = spec.iter().fold(0.0f32, |a, &b| a.max(b));
    let ref_power = max_power.max(1e-10);
    let mut db = spec.mapv(|v| 10.0 * (v / ref_power).max(1e-10).log10());
    let min_db = -top_db;
    db.mapv_inplace(|v| v.max(min_db));
    db
}

/// Proposed `power_to_db`: clamp fused into a single pass.
fn power_to_db_new(spec: &Array2<f32>, top_db: f32) -> Array2<f32> {
    let max_power = spec.iter().fold(0.0f32, |a, &b| a.max(b));
    let ref_power = max_power.max(1e-10);
    let min_db = -top_db;
    spec.mapv(|v| (10.0 * (v / ref_power).max(1e-10).log10()).max(min_db))
}

/// Mirror of the Perch/ResNet18 multiclass post-processing: build a
/// `Vec<Prob>` with a cloned label per class, softmax all of it, keep the max
/// (uses the real production `logits_to_probs`).
fn top1_old(logits: &Array2<f32>, classes: &[String]) -> Vec<Prob> {
    let mut out = Vec::with_capacity(logits.nrows());
    for i in 0..logits.nrows() {
        let mut probs: Vec<Prob> = classes
            .iter()
            .enumerate()
            .map(|(c, label)| Prob::new(label.clone(), logits[[i, c]], c as u32))
            .collect();
        probs.logits_to_probs();
        let top = probs
            .into_iter()
            .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap())
            .unwrap();
        out.push(top);
    }
    out
}

/// Proposed two-pass top-1: argmax (ties resolved last-wins, matching
/// `max_by`), allocation-free denominator in class order, one cloned label.
/// The top probability is `1.0 / sum` — bit-identical to the old path, whose
/// winning element is always `exp(0.0) / sum`.
fn top1_new(logits: &Array2<f32>, classes: &[String]) -> Vec<Prob> {
    let mut out = Vec::with_capacity(logits.nrows());
    for i in 0..logits.nrows() {
        let n = classes.len();
        let mut best = 0usize;
        let mut best_logit = logits[[i, 0]];
        for c in 1..n {
            let l = logits[[i, c]];
            if l >= best_logit {
                best = c;
                best_logit = l;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..n {
            sum += (logits[[i, c]] - best_logit).exp();
        }
        out.push(Prob::new(classes[best].clone(), 1.0 / sum, best as u32));
    }
    out
}

/// Mirror of `parse_bq_header`'s JSON section (the part the plan changes):
/// copy to `Vec`, wrap in `String`, parse with `from_str`.
fn parse_bq_header_old(content: &[u8]) -> AIMetadataRaw {
    let json_length = u32::from_le_bytes(content[8..12].try_into().unwrap()) as usize;
    let json_end = 12 + json_length;
    let json_str = String::from_utf8(content[12..json_end].to_vec()).unwrap();
    serde_json::from_str(&json_str).unwrap()
}

/// Proposed parse: `from_slice` validates UTF-8 itself, no temporaries.
fn parse_bq_header_new(content: &[u8]) -> AIMetadataRaw {
    let json_length = u32::from_le_bytes(content[8..12].try_into().unwrap()) as usize;
    let json_end = 12 + json_length;
    serde_json::from_slice(&content[12..json_end]).unwrap()
}

/// Mirror of pre-change `PredVideo::new_simple`: `exists()` preflight plus
/// unbuffered `from_reader(File)`. Verbatim on purpose — keep in sync with
/// the pre-change source, do not modernize.
#[allow(clippy::collapsible_if)]
fn pred_video_new_simple_old(file_path: std::path::PathBuf) -> PredVideo {
    if let Ok(path) = sidecar_predictions_path(&file_path) {
        if path.exists() {
            if let Ok(file) = File::open(&path) {
                if let Ok(mut cached) = serde_json::from_reader::<_, PredVideo>(file) {
                    cached.file_path = file_path;
                    return cached;
                }
            }
        }
    }
    PredVideo {
        file_path,
        width: 0,
        height: 0,
        fps: 0.0,
        n_frames: 0,
        step: 1,
        frames: Vec::new(),
        wasprocessed: false,
    }
}

/// Mirror of pre-change `AIOutputs::from_file`: unbuffered `from_reader(File)`.
fn aioutputs_from_file_old(path: &std::path::Path) -> AIOutputs {
    serde_json::from_reader(File::open(path).unwrap()).unwrap()
}

// ─────────────────────── correctness tests (always run) ───────────────────────

/// The "old" mirrors must reproduce the real (landed) public `compute_mel`
/// pipeline bit-for-bit. This both validates the pre-change mirrors used in
/// the benchmarks below and proves the landed changes preserved behavior
/// exactly.
#[test]
fn mirrors_are_faithful_to_compute_mel() {
    let samples = synthetic_signal();
    let audio = AudioData {
        samples: samples.clone(),
        sample_rate: SR,
        channels: 1,
    };
    let real = compute_mel(&audio, N_FFT, HOP, N_MELS, TOP_DB);
    let mine = power_to_db_old(
        &mel_filterbank_mirror(SR, N_FFT, N_MELS).dot(&stft_old(&samples, N_FFT, HOP)),
        TOP_DB,
    );
    assert_bit_equal(&real, &mine, "mirrored pipeline vs real compute_mel");
}

#[test]
fn stft_hann_bit_exact() {
    let samples = synthetic_signal();
    let old = stft_old(&samples, N_FFT, HOP);
    let new = stft_new(&samples, N_FFT, HOP);
    assert_bit_equal(&old, &new, "hoisted-window STFT vs per-frame-cos STFT");
}

#[test]
fn power_to_db_clamp_bit_exact() {
    let samples = synthetic_signal();
    let mel = mel_filterbank_mirror(SR, N_FFT, N_MELS).dot(&stft_old(&samples, N_FFT, HOP));
    let old = power_to_db_old(&mel, TOP_DB);
    let new = power_to_db_new(&mel, TOP_DB);
    assert_bit_equal(&old, &new, "fused-clamp power_to_db vs two-pass");
}

#[test]
fn top1_softmax_exact() {
    const N_CLASSES: usize = 10_000;
    const N_ROWS: usize = 16;
    let classes = synthetic_classes(N_CLASSES);
    let mut rng = Rng::new(0x5EED_0005);
    let mut logits = Array2::<f32>::zeros((N_ROWS, N_CLASSES));
    for r in 0..N_ROWS {
        for c in 0..N_CLASSES {
            logits[[r, c]] = rng.next_f32() * 20.0 - 5.0;
        }
    }
    let old = top1_old(&logits, &classes);
    let new = top1_new(&logits, &classes);
    assert_eq!(old.len(), new.len());
    for (i, (o, n)) in old.iter().zip(new.iter()).enumerate() {
        assert_eq!(o.label, n.label, "row {i}: label mismatch");
        assert_eq!(o.class_id, n.class_id, "row {i}: class_id mismatch");
        assert_eq!(
            o.prob.to_bits(),
            n.prob.to_bits(),
            "row {i}: prob not bit-identical (old {} vs new {})",
            o.prob,
            n.prob
        );
    }
}

#[test]
fn sidecar_json_variants_identical() {
    let dir = bench_dir();

    let pv = synthetic_video_predictions();
    let path = dir.join("long_predictions.json");
    std::fs::write(&path, serde_json::to_string(&pv).unwrap()).unwrap();

    let from_file: PredVideo = serde_json::from_reader(File::open(&path).unwrap()).unwrap();
    let from_buf: PredVideo =
        serde_json::from_reader(BufReader::new(File::open(&path).unwrap())).unwrap();
    let from_slice: PredVideo = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let via_new_simple = PredVideo::new_simple(dir.join("long.mp4"));

    let a = serde_json::to_string(&from_file).unwrap();
    assert_eq!(a, serde_json::to_string(&from_buf).unwrap());
    assert_eq!(a, serde_json::to_string(&from_slice).unwrap());
    // new_simple overrides file_path with the caller-supplied path.
    let mut expected = from_file.clone();
    expected.file_path = dir.join("long.mp4");
    assert_eq!(
        serde_json::to_string(&expected).unwrap(),
        serde_json::to_string(&via_new_simple).unwrap()
    );

    let io = synthetic_image_predictions();
    let io_path = dir.join("img_predictions.json");
    std::fs::write(&io_path, serde_json::to_string(&io).unwrap()).unwrap();
    let small: AIOutputs =
        serde_json::from_reader(BufReader::new(File::open(&io_path).unwrap())).unwrap();
    let small_prod = AIOutputs::from_file(&io_path).unwrap();
    assert_eq!(
        serde_json::to_string(&small).unwrap(),
        serde_json::to_string(&small_prod).unwrap()
    );
}

#[test]
fn bq_metadata_variants_identical() {
    let content = synthetic_bq_file();
    let old = parse_bq_header_old(&content);
    let new = parse_bq_header_new(&content);
    assert_eq!(format!("{old:?}"), format!("{new:?}"));
    assert_eq!(old.classes.len(), 10_000);
    assert!(old.audio_config.is_some());
}

// ─────────────────────── benchmarks (ignored by default) ───────────────────────

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_sidecar_video_json() {
    let dir = bench_dir();
    let path = dir.join("long_predictions.json");
    let pv = synthetic_video_predictions();
    std::fs::write(&path, serde_json::to_string(&pv).unwrap()).unwrap();
    drop(pv);
    let bytes = std::fs::metadata(&path).unwrap().len();
    let media_path = dir.join("long.mp4");
    std::fs::write(&media_path, b"").unwrap();

    bench_variants(
        "sidecar JSON - long video (PredVideo)",
        &format!(
            "sidecar: {bytes} bytes (1 h @ 25 fps, step 5, 2 boxes/frame); warm OS page cache"
        ),
        3,
        Some(bytes),
        &[
            (
                "old: from_reader(File) + exists() [pre-change new_simple]",
                Box::new(|| {
                    let v = pred_video_new_simple_old(media_path.clone());
                    let _ = black_box(v);
                }),
            ),
            (
                "new: from_reader(BufReader(File)) [landed new_simple]",
                Box::new(|| {
                    let v = PredVideo::new_simple(media_path.clone());
                    let _ = black_box(v);
                }),
            ),
            (
                "alt: fs::read + from_slice (memory spike - informational)",
                Box::new(|| {
                    let b = std::fs::read(&path).unwrap();
                    let v: PredVideo = serde_json::from_slice(&b).unwrap();
                    let _ = black_box(v);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_sidecar_small_json() {
    let dir = bench_dir();
    let path = dir.join("img_predictions.json");
    std::fs::write(
        &path,
        serde_json::to_string(&synthetic_image_predictions()).unwrap(),
    )
    .unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();

    bench_variants(
        "sidecar JSON - small image (AIOutputs)",
        &format!("sidecar: {bytes} bytes (20-class classification); warm OS page cache"),
        200,
        Some(bytes),
        &[
            (
                "old: from_reader(File) [pre-change from_file]",
                Box::new(|| {
                    let v = aioutputs_from_file_old(&path);
                    let _ = black_box(v);
                }),
            ),
            (
                "new: from_reader(BufReader(File)) [landed from_file]",
                Box::new(|| {
                    let v = AIOutputs::from_file(&path).unwrap();
                    let _ = black_box(v);
                }),
            ),
            (
                "alt: fs::read + from_slice (small files only)",
                Box::new(|| {
                    let b = std::fs::read(&path).unwrap();
                    let v: AIOutputs = serde_json::from_slice(&b).unwrap();
                    let _ = black_box(v);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_bq_metadata_parse() {
    let content = synthetic_bq_file();
    let json_bytes = u32::from_le_bytes(content[8..12].try_into().unwrap()) as u64;
    let content = black_box(content);

    bench_variants(
        ".bq embedded metadata parse (10,000-class Perch-shaped)",
        &format!("JSON section: {json_bytes} bytes; in-memory buffer (parse only, disk excluded)"),
        50,
        None,
        &[
            (
                "old: to_vec + String::from_utf8 + from_str",
                Box::new(|| {
                    let m = parse_bq_header_old(black_box(&content));
                    let _ = black_box(m);
                }),
            ),
            (
                "new: from_slice [proposed]",
                Box::new(|| {
                    let m = parse_bq_header_new(black_box(&content));
                    let _ = black_box(m);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_stft_hann_window() {
    let samples = black_box(synthetic_signal());
    bench_variants(
        "STFT Hann window (5 s @ 32 kHz, n_fft=2048, hop=512)",
        "313 frames; old recomputes cos() 2048x per frame, new hoists it once",
        10,
        None,
        &[
            (
                "old: cos() per sample per frame",
                Box::new(|| {
                    let s = stft_old(black_box(&samples), N_FFT, HOP);
                    let _ = black_box(s);
                }),
            ),
            (
                "new: window hoisted before frame loop",
                Box::new(|| {
                    let s = stft_new(black_box(&samples), N_FFT, HOP);
                    let _ = black_box(s);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_power_to_db_clamp() {
    let samples = synthetic_signal();
    let mel =
        black_box(mel_filterbank_mirror(SR, N_FFT, N_MELS).dot(&stft_old(&samples, N_FFT, HOP)));
    bench_variants(
        "power_to_db clamp fusion (128x313 mel = 40k elements)",
        "old: mapv + mapv_inplace (two passes + same alloc); new: single fused mapv",
        200,
        None,
        &[
            (
                "old: mapv then mapv_inplace clamp",
                Box::new(|| {
                    let d = power_to_db_old(black_box(&mel), TOP_DB);
                    let _ = black_box(d);
                }),
            ),
            (
                "new: clamp fused into one mapv",
                Box::new(|| {
                    let d = power_to_db_new(black_box(&mel), TOP_DB);
                    let _ = black_box(d);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_mel_end_to_end() {
    let samples = synthetic_signal();
    let audio = black_box(AudioData {
        samples: samples.clone(),
        sample_rate: SR,
        channels: 1,
    });
    bench_variants(
        "mel pipeline end-to-end (compute_mel shape: 5 s @ 32 kHz)",
        "old: pre-change pipeline (mirrored); new: real landed compute_mel (hoisted Hann + fused power_to_db)",
        10,
        None,
        &[
            (
                "old: per-frame cos + two-pass power_to_db (mirrored)",
                Box::new(|| {
                    let m = power_to_db_old(
                        &mel_filterbank_mirror(SR, N_FFT, N_MELS)
                            .dot(&stft_old(&samples, N_FFT, HOP)),
                        TOP_DB,
                    );
                    let _ = black_box(m);
                }),
            ),
            (
                "new: production compute_mel (landed)",
                Box::new(|| {
                    let m = compute_mel(black_box(&audio), N_FFT, HOP, N_MELS, TOP_DB);
                    let _ = black_box(m);
                }),
            ),
        ],
    );
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture --test-threads=1"]
fn bench_top1_softmax() {
    const N_CLASSES: usize = 10_000;
    const N_ROWS: usize = 64;
    let classes = black_box(synthetic_classes(N_CLASSES));
    let mut rng = Rng::new(0x5EED_0006);
    let mut logits = Array2::<f32>::zeros((N_ROWS, N_CLASSES));
    for r in 0..N_ROWS {
        for c in 0..N_CLASSES {
            logits[[r, c]] = rng.next_f32() * 20.0 - 5.0;
        }
    }
    let logits = black_box(logits);

    bench_variants(
        "top-1 softmax (64 windows x 10,000 classes)",
        "old: 640k Prob+String allocations per call; new: 64 allocations",
        5,
        None,
        &[
            (
                "old: full Vec<Prob> + softmax + max_by",
                Box::new(|| {
                    let v = top1_old(black_box(&logits), black_box(&classes));
                    let _ = black_box(v);
                }),
            ),
            (
                "new: two-pass argmax + 1/sum",
                Box::new(|| {
                    let v = top1_new(black_box(&logits), black_box(&classes));
                    let _ = black_box(v);
                }),
            ),
        ],
    );
}
