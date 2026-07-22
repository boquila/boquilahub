use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Prob {
    pub label: String,
    pub prob: f32,
    pub class_id: u32,
}

impl Prob {
    pub fn new(label: String, prob: f32, class_id: u32) -> Self {
        Self { label, prob, class_id }
    }
}

pub trait ProbSugar {
    fn highest_confidence(&self) -> String;
    fn top(&self) -> Option<&Prob>;
    fn logits_to_probs(&mut self);
}

impl ProbSugar for Vec<Prob> {
    fn highest_confidence(&self) -> String {
        self.top()
            .map(|p| p.label.clone())
            .unwrap_or_else(|| String::from("no prediction"))
    }

    fn top(&self) -> Option<&Prob> {
        self.iter().max_by(|a, b| {
            a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn logits_to_probs(&mut self) {
        let max_logit = self.iter().map(|p| p.prob).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = self.iter().map(|p| (p.prob - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (p, e) in self.iter_mut().zip(exps.iter()) {
            p.prob = e / sum;
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BitMatrix {
    pub data: bitvec::vec::BitVec,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SEGc {
    pub mask: BitMatrix,
    pub bbox: XYXYc,
}

impl SEGc {
    pub fn new(mask: BitMatrix, bbox: XYXYc) -> Self {
        Self {
            mask,
            bbox,
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct XYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u32,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct XY {
    pub x: f32,
    pub y: f32,
    pub prob: f32,
    pub class_id: u32,
}

impl XYXY {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, prob: f32, class_id: u32) -> Self {
        Self {x1,y1,x2,y2,prob,class_id}
    }

    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXY) -> f32 {
        let x_left = self.x1.max(other.x1);
        let y_top = self.y1.max(other.y1);
        let x_right = self.x2.min(other.x2);
        let y_bottom = self.y2.min(other.y2);

        (x_right - x_left).max(0.0) * (y_bottom - y_top).max(0.0)
    }

    pub fn iou(&self, other: &XYXY) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }

    pub fn check(&self) -> bool {
        self.x2 >= self.x1 && self.y2 >= self.y1 && self.prob >= 0.0 && self.prob <= 1.0
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct AIMetadataRaw {
    pub task: String,
    pub architecture: Option<String>, // yolo, efficientnet, whatever else
    pub post_processing: Vec<String>,
    pub classes: Vec<String>,
    pub modality: Option<String>, // "image" or "audio", defaults to "image"
    pub audio_config: Option<AudioConfig>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AudioConfig {
    pub sample_rate: u32, // e.g. 48000
    pub window_size: f32, // seconds, e.g. 5.0
    pub stride: f32,      // seconds, e.g. 1.0
    pub n_fft: u32,       // e.g. 2048
    pub hop_length: u32,  // e.g. 512
    pub top_db: f32,      // e.g. 80.0
}

/// For `input_path/file.ext`, returns `input_path/file_predictions.json`.
/// Shared by every `Pred*` type so the sidecar layout stays uniform.
pub fn sidecar_predictions_path(
    input_path: &std::path::Path,
) -> std::io::Result<std::path::PathBuf> {
    let stem = input_path
        .file_stem()
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid input path")
        })?
        .to_str()
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Non-UTF-8 file path")
        })?;
    Ok(input_path.with_file_name(format!("{}_predictions.json", stem)))
}

fn load_sidecar_aioutput(file_path: &std::path::Path) -> Option<AIOutputs> {
    let path = sidecar_predictions_path(file_path).ok()?;
    if !path.exists() {
        return None;
    }
    AIOutputs::from_file(path).ok()
}

/// Shared shape for the three media prediction types. Lets the GUI talk to
/// "any media file that has predictions attached" without caring whether it's
/// an image, an audio clip, or a video.
pub trait Pred {
    fn file_path(&self) -> &std::path::Path;
    fn is_processed(&self) -> bool;

    /// JSON written to the sidecar `_predictions.json`. Image and audio dump
    /// the bare `AIOutputs` (one prediction per file); video dumps the whole
    /// `PredVideo` so frames-as-array + probe metadata round-trip.
    fn predictions_json(&self) -> serde_json::Result<String>;

    fn predictions_file_path(&self) -> std::io::Result<std::path::PathBuf> {
        sidecar_predictions_path(self.file_path())
    }

    /// Write the sidecar JSON next to the source file. Sync IO inside; the
    /// existing GUI callers wrap it in `tokio::spawn` for fire-and-forget.
    /// Serde errors are folded into `io::Error` so the trait stays free of
    /// extra error-crate dependencies.
    fn write_predictions(&self) -> std::io::Result<()> {
        let path = self.predictions_file_path()?;
        let json = self
            .predictions_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }
}

/// Vec-level operations every `Vec<impl Pred>` gets for free.
pub trait PredListSugar {
    fn count_processed(&self) -> usize;
    fn get_progress(&self) -> f32;
}

impl<T: Pred> PredListSugar for Vec<T> {
    fn count_processed(&self) -> usize {
        self.iter().filter(|p| p.is_processed()).count()
    }

    fn get_progress(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.count_processed() as f32 / self.len() as f32
    }
}

#[derive(Clone)]
pub struct PredImg {
    pub file_path: std::path::PathBuf,
    pub aioutput: Option<AIOutputs>,
    pub wasprocessed: bool,
}

impl PredImg {
    pub fn new_simple(file_path: std::path::PathBuf) -> Self {
        let aioutput = load_sidecar_aioutput(&file_path);
        PredImg {
            wasprocessed: aioutput.is_some(),
            aioutput,
            file_path,
        }
    }

    pub fn reset(&mut self) {
        self.wasprocessed = false;
    }
}

impl Pred for PredImg {
    fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }
    fn is_processed(&self) -> bool {
        self.wasprocessed
    }
    fn predictions_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.aioutput)
    }
}

pub trait AudioProbSugar {
    fn highest_confidence(&self) -> String;
}

impl AudioProbSugar for Vec<AudioProb> {
    fn highest_confidence(&self) -> String {
        self.iter()
            .max_by(|a, b| a.prediction.prob.partial_cmp(&b.prediction.prob).unwrap_or(std::cmp::Ordering::Equal))
            .map(|audio| audio.prediction.label.clone())
            .unwrap_or_else(|| String::from("no prediction"))
    }
}

#[derive(Clone)]
pub struct PredAudio {
    pub file_path: std::path::PathBuf,
    pub aioutput: Option<AIOutputs>,
    pub wasprocessed: bool,
}

impl PredAudio {
    pub fn new_simple(file_path: std::path::PathBuf) -> Self {
        let aioutput = load_sidecar_aioutput(&file_path);
        PredAudio {
            wasprocessed: aioutput.is_some(),
            aioutput,
            file_path,
        }
    }

    pub fn reset(&mut self) {
        self.wasprocessed = false;
    }

    pub fn audio_predictions(&self) -> Option<&[AudioProb]> {
        match self.aioutput.as_ref() {
            Some(AIOutputs::AudioClassification(p)) => Some(p),
            _ => None,
        }
    }
}

impl Pred for PredAudio {
    fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }
    fn is_processed(&self) -> bool {
        self.wasprocessed
    }
    fn predictions_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.aioutput)
    }
}

/// A video that has been (or will be) analyzed frame by frame.
///
/// `frames[i]` is the prediction for frame `i`:
/// - `None`              → that frame was skipped (we only analyze every `step` frames).
/// - `Some(aioutput)`    → that frame was analyzed; the output may itself be empty.
#[derive(Clone, Serialize, Deserialize)]
pub struct PredVideo {
    pub file_path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub n_frames: u64,
    pub step: u32,
    pub frames: Vec<Option<AIOutputs>>,
    pub wasprocessed: bool,
}

impl PredVideo {
    /// Cheap constructor: stores the path and loads a sidecar
    /// `_predictions.json` if one exists (same shape as `PredImg::new_simple`
    /// / `PredAudio::new_simple`). The ffmpeg probe is deferred to
    /// [`Self::hydrate`] so picking 100 videos doesn't pay a 100x
    /// ffmpeg-init cost upfront.
    pub fn new_simple(file_path: std::path::PathBuf) -> Self {
        if let Ok(path) = sidecar_predictions_path(&file_path) {
            if path.exists() {
                if let Ok(file) = std::fs::File::open(&path) {
                    if let Ok(mut cached) = serde_json::from_reader::<_, PredVideo>(file) {
                        // Trust the caller-supplied path in case the video
                        // moved since the predictions were saved.
                        cached.file_path = file_path;
                        return cached;
                    }
                }
            }
        }
        Self {
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

    /// True once metadata has been filled (sidecar load or `hydrate`). Used
    /// as the cheap "should I probe?" check.
    pub fn is_hydrated(&self) -> bool {
        self.n_frames != 0 || !self.frames.is_empty()
    }

    /// Fill in probe metadata + allocate `frames`. No-op if already hydrated
    /// (sidecar already loaded it, or a prior open did). Called the first
    /// time the user navigates to this video.
    pub fn hydrate(&mut self, width: u32, height: u32, fps: f64, n_frames: u64) {
        if self.is_hydrated() {
            return;
        }
        self.width = width;
        self.height = height;
        self.fps = fps;
        self.n_frames = n_frames;
        self.frames = vec![None; n_frames as usize];
    }

    pub fn reset(&mut self) {
        for slot in self.frames.iter_mut() {
            *slot = None;
        }
        self.wasprocessed = false;
    }

    pub fn set_step(&mut self, step: u32) {
        self.step = step.max(1);
    }

    /// Most recent analyzed frame index at or before `frame_idx`, if any.
    pub fn last_processed_at_or_before(&self, frame_idx: u64) -> Option<u64> {
        let step = self.step.max(1) as u64;
        let mut candidate = if step <= 1 { frame_idx } else { (frame_idx / step) * step };
        loop {
            if (candidate as usize) < self.frames.len()
                && self.frames[candidate as usize].is_some()
            {
                return Some(candidate);
            }
            if candidate == 0 {
                return None;
            }
            candidate = candidate.saturating_sub(1);
        }
    }

    pub fn prediction_at(&self, frame_idx: u64) -> Option<&AIOutputs> {
        self.last_processed_at_or_before(frame_idx)
            .and_then(|i| self.frames.get(i as usize)?.as_ref())
    }

    pub fn record(&mut self, frame_idx: u64, aioutput: AIOutputs) {
        if let Some(slot) = self.frames.get_mut(frame_idx as usize) {
            *slot = Some(aioutput);
        }
    }

    pub fn processed_count(&self) -> usize {
        self.frames.iter().filter(|f| f.is_some()).count()
    }

    /// Highest analyzed frame index, or `None` if nothing has been analyzed yet.
    pub fn max_processed_frame(&self) -> Option<u64> {
        self.frames
            .iter()
            .rposition(|f| f.is_some())
            .map(|i| i as u64)
    }

    /// Fraction of *intended* frame-work that's done (i.e. processed
    /// analyzed-frames over total analyzed-frames according to `step`).
    /// Distinct from `Vec<PredVideo>::get_progress`, which counts whole videos.
    pub fn frame_progress(&self) -> f32 {
        let step = self.step.max(1) as u64;
        let target = (0..self.n_frames).step_by(step as usize).count();
        if target == 0 {
            return 0.0;
        }
        self.processed_count() as f32 / target as f32
    }
}

impl Pred for PredVideo {
    fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }
    fn is_processed(&self) -> bool {
        self.wasprocessed
    }
    fn predictions_json(&self) -> serde_json::Result<String> {
        // Video round-trips the whole struct (frames-as-array + probe metadata),
        // not just `aioutput` — that's the difference vs. PredImg / PredAudio.
        serde_json::to_string(self)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct XYXYc {
    pub xyxy: XYXY,
    pub label: String,
    pub extra_cls: Option<Vec<Prob>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct XYc {
    pub xy: XY,
    pub label: String,
}

impl XYXYc {
    pub fn new(xyxy: XYXY, label: String) -> Self {
        XYXYc {xyxy, label, extra_cls: None,}
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioProb {
    pub start: f32,
    pub end: f32,
    pub prediction: Prob,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(Vec<Prob>),
    Segmentation(Vec<SEGc>),
    AudioClassification(Vec<AudioProb>),
    Embed(Embedding),
}

impl AIOutputs {
    pub fn is_empty(&self) -> bool {
        match self {
            AIOutputs::ObjectDetection(bboxes) => bboxes.is_empty(),
            AIOutputs::Classification(probs) => probs.is_empty(),
            AIOutputs::Segmentation(segments) => segments.is_empty(),
            AIOutputs::AudioClassification(audio_probs) => audio_probs.is_empty(),
            AIOutputs::Embed(emb) => emb.values.is_empty(),
        }
    }

    pub fn from_file(input_path: impl AsRef<std::path::Path>) -> std::io::Result<AIOutputs> {
        let deserialized: AIOutputs = serde_json::from_reader(std::fs::File::open(input_path)?)?;
        Ok(deserialized)
    }

    /// `(class_id, label, prob)` for the single best prediction in this output.
    /// `None` for empty outputs or for audio classification.
    pub fn dominant_prob(&self) -> Option<(u32, &str, f32)> {
        match self {
            AIOutputs::Classification(probs) => probs
                .iter()
                .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap_or(std::cmp::Ordering::Equal))
                .map(|p| (p.class_id, p.label.as_str(), p.prob)),
            AIOutputs::ObjectDetection(bboxes) => bboxes
                .iter()
                .max_by(|a, b| {
                    a.xyxy.prob.partial_cmp(&b.xyxy.prob).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|b| (b.xyxy.class_id, b.label.as_str(), b.xyxy.prob)),
            AIOutputs::Segmentation(segs) => segs
                .iter()
                .max_by(|a, b| {
                    a.bbox
                        .xyxy
                        .prob
                        .partial_cmp(&b.bbox.xyxy.prob)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|s| (s.bbox.xyxy.class_id, s.bbox.label.as_str(), s.bbox.xyxy.prob)),
            AIOutputs::AudioClassification(_) => None,
            AIOutputs::Embed(_) => None,
        }
    }
}

impl Embedding {
    pub fn cosine(&self, other: &Embedding) -> f32 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let a = a.to_f32();
            let b = b.to_f32();
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom <= f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub geo_fence: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.45,
            nms_threshold: 0.4,
            geo_fence: "".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvailableModel {
    pub name: String,
    pub description: String,
    pub download_link: String,
}

/// A flat `[N]` embedding vector, L2-normalised.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Embedding {
    pub values: Vec<half::f16>,
    pub model: String,
}

impl Embedding {
    /// L2-normalises a flat embedding vector.
    pub fn from_raw(raw: &[f32], model: String) -> Self {
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        Self {
            values: raw.iter().map(|&v| half::f16::from_f32(v / norm)).collect(),
            model,
        }
    }
}

