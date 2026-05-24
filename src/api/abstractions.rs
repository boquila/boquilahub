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

#[derive(Clone)]
pub struct PredImg {
    pub file_path: std::path::PathBuf,
    pub aioutput: Option<AIOutputs>,
    pub wasprocessed: bool,
}

impl PredImg {
    pub fn new_simple(file_path: std::path::PathBuf) -> Self {
        let aioutput = match Self::create_predictions_file_path(&file_path) {
            Ok(path) if path.exists() => AIOutputs::from_file(path).ok(),
            _ => None,
        };
        let wasprocessed = aioutput.is_some();

        PredImg {
            file_path,
            aioutput,
            wasprocessed,
        }
    }

    pub fn reset(&mut self) {
        self.wasprocessed = false;
    }

    /// Creates the predictions file path based on the input file path
    /// For file 'img.jpg', creates path 'img_predictions.json'
    fn create_predictions_file_path(input_path: impl AsRef<std::path::Path>) -> std::io::Result<std::path::PathBuf> {
        let input_path = input_path.as_ref();
        let file_stem = input_path
            .file_stem()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid input path"))?
            .to_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Non-UTF-8 file path"))?;
        Ok(input_path.with_file_name(format!("{}_predictions.json", file_stem)))
    }

    pub fn predictions_file_path(&self) -> std::io::Result<std::path::PathBuf> {
        Self::create_predictions_file_path(&self.file_path)
    }
}

pub trait PredImgSugar {
    fn count_processed_images(&self) -> usize;
    fn get_progress(&self) -> f32;
}

impl PredImgSugar for Vec<PredImg> {
    fn count_processed_images(&self) -> usize {
        self.iter().filter(|img| img.wasprocessed).count()
    }

    fn get_progress(&self) -> f32 {
        self.count_processed_images() as f32 / self.len() as f32
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
    pub fn new(file_path: std::path::PathBuf, width: u32, height: u32, fps: f64, n_frames: u64) -> Self {
        Self {
            file_path,
            width,
            height,
            fps,
            n_frames,
            step: 1,
            frames: vec![None; n_frames as usize],
            wasprocessed: false,
        }
    }

    /// If a sidecar `_predictions.json` already exists for this video, load it
    /// and return a fully-populated `PredVideo`. Otherwise build a fresh one
    /// using the probed `width` / `height` / `fps` / `n_frames`.
    pub fn new_simple(
        file_path: std::path::PathBuf,
        width: u32,
        height: u32,
        fps: f64,
        n_frames: u64,
    ) -> Self {
        if let Ok(path) = Self::create_predictions_file_path(&file_path) {
            if path.exists() {
                if let Ok(file) = std::fs::File::open(&path) {
                    if let Ok(mut cached) = serde_json::from_reader::<_, PredVideo>(file) {
                        // Always trust the on-disk path the user just opened in case
                        // the file moved since the predictions were saved.
                        cached.file_path = file_path;
                        return cached;
                    }
                }
            }
        }
        Self::new(file_path, width, height, fps, n_frames)
    }

    fn create_predictions_file_path(
        input_path: &std::path::Path,
    ) -> std::io::Result<std::path::PathBuf> {
        let stem = input_path
            .file_stem()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid input path"))?
            .to_str()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Non-UTF-8 file path"))?;
        Ok(input_path.with_file_name(format!("{}_predictions.json", stem)))
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
    /// The "sticky" prediction the user sees on a non-analyzed frame.
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

    /// Fraction of *intended* work that's done (i.e. processed analyzed-frames over total
    /// analyzed-frames according to `step`). Returns 0 when nothing should be processed.
    pub fn get_progress(&self) -> f32 {
        let step = self.step.max(1) as u64;
        let target = (0..self.n_frames).step_by(step as usize).count();
        if target == 0 {
            return 0.0;
        }
        self.processed_count() as f32 / target as f32
    }

    /// For `image_path/myvideo.mp4`, returns `image_path/myvideo_predictions.json`.
    pub fn predictions_file_path(&self) -> std::io::Result<std::path::PathBuf> {
        Self::create_predictions_file_path(&self.file_path)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct XYXYc {
    pub xyxy: XYXY,
    pub label: String,
    pub extra_cls: Option<Vec<Prob>>,
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
}

impl AIOutputs {
    pub fn is_empty(&self) -> bool {
        match self {
            AIOutputs::ObjectDetection(bboxes) => bboxes.is_empty(),
            AIOutputs::Classification(probs) => probs.is_empty(),
            AIOutputs::Segmentation(segments) => segments.is_empty(),
            AIOutputs::AudioClassification(audio_probs) => audio_probs.is_empty()
        }
    }

    pub fn from_file(input_path: impl AsRef<std::path::Path>) -> std::io::Result<AIOutputs> {
        let deserialized: AIOutputs = serde_json::from_reader(std::fs::File::open(input_path)?)?;
        Ok(deserialized)
    }

    /// `(class_id, label, prob)` for the single best prediction in this output —
    /// used to colour the video timeline strip. `None` for empty outputs or for
    /// audio classification (which doesn't apply to video).
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