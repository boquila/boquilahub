// The idea is to have the core funcionality that will alow us to do everything we need in the app
// but also, enough abstractions so we can experiment and build more complex tools in the future
#![allow(dead_code)]
use derive_new::new;
use serde::{Deserialize, Serialize};

/// Probabilities in the YOLO format
/// `classes` is a Vec with the names for each classification
/// `probs` is a Vec with the probabilities/confidence for each classification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProbSpace {
    pub classes: Vec<String>,
    pub probs: Vec<f32>,
    pub classes_ids: Vec<u32>,
}

impl ProbSpace {
    pub fn new(classes: Vec<String>, probs: Vec<f32>, classes_ids: Vec<u32>) -> Self {
        Self {
            classes,
            probs,
            classes_ids,
        }
    }

    pub fn highest_confidence(&self) -> String {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| self.classes[index].clone())
            .unwrap_or_else(|| String::from("no prediction"))
    }

    pub fn highest_confidence_full(&self) -> (String, f32, u32) {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, &prob)| {
                (
                    self.classes
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string()),
                    prob,
                    *self.classes_ids.get(index).unwrap_or(&0),
                )
            })
            .unwrap_or_else(|| ("no prediction".to_string(), 0.0, 0))
    }

    pub fn logits_to_probabilities(&mut self) {
        // Find the maximum logit for numerical stability
        let max_logit = self.probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(logit - max_logit) for each logit
        let exp_logits: Vec<f32> = self
            .probs
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();

        // Compute the sum of all exponentials
        let sum_exp: f32 = exp_logits.iter().sum();

        // Convert to probabilities by dividing each exp by the sum
        self.probs = exp_logits
            .iter()
            .map(|&exp_logit| exp_logit / sum_exp)
            .collect();
    }

    pub fn top_n(&self, n: u32) -> ProbSpace {
        let mut indices: Vec<usize> = (0..self.probs.len()).collect();
        indices.sort_by(|&a, &b| self.probs[b].partial_cmp(&self.probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(n as usize);

        ProbSpace::new(
            indices.iter().map(|&i| self.classes[i].clone()).collect(),
            indices.iter().map(|&i| self.probs[i]).collect(),
            indices.iter().map(|&i| self.classes_ids[i]).collect(),
        )
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BitMatrix {
    pub data: bitvec::vec::BitVec,
    pub width: usize,
    pub height: usize,
}

#[derive(Serialize, Deserialize, Clone, new)]
pub struct SEGc {
    pub mask: BitMatrix,
    pub bbox: XYXYc,
}

// Trait for all bounding boxes (that don't have a string)
pub trait BoundingBoxTrait: Copy {
    fn area(&self) -> f32;
    fn intersect(&self, other: &Self) -> f32;
    fn iou(&self, other: &Self) -> f32;
    fn get_coords(&self) -> (f32, f32, f32, f32);
    fn get_prob(&self) -> f32;
    fn get_class_id(&self) -> u32;
    fn check(&self) -> bool;
    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY;
    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn;
    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH;
    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn;
    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc;
    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc;
    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc;
    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc;
}

/// Bounding box in normalized XYXY format
/// # Fields
/// - `x1` and `y1` represent the top-left corner
/// - `x2` and `y2` represent the bottom-right  corner
#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYXYn {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u32,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, new)]
pub struct XYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u32,
}

/// Bounding box in normalized XYWH format
/// # Fields
/// - `x` and `y` represent the center
/// - `w` and `h` represent width and height
#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYWHn {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub prob: f32,
    pub class_id: u32,
}

#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYWH {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub prob: f32,
    pub class_id: u32,
}

fn intersect_xyxys(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32, x4: f32, y4: f32) -> f32 {
    let x_left = x1.max(x3);
    let y_top = y1.max(y3);
    let x_right = x2.min(x4);
    let y_bottom = y2.min(y4);

    (x_right - x_left) * (y_bottom - y_top)
}

fn intersect_xywhs(x1: f32, y1: f32, w1: f32, h1: f32, x2: f32, y2: f32, w2: f32, h2: f32) -> f32 {
    let x_left = x1.max(x2);
    let y_top = y1.max(y2);
    let x_right = (x1 + w1).min(x2 + w2);
    let y_bottom = (y1 + h1).min(y2 + h2);

    (x_right - x_left) * (y_bottom - y_top)
}

fn iou<T: BoundingBoxTrait>(a: &T, b: &T) -> f32 {
    let intersection = a.intersect(b);
    let union = a.area() + b.area() - intersection;
    intersection / union
}

impl BoundingBoxTrait for XYXYn {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXYn) -> f32 {
        intersect_xyxys(
            self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2,
        )
    }

    fn iou(&self, other: &XYXYn) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u32 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x1 >= 0.0
            && self.x1 <= 1.0
            && self.y1 >= 0.0
            && self.y1 <= 1.0
            && self.x2 >= 0.0
            && self.x2 <= 1.0
            && self.y2 >= 0.0
            && self.y2 <= 1.0
            && self.x2 >= self.x1
            && self.y2 >= self.y1
            && self.prob >= 0.0
            && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    fn to_xyxyn(&self, _w: Option<f32>, _h: Option<f32>) -> XYXYn {
        return *self;
    }

    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY {
        let w = w.unwrap();
        let h = h.unwrap();
        XYXY::new(
            self.x1 * w,
            self.y1 * h,
            self.x2 * w,
            self.x2 * h,
            self.prob,
            self.class_id,
        )
    }

    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH {
        let temp = self.to_xyxy(w, h);
        return temp.to_xywh(w, h);
    }

    fn to_xywhn(&self, _w: Option<f32>, _h: Option<f32>) -> XYWHn {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        XYWHn::new(x, y, w, h, self.prob, self.class_id)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYXY {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXY) -> f32 {
        intersect_xyxys(
            self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2,
        )
    }

    fn iou(&self, other: &XYXY) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u32 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x2 >= self.x1 && self.y2 >= self.y1 && self.prob >= 0.0 && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn {
        let w = w.unwrap();
        let h = h.unwrap();
        return XYXYn::new(
            self.x1 / w,
            self.y1 / h,
            self.x2 / w,
            self.x2 / h,
            self.prob,
            self.class_id,
        );
    }

    fn to_xyxy(&self, _w: Option<f32>, _h: Option<f32>) -> XYXY {
        return *self;
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        return XYWH::new(x, y, w, h, self.prob, self.class_id);
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let temp = self.to_xyxyn(w, h);
        return temp.to_xywhn(w, h);
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYWHn {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersect(&self, other: &XYWHn) -> f32 {
        intersect_xywhs(
            self.x, self.y, self.w, self.h, other.x, other.y, other.w, other.h,
        )
    }

    fn iou(&self, other: &XYWHn) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u32 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x >= 0.0
            && self.x <= 1.0
            && self.y >= 0.0
            && self.y <= 1.0
            && self.w >= 0.0
            && self.w <= 1.0
            && self.h >= 0.0
            && self.h <= 1.0
            && (self.x + self.w) <= 1.0
            && (self.y + self.h) <= 1.0
            && self.prob >= 0.0
            && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn to_xyxyn(&self, _w: Option<f32>, _h: Option<f32>) -> XYXYn {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        return XYXYn::new(x1, y1, x2, y2, self.prob, self.class_id);
    }

    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY {
        let temp = self.to_xyxyn(w, h);
        return temp.to_xyxy(w, h);
    }

    fn to_xywhn(&self, _w: Option<f32>, _h: Option<f32>) -> XYWHn {
        return *self;
    }

    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH {
        let w = w.unwrap();
        let h = h.unwrap();

        return XYWH {
            x: self.x * w,
            y: self.y * h,
            w: self.w * w,
            h: self.h * h,
            class_id: self.class_id,
            prob: self.prob,
        };
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYWH {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersect(&self, other: &XYWH) -> f32 {
        intersect_xywhs(
            self.x, self.y, self.w, self.h, other.x, other.y, other.w, other.h,
        )
    }

    fn iou(&self, other: &XYWH) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u32 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.w >= 0.0 && self.h >= 0.0 && self.prob >= 0.0 && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn {
        let temp = self.to_xyxy(w, h);
        return temp.to_xyxyn(w, h);
    }

    fn to_xyxy(&self, _w: Option<f32>, _h: Option<f32>) -> XYXY {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        return XYXY::new(x1, y1, x2, y2, self.prob, self.class_id);
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let w = w.unwrap();
        let h = h.unwrap();

        return XYWHn {
            x: self.x / w,
            y: self.y / h,
            w: self.w / w,
            h: self.h / h,
            class_id: self.class_id,
            prob: self.prob,
        };
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        return *self;
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

// AI model for Image Processing
#[derive(Deserialize, Clone, Debug, new)]
pub struct AI {
    pub task: String,
    #[serde(default)]
    pub architecture: Option<String>, // yolo, efficientnet, whatever else
    pub post_processing: Vec<String>,
    pub classes: Vec<String>,
    #[serde(skip)]
    pub name: String,
}

impl AI {
    pub fn get_path(&self) -> String {
        return format!("models/{}.bq", self.name);
    }
}

#[derive(new, Clone)]
pub struct PredImg {
    pub file_path: std::path::PathBuf,
    pub aioutput: Option<AIOutputs>,
    pub wasprocessed: bool,
}

impl PredImg {
    // Simple constructor: only file_path is provided
    pub fn new_simple(file_path: std::path::PathBuf) -> Self {
        let aioutput = match super::import::read_predictions_from_file(&file_path) {
            Ok(predictions) => Some(predictions),
            Err(_) => None, // If file doesn't exist or can't be read, just use None
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct XYXYc {
    pub xyxy: XYXY,
    pub label: String,
    pub extra_cls: Option<ProbSpace>,
}

impl XYXYc {
    pub fn new(xyxy: XYXY, label: String) -> Self {
        XYXYc {
            xyxy,
            label,
            extra_cls: None,
        }
    }
}

#[derive(Serialize, Deserialize, new)]
pub struct XYXYnc {
    pub xyxyn: XYXYn,
    pub label: String,
}

#[derive(Serialize, Deserialize, new)]
pub struct XYWHc {
    pub xywh: XYWH,
    pub label: String,
}

#[derive(Serialize, Deserialize, new)]
pub struct XYWHnc {
    pub xywhn: XYWHn,
    pub label: String,
}

pub struct Dependency {
    version: f32,
    name: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGc>),
}

impl AIOutputs {
    pub fn is_empty(&self) -> bool {
        match self {
            AIOutputs::ObjectDetection(bboxes) => bboxes.is_empty(),
            AIOutputs::Classification(prob_space) => prob_space.classes.is_empty(),
            AIOutputs::Segmentation(segments) => segments.is_empty(),
        }
    }
}

#[derive(Clone,new)]
pub struct ModelConfig {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub geo_fence: String
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

impl ModelConfig{
    pub fn default2() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_threshold: 0.0,
            geo_fence: "".to_owned(),
        }
    }
}