use derive_new::new;
use serde::{Deserialize, Serialize};

/// Probabilities
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

    pub fn logits_to_probs(&mut self) {
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
        indices.sort_by(|&a, &b| {
            self.probs[b]
                .partial_cmp(&self.probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(n as usize);

        ProbSpace::new(
            indices.iter().map(|&i| self.classes[i].clone()).collect(),
            indices.iter().map(|&i| self.probs[i]).collect(),
            indices.iter().map(|&i| self.classes_ids[i]).collect(),
        )
    }

    pub fn filter(&self, conf: f32) -> Self {
        let mut filtered = ProbSpace {
            classes: Vec::new(),
            probs: Vec::new(),
            classes_ids: Vec::new(),
        };

        for (class, (prob, class_id)) in self
            .classes
            .iter()
            .zip(self.probs.iter().zip(self.classes_ids.iter()))
        {
            if *prob >= conf {
                filtered.classes.push(class.clone());
                filtered.probs.push(*prob);
                filtered.classes_ids.push(*class_id);
            }
        }

        filtered
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

fn iou<T: BoundingBoxTrait>(a: &T, b: &T) -> f32 {
    let intersection = a.intersect(b);
    let union = a.area() + b.area() - intersection;
    intersection / union
}

impl BoundingBoxTrait for XYXY {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXY) -> f32 {
        let x_left = self.x1.max(other.x1);
        let y_top = self.y1.max(other.y1);
        let x_right = self.x2.min(other.x2);
        let y_bottom = self.y2.min(other.y2);

        (x_right - x_left) * (y_bottom - y_top)
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

#[derive(Clone, new)]
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

impl ModelConfig {
    pub fn default2() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_threshold: 0.0,
            geo_fence: "".to_owned(),
        }
    }
}
