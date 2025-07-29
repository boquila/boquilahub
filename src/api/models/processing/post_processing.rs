use bitvec::vec::BitVec;
use ndarray::{Array, Array2, ArrayBase, Dim, IxDyn, IxDynImpl, OwnedRepr};
use ort::session::SessionOutputs;

use crate::api::abstractions::{BitMatrix, BoundingBoxTrait, ProbSpace, XYXY};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessing {
    NMS,
    GeoFence,
    Rollup,
    Ensemble,
    None,
}

impl From<&str> for PostProcessing {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nms" => PostProcessing::NMS,
            "rollup" => PostProcessing::Rollup,
            "geofence" | "geo_fence" | "geo-fence" => PostProcessing::GeoFence,
            "ensemble" | "ensemble_classification" => PostProcessing::Ensemble,
            _ => PostProcessing::None, // default fallback
        }
    }
}

pub fn nms_indices<T: BoundingBoxTrait>(boxes: &[T], iou_threshold: f32) -> Vec<usize> {
    // Create indices and sort them by probability (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .get_prob()
            .partial_cmp(&boxes[a].get_prob())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();

    while !indices.is_empty() {
        // Keep the highest scoring box
        let current_idx = indices[0];
        keep.push(current_idx);

        // Filter remaining indices
        indices = indices
            .into_iter()
            .skip(1)
            .filter(|&idx| {
                boxes[idx].get_class_id() != boxes[current_idx].get_class_id()
                    || boxes[idx].iou(&boxes[current_idx]) <= iou_threshold
            })
            .collect();
    }

    keep
}

pub fn process_mask(
    mask: Array2<f32>,
    bbox: &XYXY,
    img_width: u32,
    img_height: u32,
    mask_height: u32,
    mask_width: u32,
) -> BitMatrix {
    // Calculate bbox coordinates in mask space
    let x1 = (bbox.x1 / img_width as f32 * mask_width as f32).round() as usize;
    let y1 = (bbox.y1 / img_height as f32 * mask_height as f32).round() as usize;
    let x2 = (bbox.x2 / img_width as f32 * mask_width as f32).round() as usize;
    let y2 = (bbox.y2 / img_height as f32 * mask_height as f32).round() as usize;

    // Clamp coordinates to mask bounds to prevent out-of-bounds access
    let x1 = x1.min(mask_width as usize);
    let y1 = y1.min(mask_height as usize);
    let x2 = x2.min(mask_width as usize);
    let y2 = y2.min(mask_height as usize);

    // Ensure we have valid ranges (x2 > x1, y2 > y1)
    let x2 = x2.max(x1);
    let y2 = y2.max(y1);

    let width = x2 - x1;
    let height = y2 - y1;

    // Create a single BitVec to hold all the data
    let mut data = BitVec::new();

    // Fill the BitVec row by row
    for y in y1..y2 {
        for x in x1..x2 {
            let mask_value = mask[[y, x]];
            data.push(mask_value > 0.0);
        }
    }

    BitMatrix {
        data,
        width,
        height,
    }
}

pub fn extract_output(
    outputs: &SessionOutputs<'_, '_>,
    b: &'static str,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    return outputs[b]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();
}

pub fn process_class_output(
    conf: f32,
    classes: &Vec<String>,
    output: &Array<f32, IxDyn>,
) -> ProbSpace {
    let mut indexed_scores: Vec<(usize, f32)> = output
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= conf)
        .map(|(i, &score)| (i, score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let probs: Vec<f32> = indexed_scores.iter().map(|(_, prob)| *prob).collect();
    let classes_ids: Vec<u32> = indexed_scores.iter().map(|(idx, _)| *idx as u32).collect();
    let classes: Vec<String> = classes_ids
        .iter()
        .map(|&idx| classes[idx as usize].clone())
        .collect();

    return ProbSpace::new(classes, probs, classes_ids);
}

pub fn process_class_output_logits(
    conf: f32,
    classes: &Vec<String>,
    output: &Array<f32, IxDyn>,
) -> ProbSpace {
    // First, convert logits to probabilities using softmax
    let logits: Vec<f32> = output.iter().cloned().collect();

    // Find max logit for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute softmax probabilities
    let exp_logits: Vec<f32> = logits
        .iter()
        .map(|&logit| (logit - max_logit).exp())
        .collect();

    let sum_exp: f32 = exp_logits.iter().sum();

    let probabilities: Vec<f32> = exp_logits
        .iter()
        .map(|&exp_logit| exp_logit / sum_exp)
        .collect();

    // Now filter by confidence threshold on actual probabilities
    let mut indexed_scores: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .filter(|(_, &prob)| prob >= conf)
        .map(|(i, &prob)| (i, prob))
        .collect();

    // Sort by probability (highest first)
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let probs: Vec<f32> = indexed_scores.iter().map(|(_, prob)| *prob).collect();
    let classes_ids: Vec<u32> = indexed_scores.iter().map(|(idx, _)| *idx as u32).collect();
    let filtered_classes: Vec<String> = classes_ids
        .iter()
        .map(|&idx| classes[idx as usize].clone())
        .collect();

    ProbSpace::new(filtered_classes, probs, classes_ids)
}
