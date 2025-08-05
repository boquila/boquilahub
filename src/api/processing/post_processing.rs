use std::collections::HashMap;

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

pub fn transform_logits_to_probs(prob_space: &mut ProbSpace) {
    let logits = &prob_space.probs; // Assuming probs field contains logits

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

    // Update the probs field with actual probabilities
    prob_space.probs = probabilities;
}

#[derive(Debug)]
pub struct SpeciesRecord {
    pub uuid: String,
    pub class: String,
    pub order: String,
    pub family: String,
    pub genus: String,
    pub species: String,
    pub common_name: String,
}

impl SpeciesRecord {
    pub fn new(line: &str) -> Result<SpeciesRecord, ()> {
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() < 7 {
            return Err(());
        }
        Ok(SpeciesRecord {
            uuid: parts[0].to_string(),
            class: parts[1].to_string(),
            order: parts[2].to_string(),
            family: parts[3].to_string(),
            genus: parts[4].to_string(),
            species: parts[5].to_string(),
            common_name: parts[6].to_string(),
        })
    }

    pub fn to_taxonomic_string(&self) -> String {
        format!(
            "{};{};{};{};{}",
            self.class, self.order, self.family, self.genus, self.species
        )
    }
}

pub fn get_uuid(line: &str) -> String {
    line.split(';').nth(0).unwrap().to_string()
}

pub fn get_class(line: &str) -> String {
    line.split(';').nth(1).unwrap().to_string()
}

pub fn get_order(line: &str) -> String {
    line.split(';').nth(2).unwrap().to_string()
}

pub fn get_family(line: &str) -> String {
    line.split(';').nth(3).unwrap().to_string()
}

pub fn get_genus(line: &str) -> String {
    line.split(';').nth(4).unwrap().to_string()
}

pub fn get_species(line: &str) -> String {
    line.split(';').nth(5).unwrap().to_string()
}

pub fn get_common_name(line: &str) -> String {
    line.split(';').nth(6).unwrap().to_string()
}

pub fn apply_geofence_filter(
    probs: &mut ProbSpace,
    geofence_data: &HashMap<String, Vec<String>>,
    target_country: &str,
) {
    let mut filtered_indices: Vec<usize> = Vec::new();

    for (i, class_name) in probs.classes.iter().enumerate() {
        if let Ok(record) = SpeciesRecord::new(class_name) {
            let taxonomic_string = record.to_taxonomic_string();

            // Check if this taxonomic string exists in geofence data
            if let Some(countries) = geofence_data.get(&taxonomic_string) {
                // Check if target country is in the allowed countries list
                if countries.contains(&target_country.to_string()) {
                    filtered_indices.push(i);
                }
            } else {
                // If not in geofence data, keep it (or you could choose to filter it out)
                filtered_indices.push(i);
            }
        }
    }

    // Update ProbSpace with only the filtered results
    let new_classes: Vec<String> = filtered_indices
        .iter()
        .map(|&i| probs.classes[i].clone())
        .collect();

    let new_probs: Vec<f32> = filtered_indices.iter().map(|&i| probs.probs[i]).collect();

    let new_classes_ids: Vec<u32> = filtered_indices
        .iter()
        .map(|&i| probs.classes_ids[i])
        .collect();

    probs.classes = new_classes;
    probs.probs = new_probs;
    probs.classes_ids = new_classes_ids;
}

pub fn apply_label_rollup(probs: &mut ProbSpace, confidence_threshold: f32) {
    let record_confidence_pairs: Vec<(SpeciesRecord, f32, u32)> = probs
        .classes
        .iter()
        .zip(probs.probs.iter())
        .zip(probs.classes_ids.iter())
        .filter_map(|((class_name, &confidence), &class_id)| {
            SpeciesRecord::new(class_name)
                .ok()
                .map(|record| (record, confidence, class_id))
        })
        .collect();

    // Find the highest confidence record to use as the base for rollup
    let (best_record, best_confidence, best_class_id) = record_confidence_pairs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(record, conf, id)| (record, *conf, *id))
        .unwrap();

    // Check if the best prediction already meets the threshold
    if best_confidence >= confidence_threshold {
        let name = if !best_record.common_name.is_empty() {
            if best_record.genus.trim().is_empty() || best_record.species.trim().is_empty() {
                best_record.common_name.clone()
            } else {
                format!(
                    "{} {} ({})",
                    best_record.genus.trim(),
                    best_record.species.trim(),
                    best_record.common_name
                )
            }
        } else {
            if best_record.genus.trim().is_empty() || best_record.species.trim().is_empty() {
                "Unknown species".to_string()
            } else {
                format!(
                    "{} {}",
                    best_record.genus.trim(),
                    best_record.species.trim()
                )
            }
        };

        // Return single high-confidence prediction
        probs.classes = vec![name];
        probs.probs = vec![best_confidence];
        probs.classes_ids = vec![best_class_id];
        return;
    }

    // Try rollup at each taxonomic level until threshold is met
    let levels = [
        ("genus", &best_record.genus),
        ("family", &best_record.family),
        ("order", &best_record.order),
        ("class", &best_record.class),
    ];

    for (level_name, level_value) in levels {
        if level_value.trim().is_empty() {
            continue;
        }

        let total_confidence: f32 = record_confidence_pairs
            .iter()
            .filter(|(record, _, _)| match level_name {
                "genus" => record.genus == *level_value,
                "family" => record.family == *level_value,
                "order" => record.order == *level_value,
                "class" => record.class == *level_value,
                _ => false,
            })
            .map(|(_, conf, _)| conf)
            .sum();

        if total_confidence >= confidence_threshold {
            // Format the rolled-up name based on taxonomic level
            let rolled_up_name = match level_name {
                "genus" => format!("{} sp.", level_value.trim()),
                "family" => format!("Family {}", level_value.trim()),
                "order" => format!("Order {}", level_value.trim()),
                "class" => format!("Class {}", level_value.trim()),
                _ => level_value.trim().to_string(),
            };

            // Return single rolled-up prediction
            probs.classes = vec![rolled_up_name];
            probs.probs = vec![total_confidence];
            probs.classes_ids = vec![best_class_id];
            return;
        }
    }

    // If no rollup meets threshold, return the best individual prediction
    let name = if !best_record.common_name.is_empty() {
        if best_record.genus.trim().is_empty() || best_record.species.trim().is_empty() {
            best_record.common_name.clone()
        } else {
            format!(
                "{} {} ({})",
                best_record.genus.trim(),
                best_record.species.trim(),
                best_record.common_name
            )
        }
    } else {
        if best_record.genus.trim().is_empty() || best_record.species.trim().is_empty() {
            "Unknown species".to_string()
        } else {
            format!(
                "{} {}",
                best_record.genus.trim(),
                best_record.species.trim()
            )
        }
    };

    probs.classes = vec![name];
    probs.probs = vec![best_confidence];
    probs.classes_ids = vec![best_class_id];
}
