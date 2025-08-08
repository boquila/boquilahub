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
    b: &str,
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
    pub order: Option<String>,
    pub family: Option<String>,
    pub genus: Option<String>,
    pub species: Option<String>,
    pub common_name: Option<String>,
}

impl SpeciesRecord {
    pub fn new(line: &str) -> Result<SpeciesRecord, String> {
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() < 7 {
            return Err("Insufficient number of fields".to_string());
        }

        let uuid = parts[0].trim();
        let class = parts[1].trim();

        let to_option = |s: &str| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        Ok(SpeciesRecord {
            uuid: uuid.to_string(),
            class: class.to_string(),
            order: to_option(parts[2]),
            family: to_option(parts[3]),
            genus: to_option(parts[4]),
            species: to_option(parts[5]),
            common_name: to_option(parts[6]),
        })
    }

    /// Reconstructs the original semicolon-delimited line used in new()
    pub fn get_line(&self) -> String {
        format!(
            "{};{};{};{};{};{};{}",
            self.uuid,
            self.class,
            self.order.as_ref().unwrap_or(&String::new()),
            self.family.as_ref().unwrap_or(&String::new()),
            self.genus.as_ref().unwrap_or(&String::new()),
            self.species.as_ref().unwrap_or(&String::new()),
            self.common_name.as_ref().unwrap_or(&String::new())
        )
    }

    /// Rolls up taxonomic hierarchy by removing the most specific available classification
    /// Priority: species -> genus -> family -> order
    pub fn roll_up(&mut self) -> Result<(), ()> {
        if self.species.is_some() {
            self.species = None;
            self.common_name = None;
            Ok(())
        } else if self.genus.is_some() {
            self.genus = None;
            self.common_name = None;
            Ok(())
        } else if self.family.is_some() {
            self.family = None;
            self.common_name = None;
            Ok(())
        } else if self.order.is_some() {
            self.order = None;
            self.common_name = None;
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn to_taxonomic_string(&self) -> String {
        format!(
            "{};{};{};{};{}",
            self.class,
            self.order.as_ref().unwrap_or(&String::new()),
            self.family.as_ref().unwrap_or(&String::new()),
            self.genus.as_ref().unwrap_or(&String::new()),
            self.species.as_ref().unwrap_or(&String::new())
        )
    }
}

pub fn apply_geofence_filter(
    probs: &mut ProbSpace,
    geofence_data: &HashMap<String, Vec<String>>,
    target_country: &str,
) {
    if !target_country.is_empty() {
        probs.classes.iter_mut().for_each(|class| {
            let mut record = SpeciesRecord::new(class).unwrap();
            loop {
                let taxonomic_string = record.to_taxonomic_string();
                if let Some(countries) = geofence_data.get(&taxonomic_string) {
                    if countries.contains(&target_country.to_string()) {
                        break;
                    }
                }
                if record.roll_up().is_err() {
                    break;
                }
            }
            *class = record.get_line();
        });
    }
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

    // Find the highest confidence record (for fallback)
    let (best_record, best_confidence, best_class_id) = record_confidence_pairs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(record, conf, id)| (record, *conf, *id))
        .unwrap();

    // Check if the best prediction already meets the threshold
    if best_confidence >= confidence_threshold {
        let name = format_species_name(&best_record);
        probs.classes = vec![name];
        probs.probs = vec![best_confidence];
        probs.classes_ids = vec![best_class_id];
        return;
    }

    // Build consensus at each taxonomic level and track the best overall
    let taxonomic_levels = ["species", "genus", "family", "order", "class"];
    let mut best_rollup: Option<(String, f32, u32)> = None;

    for level in taxonomic_levels {
        let mut level_totals: HashMap<String, (f32, Vec<u32>)> = HashMap::new();

        // Aggregate confidence by taxonomic group at this level
        for (record, confidence, class_id) in &record_confidence_pairs {
            if let Some(taxon_name) = get_taxon_at_level(record, level) {
                let entry = level_totals.entry(taxon_name).or_insert((0.0, Vec::new()));
                entry.0 += confidence;
                entry.1.push(*class_id);
            }
        }

        // Find the highest confidence group at this level
        if let Some((best_taxon, (total_confidence, _))) = level_totals
            .iter()
            .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
        {
            // If this meets threshold, return immediately
            if *total_confidence >= confidence_threshold {
                let rolled_up_name = format_taxonomic_name(level, best_taxon);

                // Use the class_id from the highest individual confidence within this group
                let best_class_id_in_group = record_confidence_pairs
                    .iter()
                    .filter(|(record, _, _)| {
                        get_taxon_at_level(record, level)
                            .map_or(false, |taxon| &taxon == best_taxon)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(_, _, id)| *id)
                    .unwrap_or(best_class_id);

                probs.classes = vec![rolled_up_name];
                probs.probs = vec![*total_confidence];
                probs.classes_ids = vec![best_class_id_in_group];
                return;
            }

            // Track the best rollup even if it doesn't meet threshold
            let rolled_up_name = format_taxonomic_name(level, best_taxon);
            let best_class_id_in_group = record_confidence_pairs
                .iter()
                .filter(|(record, _, _)| {
                    get_taxon_at_level(record, level).map_or(false, |taxon| &taxon == best_taxon)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(_, _, id)| *id)
                .unwrap_or(best_class_id);

            // Update best_rollup if this is better
            if best_rollup
                .as_ref()
                .map_or(true, |(_, conf, _)| total_confidence > conf)
            {
                best_rollup = Some((rolled_up_name, *total_confidence, best_class_id_in_group));
            }
        }
    }

    // Return the best rollup found, or the best individual prediction if no rollup is better
    if let Some((name, confidence, class_id)) = best_rollup {
        if confidence > best_confidence {
            probs.classes = vec![name];
            probs.probs = vec![confidence];
            probs.classes_ids = vec![class_id];
            return;
        }
    }

    // Fallback to best individual prediction
    let name = format_species_name(&best_record);
    probs.classes = vec![name];
    probs.probs = vec![best_confidence];
    probs.classes_ids = vec![best_class_id];
}

fn get_taxon_at_level(record: &SpeciesRecord, level: &str) -> Option<String> {
    match level {
        "species" => {
            // Only return species if we have both genus and species
            if let (Some(genus), Some(species)) = (&record.genus, &record.species) {
                Some(format!("{} {}", genus, species))
            } else {
                None
            }
        }
        "genus" => record.genus.clone(),
        "family" => record.family.clone(),
        "order" => record.order.clone(),
        "class" => Some(record.class.clone()),
        _ => None,
    }
}

fn format_taxonomic_name(level: &str, taxon: &str) -> String {
    match level {
        "species" => taxon.to_string(), // Already formatted as "Genus species"
        "genus" => format!("{} sp.", taxon),
        "family" => format!("Family {}", taxon),
        "order" => format!("Order {}", taxon),
        "class" => format!("Class {}", taxon),
        _ => taxon.to_string(),
    }
}

fn format_species_name(record: &SpeciesRecord) -> String {
    match (&record.genus, &record.species, &record.common_name) {
        (Some(genus), Some(species), Some(common_name)) if !common_name.is_empty() => {
            format!("{} {} ({})", genus, species, common_name)
        }
        (Some(genus), Some(species), _) => {
            format!("{} {}", genus, species)
        }
        (_, _, Some(common_name)) if !common_name.is_empty() => common_name.clone(),
        _ => {
            // Fallback: try more general taxonomy levels in order
            if let Some(genus) = &record.genus {
                genus.clone()
            } else if let Some(family) = &record.family {
                family.clone()
            } else if let Some(order) = &record.order {
                order.clone()
            } else {
                record.class.clone()
            }
        }
    }
}
