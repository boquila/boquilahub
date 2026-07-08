use crate::api::abstractions::{BitMatrix, Prob, XYXY};
use bitvec::vec::BitVec;
use ndarray::{Array, Array2, ArrayBase, Dim, IxDyn, IxDynImpl, OwnedRepr};
use ort::session::SessionOutputs;
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessing {
    NMS,
    GeoFence,
    Rollup,
    Ensemble,
    Sigmoid,
    Softmax,
    BinaryClassification,
    None,
}

impl From<&str> for PostProcessing {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nms" => PostProcessing::NMS,
            "rollup" => PostProcessing::Rollup,
            "geofence" | "geo_fence" | "geo-fence" => PostProcessing::GeoFence,
            "ensemble" | "ensemble_classification" => PostProcessing::Ensemble,
            "sigmoid" => PostProcessing::Sigmoid,
            "softmax" => PostProcessing::Softmax,
            "binary" | "binary_classification" => PostProcessing::BinaryClassification,
            _ => PostProcessing::None, // default fallback
        }
    }
}

pub fn nms_indices(boxes: &[XYXY], iou_threshold: f32, per_class: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .prob
            .partial_cmp(&boxes[a].prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();

    while !indices.is_empty() {
        let current_idx = indices[0];
        keep.push(current_idx);

        let mut write = 0;
        for read in 1..indices.len() {
            let idx = indices[read];
            if (per_class && boxes[idx].class_id != boxes[current_idx].class_id)
                || boxes[idx].iou(&boxes[current_idx]) <= iou_threshold
            {
                indices[write] = idx;
                write += 1;
            }
        }
        indices.truncate(write);
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
    outputs: &SessionOutputs<'_>,
    output_name: &str,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    outputs[output_name]
        .try_extract_array::<f32>()
        .unwrap()
        .t()
        .into_owned()
}

pub fn process_class_output(
    conf: Option<f32>,
    classes: &[String],
    output: &Array<f32, IxDyn>,
) -> Vec<Prob> {
    let mut probs: Vec<Prob> = output
        .iter()
        .enumerate()
        .filter(|&(_, &score)| conf.map_or(true, |c| score >= c))
        .map(|(i, &score)| Prob::new(classes[i].clone(), score, i as u32))
        .collect();
    probs.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    probs
}

#[derive(Debug)]
pub struct SpeciesRecord<'a> {
    pub uuid: Cow<'a, str>,
    pub class: Cow<'a, str>,
    pub order: Option<String>,
    pub family: Option<String>,
    pub genus: Option<String>,
    pub species: Option<String>,
    pub common_name: Option<String>,
}

impl<'a> SpeciesRecord<'a> {
    pub fn new(line: &'a str) -> Result<SpeciesRecord<'a>, String> {
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
            uuid: Cow::from(uuid),
            class: Cow::from(class),
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
    probs: &mut Vec<Prob>,
    geofence_data: &HashMap<String, Vec<String>>,
    target_country: &str,
) {
    if target_country.is_empty() {
        return;
    }
    for p in probs.iter_mut() {
        let mut record = SpeciesRecord::new(&p.label).unwrap();
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
        p.label = record.get_line();
    }
}

pub fn apply_label_rollup(probs: &mut Vec<Prob>, confidence_threshold: f32) {
    let record_pairs: Vec<(SpeciesRecord, f32, u32)> = probs
        .iter()
        .filter_map(|p| {
            SpeciesRecord::new(&p.label)
                .ok()
                .map(|record| (record, p.prob, p.class_id))
        })
        .collect();

    let (best_record, best_confidence, best_class_id) = record_pairs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(record, conf, id)| (record, *conf, *id))
        .unwrap();

    if best_confidence >= confidence_threshold {
        let name = format_species_name(best_record);
        *probs = vec![Prob::new(name, best_confidence, best_class_id)];
        return;
    }

    let taxonomic_levels = ["species", "genus", "family", "order", "class"];
    let mut best_rollup: Option<Prob> = None;

    for level in taxonomic_levels {
        let mut level_totals: HashMap<String, f32> = HashMap::new();
        for (record, confidence, _) in &record_pairs {
            if let Some(taxon_name) = get_taxon_at_level(record, level) {
                *level_totals.entry(taxon_name).or_insert(0.0) += confidence;
            }
        }

        let Some((best_taxon, total_confidence)) = level_totals
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, c)| (t.clone(), *c))
        else {
            continue;
        };

        let class_id_in_group = record_pairs
            .iter()
            .filter(|(record, _, _)| {
                get_taxon_at_level(record, level).map_or(false, |t| t == best_taxon)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(_, _, id)| *id)
            .unwrap_or(best_class_id);

        let name = format_taxonomic_name(level, &best_taxon);

        if total_confidence >= confidence_threshold {
            *probs = vec![Prob::new(name, total_confidence, class_id_in_group)];
            return;
        }

        if best_rollup.as_ref().map_or(true, |r| total_confidence > r.prob) {
            best_rollup = Some(Prob::new(name, total_confidence, class_id_in_group));
        }
    }

    if let Some(rollup) = best_rollup {
        if rollup.prob > best_confidence {
            *probs = vec![rollup];
            return;
        }
    }

    let name = format_species_name(best_record);
    *probs = vec![Prob::new(name, best_confidence, best_class_id)];
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
        "class" => Some(record.class.to_string()),
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
                record.class.to_string()
            }
        }
    }
}
