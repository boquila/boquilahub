#![allow(dead_code)]
use crate::api::models::processing::inference::AIOutputs;

use super::abstractions::PredImg;
use super::abstractions::XYXYc;
use super::abstractions::XYXY;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

pub async fn read_predictions_from_file(input_path: &str) -> io::Result<Vec<XYXYc>> {
    // Create expected filename based on input filepath
    let input_path = Path::new(input_path);
    let file_stem = input_path
        .file_stem()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid input path"))?;

    let parent = input_path.parent().unwrap_or(Path::new(""));
    let prediction_path = parent.join(format!("{}_predictions.txt", file_stem.to_string_lossy()));

    // Check if file exists
    if !prediction_path.exists() {
        // println!("No prediction file found at: {:?}", prediction_path);
        return Ok(Vec::new());
    }

    // Read and parse file
    let mut bboxes = Vec::new();
    let file = File::open(&prediction_path)?;
    let reader = io::BufReader::new(file);

    for line_result in reader.lines() {
        let line = line_result?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != 7 {
            // println!("Warning: Skipping invalid line format: {}", line);
            continue;
        }

        match (
            parts[1].parse::<f32>(), // x1
            parts[2].parse::<f32>(), // y1
            parts[3].parse::<f32>(), // x2
            parts[4].parse::<f32>(), // y2
            parts[5].parse::<u16>(), // class_id
            parts[6].parse::<f32>(), // confidence
        ) {
            (Ok(x1), Ok(y1), Ok(x2), Ok(y2), Ok(class_id), Ok(confidence)) => {
                bboxes.push(XYXYc {
                    bbox: XYXY::new(x1, y1, x2, y2, confidence, class_id),
                    label: parts[0].to_string(),
                });
            }
            _ => {
                // println!("Warning: Error parsing line: {}", line);
                continue;
            }
        }
    }

    // println!("Successfully read {} predictions from: {:?}", bboxes.len(), prediction_path);
    Ok(bboxes)
}

pub async fn write_pred_img_to_file(pred_img: &PredImg) -> io::Result<()> {
    // Create output filename based on input filepath
    let input_path = Path::new(&pred_img.file_path);
    let file_stem = input_path
        .file_stem()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid input path"))?;

    let parent = input_path.parent().unwrap_or(Path::new(""));
    let output_path = parent.join(format!("{}_predictions.txt", file_stem.to_string_lossy()));

    // Open file for writing - using tokio's async File
    let mut file = File::create(&output_path)?;

    // Create the full content string
    let content = match &pred_img.aioutput.as_ref().unwrap() {
        AIOutputs::ObjectDetection(bboxes) => bboxes
            .iter()
            .map(|bbox| {
                format!(
                    "{} {} {} {} {} {} {}\n",
                    bbox.label,
                    bbox.bbox.x1,
                    bbox.bbox.y1,
                    bbox.bbox.x2,
                    bbox.bbox.y2,
                    bbox.bbox.class_id,
                    bbox.bbox.prob
                )
            })
            .collect::<String>(),
        AIOutputs::Classification(prob_space) => prob_space
            .classes
            .iter()
            .zip(prob_space.probs.iter())
            .map(|(class, prob)| format!("{} {}\n", class, prob))
            .collect::<String>(),
        AIOutputs::Segmentation(segments) => segments
            .iter()
            .map(|seg| {
                let x_coords = seg
                    .seg
                    .x
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let y_coords = seg
                    .seg
                    .y
                    .iter()
                    .map(|y| y.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "{} [{}] [{}] {} {}\n",
                    seg.label, x_coords, y_coords, seg.seg.class_id, seg.seg.prob
                )
            })
            .collect::<String>(),
    };
    file.write_all(content.as_bytes())?;
    Ok(())
}

// Count processed images from a list of PredImg structs
pub fn count_processed_images(images: &Vec<PredImg>) -> usize {
    images.iter().filter(|img| img.wasprocessed).count()
}

// Get the most frequent label from a list of bounding boxes
fn get_most_frequent_label<T>(items: &[T], get_label: impl Fn(&T) -> &String) -> String {
    if items.is_empty() {
        return String::from("no predictions");
    }

    let mut label_counts: HashMap<&String, usize> = HashMap::new();
    for item in items {
        *label_counts.entry(get_label(item)).or_insert(0) += 1;
    }

    label_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| (*label).clone())
        .unwrap()
}

fn get_main_label(output: &AIOutputs) -> String {
    match output {
        AIOutputs::ObjectDetection(bboxes) => get_most_frequent_label(bboxes, |bbox| &bbox.label),
        AIOutputs::Segmentation(segments) => get_most_frequent_label(segments, |seg| &seg.label),
        AIOutputs::Classification(_prob_space) => {
            todo!()
        }
    }
}

pub async fn copy_to_folder(pred_imgs: &Vec<PredImg>, output_path: &str) {
    for pred_img in pred_imgs {
        let image_file_path = &pred_img.file_path;
        if std::path::Path::new(image_file_path).exists() {
            let main_label = get_main_label(&pred_img.aioutput.as_ref().unwrap());

            let folder_path = format!("{}/{}", output_path, main_label);

            // Create directory if it doesn't exist
            if !std::path::Path::new(&folder_path).exists() {
                std::fs::create_dir_all(&folder_path).unwrap();
            }

            // Extract the image name from path
            let image_name = std::path::Path::new(image_file_path)
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default();

            let new_image_path = format!("{}/{}", folder_path, image_name);

            // Read the original file and write to the new location
            let image_data = tokio::fs::read(image_file_path).await.unwrap();
            tokio::fs::write(&new_image_path, image_data).await.unwrap();
        }
    }
}
