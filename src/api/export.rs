#![allow(dead_code)]
use super::abstractions::PredImg;
use crate::api::abstractions::AIOutputs;
use crate::api::utils::create_predictions_file_path;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

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
        AIOutputs::Segmentation(segments) => {
            get_most_frequent_label(segments, |seg| &seg.bbox.label)
        }
        AIOutputs::Classification(prob_space) => prob_space.highest_confidence(),
    }
}

pub async fn copy_to_folder(pred_imgs: &[PredImg], output_path: &str) {
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

impl PredImg {
    pub fn save(&self) {
        let img_data = image::DynamicImage::ImageRgba8(self.draw()).to_rgb8();
        let filename = prepare_export_img(&self.file_path);
        img_data.save(&filename).unwrap();
    }
    
    // For file 'img.jpg', creates a file 'img_predictions.json' that contains the AI outputs
    pub async fn write_pred_img_to_file(&self) -> io::Result<()> {
        let output_path = create_predictions_file_path(&self.file_path)?;
        let mut file = File::create(&output_path)?;
        let json_string = serde_json::to_string(&self.aioutput)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }
}

pub fn prepare_export_img(path: &PathBuf) -> String {
    std::fs::create_dir_all("export").expect("Failed to create export directory");
    return format!(
        "export/exported_{}.jpg",
        std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
    );
}
