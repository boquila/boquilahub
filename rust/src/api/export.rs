#![allow(dead_code)]
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use csv::Writer;
use csv::WriterBuilder;
use std::collections::HashSet;
use super::abstractions::ImgPred;
use super::abstractions::BBox;

pub fn write_csv(pred_imgs: Vec<ImgPred>, output_path: &str) -> io::Result<()> {
    let mut wtr = Writer::from_path(output_path)?;
    wtr.write_record(&["File Path", "X1", "Y1", "X2", "Y2", "Label", "Confidence"])?;

    for pred_img in pred_imgs {
        for bbox in pred_img.list_bbox {
            wtr.write_record(&[
                pred_img.file_path.clone(),
                bbox.x1.to_string(),
                bbox.y1.to_string(),
                bbox.x2.to_string(),
                bbox.y2.to_string(),
                bbox.class_id.to_string(),
                bbox.confidence.to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}

pub fn write_csv2(pred_imgs: Vec<ImgPred>, output_path: &str) -> io::Result<()> {
    
    let file = File::create(output_path)?;
    let mut wtr = WriterBuilder::new().has_headers(true).from_writer(file);

    wtr.write_record(&["File Path", "n", "observaciones"])?;

    // Iterate through each predicted image.
    for pred_img in pred_imgs {
        // Track unique labels for the current image.
        let mut labels = HashSet::new();
        let mut bbox_rows = Vec::new();

        // Process each bounding box in the predicted image.
        for bbox in pred_img.list_bbox {
            // Add the bounding box details to the bbox_rows.
            bbox_rows.push(vec![
                bbox.x1.to_string(),
                bbox.y1.to_string(),
                bbox.x2.to_string(),
                bbox.y2.to_string(),
                bbox.class_id.to_string(),
                bbox.confidence.to_string(),
            ]);
            // Add the label to the set of unique labels.
            labels.insert(bbox.class_id.to_string());
        }

        // Write a row for the predicted image, including the count of bounding boxes
        // and the unique labels.
        wtr.write_record(&[
            &pred_img.file_path,
            &bbox_rows.len().to_string(),
            &labels.into_iter().collect::<Vec<String>>().join(", "),
        ])?;

    }

    // Flush and write the CSV.
    wtr.flush()?;
    Ok(())
}

// The final implementation should be more like: 

// struct ImgPred<T: BoundingBoxTrait> {
//     file_path: String,
//     list_bbox: Vec<T>,
// }

// fn write_csv<T: BoundingBoxTrait>(pred_imgs: Vec<ImgPred<T>>, output_path: &str) -> io::Result<()> {
//     // Create a CSV writer.
//     let mut wtr = Writer::from_path(output_path)?;

//     // Write the header row.
//     wtr.write_record(&["File Path", "X1", "Y1", "X2", "Y2", "Label", "Confidence"])?;

//     // Write each row for the bounding boxes.
//     for pred_img in pred_imgs {
//         for bbox in pred_img.list_bbox {
//             let coords = bbox.get_coords();
//             wtr.write_record(&[
//                 pred_img.file_path.clone(),
//                 coords.0.to_string(),
//                 coords.1.to_string(),
//                 coords.2.to_string(),
//                 coords.3.to_string(),
//                 bbox.get_class_id().to_string(),
//                 bbox.get_prob().to_string(),
//             ])?;
//         }
//     }

//     // Flush the writer to ensure all data is written.
//     wtr.flush()?;
//     Ok(())
// }

pub async fn read_predictions_from_file(input_path: &str) -> io::Result<Vec<BBox>> {
    // Create expected filename based on input filepath
    let input_path = Path::new(input_path);
    let file_stem = input_path.file_stem().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Invalid input path")
    })?;
    
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
            parts[1].parse::<f32>(),  // x1
            parts[2].parse::<f32>(),  // y1
            parts[3].parse::<f32>(),  // x2
            parts[4].parse::<f32>(),  // y2
            parts[5].parse::<u16>(),  // class_id
            parts[6].parse::<f32>(),  // confidence
        ) {
            (Ok(x1), Ok(y1), Ok(x2), Ok(y2), Ok(class_id), Ok(confidence)) => {
                bboxes.push(BBox {
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    class_id,
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

pub async fn write_pred_img_to_file(pred_img: &ImgPred) -> io::Result<()> {
    // Create output filename based on input filepath
    let input_path = Path::new(&pred_img.file_path);
    let file_stem = input_path.file_stem().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Invalid input path")
    })?;
    
    let parent = input_path.parent().unwrap_or(Path::new(""));
    let output_path = parent.join(format!("{}_predictions.txt", file_stem.to_string_lossy()));
    
    // Open file for writing - using tokio's async File
    let mut file = File::create(&output_path)?;
    
    // Create the full content string
    let mut content = String::new();
    for bbox in &pred_img.list_bbox {
        content.push_str(&format!(
            "{} {} {} {} {} {} {}\n", 
            bbox.label, 
            bbox.x1, 
            bbox.y1, 
            bbox.x2, 
            bbox.y2, 
            bbox.class_id, 
            bbox.confidence
        ));
    }
    
    // Write the content to file in one operation
    file.write_all(content.as_bytes())?;
    
    // println!("Successfully wrote predictions to: {:?}", output_path);
    Ok(())
}

// Count processed images from a list of ImgPred structs
fn count_processed_images(images: &Vec<ImgPred>) -> usize {
    images.iter().filter(|img| img.wasprocessed).count()
}

// Check if all images have empty bounding boxes
fn are_boxes_empty(images: &Vec<ImgPred>) -> bool {
    for image in images {
        if !image.list_bbox.is_empty() {
            return false;
        }
    }
    true
}

// Get the most frequent label from a list of bounding boxes
fn get_main_label(listbbox: &Vec<BBox>) -> String {
    if listbbox.is_empty() {
        return String::from("no predictions");
    } else {
        let mut label_counts: std::collections::HashMap<&String, usize> = std::collections::HashMap::new();

        for bbox in listbbox {
            *label_counts.entry(&bbox.label).or_insert(0) += 1;
        }

        let main_label = label_counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| *label)
            .unwrap();

        return main_label.clone();
    }
}

async fn copy_to_folder(pred_imgs: &Vec<ImgPred>, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    for pred_img in pred_imgs {
        let image_file_path = &pred_img.file_path;
        if std::path::Path::new(image_file_path).exists() {
            let main_label = get_main_label(&pred_img.list_bbox);
            
            let folder_path = format!("{}/{}", output_path, main_label);
            
            // Create directory if it doesn't exist
            if !std::path::Path::new(&folder_path).exists() {
                std::fs::create_dir_all(&folder_path)?;
            }
            
            // Extract the image name from path
            let image_name = std::path::Path::new(image_file_path)
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default();
                
            let new_image_path = format!("{}/{}", folder_path, image_name);
            
            // Read the original file and write to the new location
            let image_data = tokio::fs::read(image_file_path).await?;
            tokio::fs::write(&new_image_path, image_data).await?;
        }
    }
    
    Ok(())
}