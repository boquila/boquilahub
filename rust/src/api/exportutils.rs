#![allow(dead_code)]
use std::fs::File;
use std::io::{self};
use csv::Writer;
use csv::WriterBuilder;
use std::collections::HashSet;
use super::abstractions::ImgPred;

pub fn write_csv(pred_imgs: Vec<ImgPred>, output_path: &str) -> io::Result<()> {
    // Create a CSV writer.
    let mut wtr = Writer::from_path(output_path)?;

    // Write the header row.
    wtr.write_record(&["File Path", "X1", "Y1", "X2", "Y2", "Label", "Confidence"])?;

    // Write each row for the bounding boxes.
    for pred_img in pred_imgs {
        for bbox in pred_img.list_bbox {
            wtr.write_record(&[
                pred_img.file_path.clone(),
                bbox.x1.to_string(),
                bbox.y1.to_string(),
                bbox.x2.to_string(),
                bbox.y2.to_string(),
                bbox.class_id.to_string(),
                bbox.prob.to_string(),
            ])?;
        }
    }

    // Flush the writer to ensure all data is written.
    wtr.flush()?;
    Ok(())
}

pub fn write_csv2(pred_imgs: Vec<ImgPred>, output_path: &str) -> io::Result<()> {
    // Open the output file.
    let file = File::create(output_path)?;
    let mut wtr = WriterBuilder::new().has_headers(true).from_writer(file);

    // Write the header row.
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
                bbox.prob.to_string(),
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

        // Optionally, you could add bbox_rows data here if needed, though
        // the original Dart function only writes summary information for each image.
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
