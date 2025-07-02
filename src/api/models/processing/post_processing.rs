use image::{imageops::FilterType, GenericImage, Rgba};
use ndarray::Array2;

use crate::api::abstractions::BoundingBoxTrait;

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

pub fn process_mask(mask:Array2<f32>,rect:(f32,f32,f32,f32),img_width:u32, img_height:u32) -> Vec<Vec<u8>> {
    let (x1,y1,x2,y2) = rect;
    let mut mask_img = image::DynamicImage::new_rgb8(161,161);
    let mut index = 0.0;
    mask.for_each(|item| {
        let color = if *item > 0.0 { Rgba::<u8>([255,255,255,1])  } else { Rgba::<u8>([0,0,0,1]) };
        let y = f32::floor(index / 160.0);
        let x = index - y * 160.0;
        mask_img.put_pixel(x as u32, y as u32, color);
        index += 1.0;
    });
    mask_img = mask_img.crop((x1 / img_width as f32 * 160.0).round() as u32,
                             (y1 / img_height as f32 * 160.0).round() as u32,
                             ((x2-x1) / img_width as f32 * 160.0).round() as u32,
                             ((y2-y1) / img_height as f32 * 160.0).round() as u32
    );
    mask_img = mask_img.resize_exact((x2-x1) as u32,(y2-y1) as u32, FilterType::Nearest);
    let mut result = vec![];
    for y in 0..(y2-y1) as usize {
        let mut row = vec![];
        for x in 0..(x2-x1) as usize {
            let color= image::GenericImageView::get_pixel(&mask_img, x as u32, y as u32);
            row.push(*color.0.iter().nth(0).unwrap());
        }
        result.push(row);
    }
    return result;
}