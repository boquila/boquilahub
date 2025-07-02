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


// Extract polygon contour from segmentation mask
pub fn extract_polygon_from_mask(
    mask: Array2<f32>,
    rect: (f32, f32, f32, f32),
    _img_width: u32,
    _img_height: u32,
) -> (Vec<i32>, Vec<i32>) {
    let (x1, y1, x2, y2) = rect;
    
    // Apply sigmoid and threshold to create binary mask
    let threshold = 0.5;
    let mut binary_mask = vec![vec![false; 160]; 160];
    
    for row in 0..160 {
        for col in 0..160 {
            let sigmoid_val = 1.0 / (1.0 + (-mask[[row, col]]).exp());
            binary_mask[row][col] = sigmoid_val > threshold;
        }
    }
    
    // Find contour using simple boundary following
    let contour = find_boundary_pixels(&binary_mask);
    
    if contour.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    // Transform coordinates from 160x160 mask space to image space
    let mut x_coords = Vec::new();
    let mut y_coords = Vec::new();
    
    for (mask_x, mask_y) in contour {
        // Convert from mask coordinates (0-159) to bounding box coordinates
        let rel_x = mask_x as f32 / 159.0;
        let rel_y = mask_y as f32 / 159.0;
        
        // Convert to image coordinates
        let img_x = x1 + rel_x * (x2 - x1);
        let img_y = y1 + rel_y * (y2 - y1);
        
        x_coords.push(img_x.round() as i32);
        y_coords.push(img_y.round() as i32);
    }
    
    (x_coords, y_coords)
}

fn find_boundary_pixels(binary_mask: &[Vec<bool>]) -> Vec<(usize, usize)> {
    let mut boundary_pixels = Vec::new();
    
    for row in 0..160 {
        for col in 0..160 {
            if binary_mask[row][col] {
                // Check if this pixel is on the boundary (has at least one non-foreground neighbor)
                let mut is_boundary = false;
                
                // Check 4-connected neighbors
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    
                    if nr < 0 || nr >= 160 || nc < 0 || nc >= 160 {
                        is_boundary = true; // Edge pixels are boundary
                        break;
                    }
                    
                    if !binary_mask[nr as usize][nc as usize] {
                        is_boundary = true; // Adjacent to background
                        break;
                    }
                }
                
                if is_boundary {
                    boundary_pixels.push((col, row)); // (x, y) format
                }
            }
        }
    }
    
    if boundary_pixels.is_empty() {
        return boundary_pixels;
    }
    
    // Sort boundary pixels to create a more coherent polygon
    // Simple approach: sort by angle from centroid
    let centroid_x = boundary_pixels.iter().map(|(x, _)| *x as f32).sum::<f32>() / boundary_pixels.len() as f32;
    let centroid_y = boundary_pixels.iter().map(|(_, y)| *y as f32).sum::<f32>() / boundary_pixels.len() as f32;
    
    boundary_pixels.sort_by(|a, b| {
        let angle_a = (a.1 as f32 - centroid_y).atan2(a.0 as f32 - centroid_x);
        let angle_b = (b.1 as f32 - centroid_y).atan2(b.0 as f32 - centroid_x);
        angle_a.partial_cmp(&angle_b).unwrap()
    });
    
    // Simplify by taking every nth point if too many
    if boundary_pixels.len() > 100 {
        let step = boundary_pixels.len() / 50;
        boundary_pixels = boundary_pixels.into_iter().step_by(step).collect();
    }
    
    boundary_pixels
}