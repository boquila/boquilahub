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
    img_width: u32,
    img_height: u32,
) -> (Vec<i32>, Vec<i32>) {
    let (x1, y1, x2, y2) = rect;

    // Create binary mask
    let mut binary_mask = vec![vec![0u8; 160]; 160];
    for i in 0..160 {
        for j in 0..160 {
            binary_mask[i][j] = if mask[[i, j]] > 0.0 { 255 } else { 0 };
        }
    }

    // Find contours using a simple contour tracing algorithm
    let contours = find_contours(&binary_mask);

    // Get the largest contour (main object)
    let main_contour = contours
        .into_iter()
        .max_by_key(|contour| contour.len())
        .unwrap_or_default();

    // Transform coordinates from 160x160 mask space to image space
    let mut x_coords = Vec::new();
    let mut y_coords = Vec::new();

    for (mask_x, mask_y) in main_contour {
        // Convert from mask coordinates (0-160) to bounding box coordinates
        let rel_x = mask_x as f32 / 160.0;
        let rel_y = mask_y as f32 / 160.0;

        // Convert to image coordinates
        let img_x = x1 + rel_x * (x2 - x1);
        let img_y = y1 + rel_y * (y2 - y1);

        x_coords.push(img_x.round() as i32);
        y_coords.push(img_y.round() as i32);
    }

    (x_coords, y_coords)
}

// Simple contour finding using Moore neighborhood tracing
fn find_contours(binary_mask: &[Vec<u8>]) -> Vec<Vec<(usize, usize)>> {
    let height = binary_mask.len();
    let width = binary_mask[0].len();
    let mut visited = vec![vec![false; width]; height];
    let mut contours = Vec::new();

    // 8-connected neighborhood offsets
    let directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ];

    for i in 0..height {
        for j in 0..width {
            if binary_mask[i][j] == 255 && !visited[i][j] {
                // Found a new contour starting point
                let mut contour = Vec::new();
                let mut stack = vec![(i, j)];

                while let Some((y, x)) = stack.pop() {
                    if visited[y][x] || binary_mask[y][x] != 255 {
                        continue;
                    }

                    visited[y][x] = true;
                    contour.push((x, y));

                    // Add neighbors to stack
                    for (dy, dx) in &directions {
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;

                        if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                            let ny = ny as usize;
                            let nx = nx as usize;

                            if !visited[ny][nx] && binary_mask[ny][nx] == 255 {
                                stack.push((ny, nx));
                            }
                        }
                    }
                }

                if !contour.is_empty() {
                    // Simplify contour to reduce points (Douglas-Peucker could be applied here)
                    contours.push(simplify_contour(contour));
                }
            }
        }
    }

    contours
}

// Simple contour simplification - keep every nth point
fn simplify_contour(contour: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if contour.len() <= 10 {
        return contour;
    }

    let step = contour.len() / 10; // Keep roughly 10 points
    contour.into_iter().step_by(step.max(1)).collect()
}
