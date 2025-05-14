use crate::api::abstractions::{nms_indices, BoundingBoxTrait, XYXY};

use super::*;
use ndarray::{s, Array, Axis, IxDyn};

pub struct Yolo {
    pub input_width: u32,
    pub input_height: u32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub num_classes: u32,
    pub num_masks: u32,
    pub classes: Vec<String>,
    pub task: Task,
}

impl Yolo {
    pub fn new(
        input_width: u32,
        input_height: u32,
        confidence_threshold: f32,
        nms_threshold: f32,
        num_classes: u32,
        num_masks: u32,
        classes: Vec<String>,
        task: Task,
    ) -> Self {
        Self {
            input_width,
            input_height,
            confidence_threshold,
            nms_threshold,
            num_classes,
            num_masks,
            classes,
            task,
        }
    }

    pub fn process_output(
        &self,
        output: &Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
        input_width: u32,
        input_height: u32,
    ) -> Vec<XYXY> {
        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<f32> = row.iter().map(|x| *x).collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < self.confidence_threshold {
                continue;
            }
            let label = class_id as u16;
            // XYWHn::new(row[0],row[1],row[0],row[3],prob,label);
            let xc = row[0] / input_width as f32 * (img_width as f32);
            let yc = row[1] / input_height as f32 * (img_height as f32);
            let w = row[2] / input_width as f32 * (img_width as f32);
            let h = row[3] / input_height as f32 * (img_height as f32);
            let x1 = xc - w / 2.0;
            let x2 = xc + w / 2.0;
            let y1 = yc - h / 2.0;
            let y2 = yc + h / 2.0;
            let temp = XYXY::new(x1, y1, x2, y2, prob, label);
            boxes.push(temp);
        }

        let indices = nms_indices(&boxes, self.nms_threshold);
        let result = indices.iter().map(|&idx| boxes[idx].clone()).collect();

        return result;
    }
}
