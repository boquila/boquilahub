use super::*;
use crate::api::{
    abstractions::{BoundingBoxTrait, XYXY},
    models::processing::{inference::AIOutputs, post_processing::nms_indices, pre_processing::imgbuf_to_input_array},
};
use derive_new::new;
use image::{ImageBuffer, Rgb};
use ndarray::{s, Array, Axis, Ix4, IxDyn};
use ort::{inputs, session::Session};

#[derive(new)]
pub struct Yolo {
    pub name: String,
    pub description: String,
    pub version: f32,
    pub classes: Vec<String>,
    pub input_width: u32,
    pub input_height: u32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub num_classes: u32,
    pub num_masks: u32,
    pub task: Task,
    pub session: Session,
}

impl Yolo {
    fn inference(&self, input: &Array<f32, Ix4>) -> Array<f32, IxDyn> {
        let outputs = self
            .session
            .run(inputs!["images" => input.view()].unwrap())
            .unwrap();

        let predictions = outputs["output0"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .t()
            .into_owned();
        return predictions;
    }

    fn process_detect_output(
        &self,
        output: &Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
    ) -> Vec<XYXYc> {
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
            // XYWHn::new(row[0],row[1],row[0],row[3],prob,label);
            let xc = row[0] / self.input_width as f32 * (img_width as f32);
            let yc = row[1] / self.input_height as f32 * (img_height as f32);
            let w = row[2] / self.input_width as f32 * (img_width as f32);
            let h = row[3] / self.input_height as f32 * (img_height as f32);
            let x1 = xc - w / 2.0;
            let x2 = xc + w / 2.0;
            let y1 = yc - h / 2.0;
            let y2 = yc + h / 2.0;
            boxes.push(XYXY::new(x1, y1, x2, y2, prob, class_id as u16));
        }

        let indices = nms_indices(&boxes, self.nms_threshold);
        let result: Vec<XYXY> = indices.iter().map(|&idx| boxes[idx].clone()).collect();
        return self.t(&result);
    }

    fn t(&self, boxes: &Vec<XYXY>) -> Vec<XYXYc> {
        boxes
            .into_iter()
            .map(|xyxy| {
                let label = &self.classes[xyxy.get_class_id() as usize];
                xyxy.to_xyxyc(None, None, label.to_string())
            })
            .collect()
    }

    pub fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (input, img_width, img_height) =
            imgbuf_to_input_array(1, 3, self.input_height, self.input_width, img);
        match self.task {
            Task::Detect => {
                let output = self.inference(&input);
                let boxes = self.process_detect_output(&output, img_width, img_height);
                return AIOutputs::ObjectDetection(boxes);
            }
            Task::Classify => {
                todo!();
            }
            Task::Segment => {
                todo!();
            }
        }
    }
}
