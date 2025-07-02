use super::*;
use crate::api::{
    abstractions::{BoundingBoxTrait, XYXY},
    models::processing::{
        inference::AIOutputs, post_processing::*, pre_processing::imgbuf_to_input_array,
    },
};
use derive_new::new;
use image::{ImageBuffer, Rgb};
use ndarray::{s, Array, Array2, Axis, Ix4, IxDyn};
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

    fn process_class_output(&self, output: &Array<f32, IxDyn>) -> ProbSpace {
        let mut indexed_scores: Vec<(usize, f32)> = output
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= self.confidence_threshold)
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let probs: Vec<f32> = indexed_scores.iter().map(|(_, prob)| *prob).collect();
        let classes_ids: Vec<usize> = indexed_scores.iter().map(|(idx, _)| *idx).collect();
        let classes: Vec<String> = classes_ids
            .iter()
            .map(|&idx| self.classes[idx].clone())
            .collect();

        return ProbSpace::new(probs, classes_ids, classes);
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

    fn t_seg(&self, segs: &Vec<SEG>, bboxes: &Vec<XYXY>) -> Vec<SEGc> {
        segs.iter()
            .zip(bboxes.iter())
            .map(|(seg, bbox)| {
                let label = &self.classes[seg.class_id as usize];
                SEGc {
                    seg: seg.clone(),
                    bbox: bbox.clone(),
                    label: label.to_string(),
                }
            })
            .collect()
    }

    fn process_seg_output(
        &self,
        outputs: (Array<f32, IxDyn>, Array<f32, IxDyn>),
        img_width: u32,
        img_height: u32,
    ) -> Vec<SEGc> {
        let (output0, output1) = outputs;
        let boxes_output = output0.slice(s![.., 0..84, 0]).to_owned();
        let masks_output: Array2<f32> = output1
            .slice(s![.., .., .., 0])
            .to_owned()
            .into_shape_with_order((160 * 160, 32))
            .unwrap()
            .permuted_axes([1, 0])
            .to_owned();
        let masks_output2: Array2<f32> = output0.slice(s![.., 84..116, 0]).to_owned();
        let masks = masks_output2
            .dot(&masks_output)
            .into_shape_with_order((8400, 160, 160))
            .unwrap()
            .to_owned();

        let mut segs = Vec::new();
        let mut bboxes = Vec::new();

        for (index, row) in boxes_output.axis_iter(Axis(0)).enumerate() {
            let row: Vec<_> = row.iter().map(|x| *x).collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < 0.2 {
                continue;
            }

            let mask: Array2<f32> = masks.slice(s![index, .., ..]).to_owned();

            let xc = row[0] / 640.0 * (img_width as f32);
            let yc = row[1] / 640.0 * (img_height as f32);
            let w = row[2] / 640.0 * (img_width as f32);
            let h = row[3] / 640.0 * (img_height as f32);
            let x1 = xc - w / 2.0;
            let x2 = xc + w / 2.0;
            let y1 = yc - h / 2.0;
            let y2 = yc + h / 2.0;

            // Extract polygon from mask
            let polygon = extract_polygon_from_mask(mask, (x1, y1, x2, y2), img_width, img_height);
            bboxes.push(XYXY::new(x1, y1, x2, y2, prob, class_id as u16));
            segs.push(SEG::new(polygon.0, polygon.1, prob, class_id as u16));
        }

        let indices: Vec<usize> = nms_indices(&bboxes, self.nms_threshold);

        let filtered_segs: Vec<SEG> = indices.iter().map(|&i| segs[i].clone()).collect();
        let filtered_bboxes: Vec<XYXY> = indices.iter().map(|&i| bboxes[i].clone()).collect();

        let segs: Vec<SEGc> = self.t_seg(&filtered_segs, &filtered_bboxes);
        return segs;
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
                let output = self.inference(&input);
                let probs = self.process_class_output(&output);
                return AIOutputs::Classification(probs);
            }
            Task::Segment => {
                let outputs = self
            .session
            .run(inputs!["images" => input.view()].unwrap())
            .unwrap();
    let output0 = outputs["output0"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .t()
            .into_owned();
        let output1 = outputs["output1"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .t()
            .into_owned();
    let o = self.process_seg_output((output0,output1), img_width, img_height);
             return AIOutputs::Segmentation(o);   
            }
        }
    }
}

