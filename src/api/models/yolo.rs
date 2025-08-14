use super::*;
use crate::api::{
    abstractions::{BoundingBoxTrait, XYXY},
    processing::post_processing::{nms_indices, process_mask},
    processing::{
        inference::inference,
        post_processing::*,
        pre_processing::{imgbuf_to_input_array, TensorFormat},
    },
};
use image::{ImageBuffer, Rgb};
use ndarray::{s, Array, Array2, Axis, IxDyn};
use ort::{session::Session, value::ValueType};

enum YoloType {
    Yolov5,
    Yolov8plus,
}

type DetectProcessor = fn(&Yolo, &Array<f32, IxDyn>, u32, u32) -> Vec<XYXYc>;

pub struct Yolo {
    pub classes: Vec<String>,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    num_masks: u32,
    mask_height: u32,
    mask_width: u32,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
    pub config: ModelConfig,
    yolotype: YoloType,
    detect_processor: DetectProcessor,
}

impl Yolo {
    fn process_detect_output(
        &self,
        output: &Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
    ) -> Vec<XYXYc> {
        let output = output.slice(s![.., .., 0]);
        let x_scale = img_width as f32 / self.input_width as f32;
        let y_scale = img_height as f32 / self.input_height as f32;

        let mut boxes: Vec<XYXY> = output
            .axis_iter(Axis(0))
            .filter_map(|row| {
                let row: Vec<f32> = row.iter().copied().collect();
                let (class_id, prob) = row
                    .iter()
                    .skip(4)
                    .enumerate()
                    .map(|(index, &value)| (index, value))
                    .reduce(|a, b| if b.1 > a.1 { b } else { a })?;

                if prob < self.config.confidence_threshold {
                    return None;
                }

                let xc = row[0] * x_scale;
                let yc = row[1] * y_scale;
                let w = row[2] * x_scale;
                let h = row[3] * y_scale;

                let x1 = xc - w * 0.5;
                let x2 = xc + w * 0.5;
                let y1 = yc - h * 0.5;
                let y2 = yc + h * 0.5;

                Some(XYXY::new(x1, y1, x2, y2, prob, class_id as u32))
            })
            .collect();

        for technique in &self.post_processing {
            if matches!(technique, PostProcessing::NMS) {
                let indices = nms_indices(&boxes, self.config.nms_threshold);
                boxes = indices.iter().map(|&idx| boxes[idx]).collect();
            }
        }

        self.t(&boxes)
    }

    fn process_detect_output_yolov5(
        &self,
        output: &Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
    ) -> Vec<XYXYc> {
        let output = output.slice(s![.., .., 0]);
        let x_scale = img_width as f32 / self.input_width as f32;
        let y_scale = img_height as f32 / self.input_height as f32;
        let mut boxes: Vec<XYXY> = output
            .axis_iter(Axis(1))
            .filter_map(|row| {
                let row: Vec<f32> = row.iter().copied().collect();
                let prob = row[4];
                if prob < self.config.confidence_threshold {
                    return None;
                }
                let class_id = row
                    .iter()
                    .skip(6)
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                let xc = row[0] * x_scale;
                let yc = row[1] * y_scale;
                let w = row[2] * x_scale;
                let h = row[3] * y_scale;

                let x1 = xc - w * 0.5;
                let x2 = xc + w * 0.5;
                let y1 = yc - h * 0.5;
                let y2 = yc + h * 0.5;

                Some(XYXY::new(x1, y1, x2, y2, prob, class_id as u32))
            })
            .collect();

        for technique in &self.post_processing {
            if matches!(technique, PostProcessing::NMS) {
                let indices = nms_indices_all_cls(&boxes, self.config.nms_threshold);
                boxes = indices.iter().map(|&idx| boxes[idx]).collect();
            }
        }

        self.t(&boxes)
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

    fn process_seg_output(
        &self,
        outputs: (Array<f32, IxDyn>, Array<f32, IxDyn>),
        img_width: u32,
        img_height: u32,
    ) -> Vec<SEGc> {
        let (raw_detections, proto_tensor) = outputs;
        let coeff_limit = (self.classes.len() + 4) as usize;
        // Extract bounding boxes and class scores
        let bbox_and_scores = raw_detections.slice(s![.., 0..coeff_limit, 0]).to_owned();
        // Extract mask coefficients for each detection
        let coefs: Array2<f32> = raw_detections
            .slice(s![.., coeff_limit..(self.output_width as usize), 0])
            .to_owned();
        // Reshape prototype tensor to (channels, height * width)
        let proto_raw = proto_tensor.slice(s![.., .., .., 0]); // shape: (batch, h, w)

        #[allow(deprecated)]
        let proto_mask_features = proto_raw
            .to_owned()
            .into_shape((
                self.mask_height as usize * self.mask_width as usize,
                self.num_masks as usize,
            )) // (h * w, channels)
            .unwrap()
            .permuted_axes([1, 0]) // -> (channels, h * w)
            .to_owned();
        let x_scale = img_width as f32 / self.input_width as f32;
        let y_scale = img_height as f32 / self.input_height as f32;

        // Process all detections with iterator chain
        let (mut segmentations, bounding_boxes): (Vec<SEGc>, Vec<XYXY>) = bbox_and_scores
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(index, row)| {
                let values: Vec<f32> = row.iter().copied().collect();
                // Determine most probable class
                let (class_id, score) = values
                    .iter()
                    .skip(4)
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .reduce(|acc, val| if val.1 > acc.1 { val } else { acc })
                    .unwrap();
                if score < self.config.confidence_threshold {
                    return None;
                }

                let coeffs = coefs.row(index).insert_axis(ndarray::Axis(0)); // shape: (1, 32)

                #[allow(deprecated)]
                let mask: Array2<f32> = coeffs
                    .dot(&proto_mask_features) // shape: (1, h * w)
                    .into_shape((self.mask_height as usize, self.mask_width as usize)) // reshape
                    .expect("Failed to reshape mask")
                    .to_owned(); // make it an Array2<f32>

                let xc = values[0] * x_scale;
                let yc = values[1] * y_scale;
                let w = values[2] * x_scale;
                let h = values[3] * y_scale;

                let x1 = xc - w * 0.5;
                let x2 = xc + w * 0.5;
                let y1 = yc - h * 0.5;
                let y2 = yc + h * 0.5;
                let bbox = XYXY::new(x1, y1, x2, y2, score, class_id as u32);
                let seg = process_mask(
                    mask,
                    &bbox,
                    img_width,
                    img_height,
                    self.mask_height,
                    self.mask_width,
                );
                let segc = SEGc::new(seg, XYXYc::new(bbox, self.classes[class_id].to_string()));
                Some((segc, bbox))
            })
            .unzip();

        for technique in &self.post_processing {
            if matches!(technique, PostProcessing::NMS) {
                let keep_indices: Vec<usize> =
                    nms_indices(&bounding_boxes, self.config.nms_threshold);
                segmentations = keep_indices
                    .iter()
                    .map(|&i| segmentations[i].clone()) // use `.clone()` if needed, depending on the type
                    .collect();
            }
        }

        return segmentations;
    }
}

impl ModelTrait for Yolo {
    fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        config: ModelConfig,
    ) -> Self {
        let (_batch_size, _input_depth, input_width, input_height) =
            match &session.inputs[0].input_type {
                ValueType::Tensor { dimensions, .. } => (
                    dimensions[0] as u32,
                    dimensions[1] as u32,
                    dimensions[2] as u32,
                    dimensions[3] as u32,
                ),
                _ => {
                    panic!("Not supported");
                }
            };

        let (output_n, output_width, output_height) = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => (
                dimensions[0] as u32,
                dimensions[1] as u32,
                dimensions[2] as u32,
            ),
            _ => {
                panic!("Not supported");
            }
        };

        let (yolotype, detect_processor): (YoloType, DetectProcessor) = if output_n > output_width {
            (YoloType::Yolov8plus, Yolo::process_detect_output)
        } else {
            (YoloType::Yolov5, Yolo::process_detect_output_yolov5)
        };

        let (num_masks, mask_width, mask_height) = if let Some(output) = session.outputs.get(1) {
            match &output.output_type {
                ValueType::Tensor { dimensions, .. } => (
                    dimensions[1] as u32,
                    dimensions[2] as u32,
                    dimensions[3] as u32,
                ),
                _ => {
                    panic!("This shouldn't happen");
                }
            }
        } else {
            (0, 0, 0)
        };

        Yolo {
            classes,
            input_width,
            input_height,
            output_width,
            output_height,
            num_masks,
            mask_height,
            mask_width,
            task,
            post_processing,
            session,
            config,
            yolotype,
            detect_processor
        }
    }

    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (input, img_width, img_height) = imgbuf_to_input_array(
            1,
            3,
            self.input_height,
            self.input_width,
            img,
            TensorFormat::NCHW,
        );
        let outputs = inference(&self.session, &input, "images");
        match self.task {
            Task::Detect => {
                let output = extract_output(&outputs, "output0");
                let boxes = (self.detect_processor)(self, &output, img_width, img_height);                
                return AIOutputs::ObjectDetection(boxes);
            }
            Task::Classify => {
                let output = extract_output(&outputs, "output0");
                let probs =
                    process_class_output(self.config.confidence_threshold, &self.classes, &output);
                return AIOutputs::Classification(probs);
            }
            Task::Segment => {
                let output0 = extract_output(&outputs, "output0");
                let output1 = extract_output(&outputs, "output1");
                let segc_vec = self.process_seg_output((output0, output1), img_width, img_height);
                return AIOutputs::Segmentation(segc_vec);
            }
        }
    }
}
