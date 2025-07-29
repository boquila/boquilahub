use crate::api::{
    abstractions::ProbSpace,
    models::{
        processing::{
            ensemble::ensemble::get_common_name,
            inference::{inference, AIOutputs},
            post_processing::{extract_output, process_class_output_logits, PostProcessing},
            pre_processing::{imgbuf_to_input_array_nhwc},
        },
        ModelTrait, Task,
    },
};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

pub struct EfficientNetV2 {
    pub classes: Vec<String>,
    batch_size: i32,
    input_depth: u32, // 3, RGB or similar
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
}

impl ModelTrait for EfficientNetV2 {
    fn new(
        classes: Vec<String>,
        confidence_threshold: f32,
        nms_threshold: f32,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
    ) -> Self {
        let (batch_size, input_width, input_height, input_depth) =
            match &session.inputs[0].input_type {
                ValueType::Tensor { dimensions, .. } => (
                    dimensions[0] as i32,
                    dimensions[1] as u32,
                    dimensions[2] as u32,
                    dimensions[3] as u32,
                ),
                _ => {
                    panic!("Not supported");
                }
            };

        let (output_width, output_height) = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => (dimensions[0] as u32, dimensions[1] as u32),
            _ => {
                panic!("Not supported");
            }
        };

        EfficientNetV2 {
            classes,
            batch_size,
            input_depth,
            input_width,
            input_height,
            output_width,
            output_height,
            confidence_threshold,
            nms_threshold,
            task,
            post_processing,
            session,
        }
    }
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let input =
            imgbuf_to_input_array_nhwc(1, 3, self.input_height, self.input_width, img);
        let outputs = inference(&self.session, &input, "input_2:0");
        let output = extract_output(&outputs, "Identity:0");
        let mut probs: ProbSpace =
            process_class_output_logits(self.confidence_threshold, &self.classes, &output);
        for technique in &self.post_processing {
            if matches!(technique, PostProcessing::Rollup) {
                probs.classes = probs
                    .classes
                    .iter()
                    .map(|line| get_common_name(line))
                    .collect();
            }
        }

        return AIOutputs::Classification(probs);
    }
}
