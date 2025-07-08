use crate::api::models::{processing::{inference::{inference, AIOutputs}, post_processing::{extract_output, PostProcessingTechnique}, pre_processing::imgbuf_to_input_array_nhwc}, ModelTrait, Task};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

pub struct EfficientNetV2 {
    pub classes: Vec<String>,
    batch_size: i32,
    input_depth: u32, // 3, RGB
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    pub confidence_threshold: f32,
    pub task: Task,
    pub post_processing: Vec<PostProcessingTechnique>,
    pub session: Session,
}
impl EfficientNetV2 {
    pub fn new(
        classes: Vec<String>,
        confidence_threshold: f32,
        task: Task,
        post_processing: Vec<PostProcessingTechnique>,
        session: Session,
    ) -> Self {
        let (batch_size, input_depth, input_width, input_height) =
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
            ValueType::Tensor { dimensions, .. } => (dimensions[1] as u32, dimensions[2] as u32),
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
            task,
            post_processing,
            session,
        }
    }
}

impl ModelTrait for EfficientNetV2 {
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let input =
            imgbuf_to_input_array_nhwc(1, 3, self.input_height, self.input_width, img);
        let outputs = inference(&self.session, &input, "input_2:0");
        let output = extract_output(&outputs, "Identity:0");
        todo!()
    }
}