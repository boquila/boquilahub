use super::*;
use crate::api::{
    abstractions::XYXY,
    processing::post::{nms_indices, process_mask},
    processing::{
        inference::inference,
        post::*,
        pre::{imgbuf_to_input_array, TensorFormat},
    },
};
use image::{ImageBuffer, Rgb};
use ndarray::{s, Array, Array2, Axis, IxDyn};
use ort::{session::Session, value::ValueType};

pub struct ResNet18 {
    pub classes: Vec<String>,

    // Input Tensor
    pub batch_size: i32,   // number of windows/clips
    pub channel: u32,      //  number of channel, 1 for single channel mel spectrogram
    pub input_height: u32, // number of mel freq bins
    pub input_width: i32,  // time steps, time frames, (width of the spectrogram)
    pub input_name: String,
    // Output Tensor
    pub output_width: i32,
    pub output_height: u32,
    pub output_name: String,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
    pub config: ModelConfig,
}

impl ModelTrait for ResNet18 {
    fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        config: ModelConfig,
    ) -> Self {
        let (batch_size, channel, input_height, input_width) = match &session.inputs[0].input_type {
            ValueType::Tensor { dimensions, .. } => (
                dimensions[0] as i32,
                dimensions[1] as u32,
                dimensions[2] as u32,
                dimensions[3] as i32,
            ),
            _ => {
                panic!("Not supported");
            }
        };

        let input_name = session.inputs[0].name.clone();

        let (output_width, output_height) = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => (dimensions[0] as i32, dimensions[1] as u32),
            _ => {
                panic!("Not supported");
            }
        };

        let output_name: String = session.outputs[0].name.clone();

        ResNet18 {
            classes,
            batch_size,
            channel,
            input_width,
            input_height,
            input_name,
            output_width,
            output_height,
            output_name,
            task,
            post_processing,
            session,
            config,
        }
    }

    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let probs = ProbSpace {
            classes: vec![],
            probs: vec![],
            classes_ids: vec![],
        };
        return AIOutputs::Classification(probs);
    }
}
