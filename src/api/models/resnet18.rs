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
    pub batch_size: i32,
    pub channel: u32, // 3, RGB or similar
    pub input_width: u32,
    pub input_height: u32,
    pub input_name: String,
    pub output_width: u32,
    pub output_height: u32,
    pub output_name: String,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
    pub config: ModelConfig,
    pub input_format: TensorFormat,
}