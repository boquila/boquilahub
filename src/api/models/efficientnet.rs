use ort::session::Session;
use crate::api::models::Task;

pub struct EfficientNetV2 {
    pub classes: Vec<String>,
    batch_size: i32,
    input_depth: u32, // 3, RGB
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub num_classes: u32,
    pub task: Task,
    pub session: Session,
}