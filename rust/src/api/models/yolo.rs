
use super::*;

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
}