pub mod yolo;
use ndarray::{Array, IxDyn};
pub use yolo::Yolo;

use super::abstractions::XYXY;

pub struct AIModel {
    pub name: String,
    pub description: String,
    pub version: f32,
    pub classes: Vec<String>,
    pub model: Architecture,
}

impl AIModel {
    // Basic constructor
    pub fn new(name: String, description: String, version: f32, classes: Vec<String>, model: Architecture) -> Self {
        AIModel {
            name,
            description,
            version,
            classes,
            model,
        }
    }

    pub fn default() -> Self {
        AIModel::new(
            "boquilanet-gen".to_string(),
            "Generic animal detection".to_string(),
            0.1,
            vec!["animal".to_string()],
            Architecture::Yolo(Yolo::new(
                1024,
                1024,
                0.45,
                0.5,
                1,
                0,                
                Task::Detect,
            )),
        )
    }

    pub fn process_output(
        &self,
        output: &Array<f32, IxDyn>,
        img_width: u32,
        img_height: u32,
        input_width: u32,
        input_height: u32,
    ) -> Vec<XYXY> {
        match &self.model {
            Architecture::Yolo(yolo) => {
                yolo.process_output(output, img_width, img_height, input_width, input_height)
            }
        }
    }

    pub fn get_input_dimensions(&self) -> (u32, u32) {
        match &self.model {
            Architecture::Yolo(yolo) => (yolo.input_height, yolo.input_width),
        }
    }
}

pub enum Task {
    Classify,
    Segment,
    Detect,
}

impl From<&str> for Task {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "detect" => Task::Detect,
            "classify" => Task::Classify,
            "segment" => Task::Segment,
            _ => Task::Detect, // Default to Detect if unknown
        }
    }
}

pub enum PostProcessing {
    NMS,
}

pub enum Architecture {
    Yolo(Yolo),
}

