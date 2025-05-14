mod yolo;
use yolo::Yolo;

pub struct AIModel{
    pub name: String,
    pub description: String,          
    pub version: f32, 
    pub model: ModelType
}

// Supported models
pub enum ModelType {
    Yolo(Yolo),
}

pub enum Task {
    Classify,
    Segment,
    Detect,
}

