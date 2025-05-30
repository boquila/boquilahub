#![allow(dead_code)]
pub mod yolo;
pub use yolo::Yolo;
use super::{abstractions::*};

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

impl From<&str> for PostProcessing {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "NMS" => PostProcessing::NMS,
            _ => PostProcessing::NMS,
        }
    }
}

// All supported architectures
pub enum Architecture {
    Yolo(Yolo),
}

pub enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGn>),
}
