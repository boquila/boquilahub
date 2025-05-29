#![allow(dead_code)]
pub mod yolo;
use super::{abstractions::*, bq::import_bq};
use ndarray::{Array, Ix4, IxDyn};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
};
pub use yolo::Yolo;

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

enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGn>),
}
