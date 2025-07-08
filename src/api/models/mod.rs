#![allow(dead_code)]
pub mod yolo;
pub mod efficientnet;
pub mod processing;
use image::{ImageBuffer, Rgb};
pub use yolo::Yolo;
use crate::api::models::{efficientnet::EfficientNetV2, processing::inference::AIOutputs};
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

// All supported architectures
pub enum Model {
    EfficientNetV2(EfficientNetV2),
    Yolo(Yolo),
}

pub trait ModelTrait {
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs;
}

// Implement the trait for the enum
impl ModelTrait for Model {
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        match self {
            Model::EfficientNetV2(model) => model.run(img),
            Model::Yolo(model) => model.run(img),
        }
    }
}