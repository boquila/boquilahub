#![allow(dead_code)]
pub mod efficientnet;
pub mod yolo;
use super::abstractions::*;
use crate::api::{models::efficientnet::EfficientNetV2, processing::post_processing::PostProcessing};
use image::{ImageBuffer, Rgb};
use ort::session::Session;
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

// All supported architectures
pub enum Model {
    EfficientNetV2(EfficientNetV2),
    Yolo(Yolo),
}

pub trait ModelTrait {
    fn new(
        classes: Vec<String>,
        confidence_threshold: f32,
        nms_threshold: f32,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
    ) -> Self;
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs;
}

impl Model {
    pub fn new(
        classes: Vec<String>,
        confidence_threshold: f32,
        nms_threshold: f32,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        architecture: Option<String>,
    ) -> Result<Self, String> {
        let arch = architecture.as_ref().map(|s| s.to_lowercase());
        match arch.as_deref() {
            Some("yolo") => Ok(Model::Yolo(Yolo::new(
                classes,
                confidence_threshold,
                nms_threshold,
                task,
                post_processing,
                session,
            ))),
            Some("efficientnetv2") => {
                Ok(Model::EfficientNetV2(EfficientNetV2::new(
                    classes,
                    confidence_threshold, 
                    nms_threshold,
                    task,
                    post_processing,
                    session,
                )))
            }
            Some(arch) => {
                Err(format!("Unsupported model architecture: {}", arch))
            }
            None => {
                Err("No architecture specified".to_string())
            }
        }
    }

    pub fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        match self {
            Model::EfficientNetV2(model) => model.run(img),
            Model::Yolo(model) => model.run(img),
        }
    }
}
