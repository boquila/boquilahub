pub mod efficientnet;
pub mod resnet18;
pub mod yolo;
use crate::api::models::resnet18::ResNet18;

use super::{audio::AudioData, abstractions::*, processing::post::PostProcessing};
use anyhow::{anyhow, Error, Result};
pub use efficientnet::EfficientNetV2;
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
    ResNet18(ResNet18),
}

pub enum AIInput<'a> {
    Image(&'a ImageBuffer<Rgb<u8>, Vec<u8>>),
    Audio(&'a AudioData),
}

impl Model {
    pub fn config_mut(&mut self) -> &mut ModelConfig {
        match self {
            Model::EfficientNetV2(inner) => &mut inner.config,
            Model::Yolo(inner) => &mut inner.config,
            Model::ResNet18(inner) => &mut inner.config,
        }
    }
}

pub trait ModelTrait {
    fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error>
    where
        Self: Sized;
}

impl Model {
    pub fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        architecture: Option<String>,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let arch = architecture.as_ref().map(|s| s.to_lowercase());
        match arch.as_deref() {
            Some("yolo") => Ok(Model::Yolo(Yolo::new(
                classes,
                task,
                post_processing,
                session,
                config,
            )?)),
            Some("efficientnetv2") => Ok(Model::EfficientNetV2(EfficientNetV2::new(
                classes,
                task,
                post_processing,
                session,
                config,
            )?)),
            Some("resnet18") => Ok(Model::ResNet18(ResNet18::new(
                classes,
                task,
                post_processing,
                session,
                config,
            )?)),
            Some(arch) => Err(anyhow!("Unsupported model architecture: {}", arch)),
            None => Err(anyhow!("No architecture specified")),
        }
    }

    pub fn run(&self, input: &AIInput<'_>) -> AIOutputs {
        match (self, input) {
            (Model::EfficientNetV2(m), AIInput::Image(img)) => m.run_image(img),
            (Model::Yolo(m), AIInput::Image(img)) => m.run_image(img),
            (Model::ResNet18(m), AIInput::Audio(audio)) => m.run_audio(audio),
            _ => panic!("wrong input type for this model architecture"),
        }
    }
}
