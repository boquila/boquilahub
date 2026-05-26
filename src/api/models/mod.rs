pub mod efficientnet;
pub mod perch;
pub mod resnet18;
pub mod yolo;
use crate::api::models::perch::PerchV2;
use crate::api::models::resnet18::ResNet18;
use super::{audio::AudioData, abstractions::*, bq::AIMetadata, processing::post::PostProcessing};
use anyhow::{anyhow, Error, Result};
pub use efficientnet::EfficientNetV2;
use image::{ImageBuffer, Rgb};
use ort::session::Session;
pub use yolo::Yolo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Classify,
    Segment,
    Detect,
}

impl Task {
    pub const fn name(&self) -> &'static str {
        match self {
            Task::Classify => "classify",
            Task::Segment => "segment",
            Task::Detect => "detect",
        }
    }
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
    PerchV2(PerchV2),
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
            Model::PerchV2(inner) => &mut inner.config,
        }
    }
}

impl Model {
    pub fn new(
        metadata: AIMetadata,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let arch = metadata.architecture.to_lowercase();
        match arch.as_str() {
            "yolo" => Ok(Model::Yolo(Yolo::new(metadata, session, config)?)),
            "efficientnetv2" => Ok(Model::EfficientNetV2(EfficientNetV2::new(metadata, session, config)?)),
            "resnet18" => Ok(Model::ResNet18(ResNet18::new(metadata, session, config)?)),
            "perch_v2" | "perch" | "perch2" => Ok(Model::PerchV2(PerchV2::new(metadata, session, config)?)),
            arch => Err(anyhow!("Unsupported model architecture: {}", arch)),
        }
    }

    pub fn run(&self, input: &AIInput<'_>) -> AIOutputs {
        match (self, input) {
            (Model::EfficientNetV2(m), AIInput::Image(img)) => m.run_image(img),
            (Model::Yolo(m), AIInput::Image(img)) => m.run_image(img),
            (Model::ResNet18(m), AIInput::Audio(audio)) => m.run_audio(audio),
            (Model::PerchV2(m), AIInput::Audio(audio)) => m.run_audio(audio),
            _ => panic!("wrong input type for this model architecture"),
        }
    }
}
