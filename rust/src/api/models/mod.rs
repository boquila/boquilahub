pub mod yolo;
use super::{abstractions::*, bq::import_bq};
use ndarray::{Array, Ix4, IxDyn};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
};
pub use yolo::Yolo;

pub struct AIModel {
    pub name: String,
    pub description: String,
    pub version: f32,
    pub classes: Vec<String>,
    pub model: Architecture,
    pub session: Session,
}

impl AIModel {
    // Basic constructor
    pub fn new(
        name: String,
        description: String,
        version: f32,
        classes: Vec<String>,
        model: Architecture,
        session: Session,
    ) -> Self {
        AIModel {
            name,
            description,
            version,
            classes,
            model,
            session,
        }
    }

    pub fn default() -> Self {
        let (_, data) = import_bq("models/boquilanet-gen.bq").unwrap();
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_memory(&data)
            .unwrap();

        AIModel::new(
            "boquilanet-gen".to_string(),
            "Generic animal detection".to_string(),
            0.1,
            vec!["animal".to_string()],
            Architecture::Yolo(Yolo::new(1024, 1024, 0.45, 0.5, 1, 0, Task::Detect)),
            session,
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

    pub fn run_model(&self, input: &Array<f32, Ix4>) -> Array<f32, IxDyn> {
        let outputs = self
            .session
            .run(inputs!["images" => input.view()].unwrap())
            .unwrap();

        let predictions = outputs["output0"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .t()
            .into_owned();
        return predictions;
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

impl From<&str> for PostProcessing {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "NMS" => PostProcessing::NMS,
            _ => PostProcessing::NMS,
        }
    }
}

pub enum Architecture {
    Yolo(Yolo),
}

enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGn>),
}
