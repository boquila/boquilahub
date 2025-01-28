#![allow(dead_code)]
use super::abstractions::{xyxy_to_bbox, BBox, ProbSpace, SEGn, AI, XYXY};
use super::bq::import_bq;
use super::eps::EP;
use super::preprocessing::prepare_input;
use ndarray::{Array, Ix4, IxDyn};
use once_cell::sync::Lazy; 
// will help us manage the MODEL global variable
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};

use std::{sync::Mutex, vec};

use super::postprocessing::process_output;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}

fn default_model() -> Session {
    let (_model_metadata, data): (AI, Vec<u8>) = import_bq("models/boquilanet-gen.bq").unwrap();
    let model = Session::builder()
        .unwrap()
        .with_execution_providers([CUDAExecutionProvider::default().build()]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_memory(&data)
        .unwrap();
    return model;
}

// Lazily initialized global variables for the MODEL
static CURRENT_AI: Lazy<Mutex<AI>> = Lazy::new(|| Mutex::new(AI::default())); //
static MODEL: Lazy<Mutex<Session>> = Lazy::new(|| Mutex::new(default_model()));

fn import_model(model_data: &Vec<u8>, ep: EP) -> Session {    
    if ep.name == "CUDA" {
        let model = Session::builder()
            .unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_memory(&model_data)
            .unwrap();

        return model;
    } else {
        
        let model = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_memory(&model_data)
            // .with_execution_providers([cuda.build()]).unwrap()
            .unwrap();

        return model;
    }
}

pub fn set_model(value: String, ep: EP) {
    let (model_metadata, data): (AI, Vec<u8>) = import_bq(&value).unwrap();
    *MODEL.lock().unwrap() = import_model(&data, ep);
    *CURRENT_AI.lock().unwrap() = model_metadata.clone();
}

fn run_model(input: Array<f32, Ix4>) -> Array<f32, IxDyn> {
    let binding = MODEL.lock().unwrap();

    let outputs = binding
        .run(inputs!["images" => input.view()].unwrap())
        .unwrap();

    let predictions = outputs["output0"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();
    return predictions;
}

#[flutter_rust_bridge::frb(dart_async)]
pub fn detect(file_path: String) -> Vec<XYXY> {
    let buf = std::fs::read(file_path).unwrap_or(vec![]);

    let input_width = CURRENT_AI.lock().unwrap().input_width;
    let input_height = CURRENT_AI.lock().unwrap().input_height;

    let (input, img_width, img_height) = prepare_input(buf, input_width, input_height);
    let output = run_model(input);
    let boxes = process_output(output, img_width, img_height, input_width, input_height);
    return boxes;
}

#[flutter_rust_bridge::frb(dart_async)]
pub fn detect_bbox(file_path: String) -> Vec<BBox> {
    let data = detect(file_path);
    return xyxy_to_bbox(data, CURRENT_AI.lock().unwrap().clone());
}

// #[flutter_rust_bridge::frb(dart_async)]
// pub fn classify(file_path: String) -> ProbSpace {

// }

// #[flutter_rust_bridge::frb(dart_async)]
// pub fn segment(file_path: String) -> Vec<SEG> {

// }

enum AIOutputs {
    ObjectDetection(Vec<XYXY>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGn>),
}

enum AIFunctions {
    Classify,
    Detect,
    Segment,
}

impl AIFunctions {
    fn execute(&self, input: String) -> AIOutputs {
        match self {
            AIFunctions::Classify => todo!("{}", input),
            AIFunctions::Detect => AIOutputs::ObjectDetection(detect(input)),
            AIFunctions::Segment => todo!("{}", input),
        }
    }
}
