#![allow(dead_code)]
use super::abstractions::{XYXY,ProbSpace,SEGn};
use super::preprocessing::prepare_input;
use ndarray::{Array, Ix4, IxDyn};
use once_cell::sync::Lazy; // will help us manage the MODEL global variable
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::{sync::Mutex, vec};

use super::postprocessing::process_output;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}

// Lazily initialized global variables for the MODEL
static INPUT_WIDTH: Lazy<Mutex<u32>> = Lazy::new(|| Mutex::new(1024)); // Replace 256 with your desired width
static INPUT_HEIGHT: Lazy<Mutex<u32>> = Lazy::new(|| Mutex::new(1024)); // Replace 256 with your desired height
static MODEL: Lazy<Mutex<Session>> =
    Lazy::new(|| Mutex::new(import_model("models/boquilanet-gen.onnx")));

fn import_model(model_path: &str) -> Session {
    // let cuda = CUDAExecutionProvider::default();
    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        // .with_execution_providers([cuda.build()]).unwrap()
        .commit_from_file(model_path)
        .unwrap();

    return model;
}

pub fn set_model(value: String, new_input_width: u32, new_input_height: u32) {
    *MODEL.lock().unwrap() = import_model(&value);
    *INPUT_WIDTH.lock().unwrap() = new_input_width;
    *INPUT_HEIGHT.lock().unwrap() = new_input_height;
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

    let input_width = *INPUT_HEIGHT.lock().unwrap();
    let input_height = *INPUT_HEIGHT.lock().unwrap();

    let (input, img_width, img_height) = prepare_input(buf, input_width, input_height);
    let output = run_model(input);
    let boxes = process_output(output, img_width, img_height, input_width, input_height);
    return boxes;
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
    Segmentation(Vec<SEGn>)
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