use std::{sync::Mutex, vec}; 
use ndarray::{Array, Ix4, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::inputs;
use once_cell::sync::Lazy; // will help us manage the MODEL global variable
// use super::abstractions::*;
use super::preprocessing::prepare_input;

use super::postprocessing::process_output;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}

// Global variables for the MODEL
static INPUT_WIDTH: Lazy<Mutex<u32>> = Lazy::new(|| Mutex::new(1024)); // Replace 256 with your desired width
static INPUT_HEIGHT: Lazy<Mutex<u32>> = Lazy::new(|| Mutex::new(1024)); // Replace 256 with your desired height

static MODEL: Lazy<Mutex<Session>> =
    Lazy::new(|| Mutex::new(import_model("models/boquilanet-gen.onnx")));

pub fn set_model(value: String,new_input_width:u32, new_input_height: u32) {
    *MODEL.lock().unwrap() = import_model(&value);
    *INPUT_WIDTH.lock().unwrap() = new_input_width;
    *INPUT_HEIGHT.lock().unwrap() = new_input_height;
}

// TODO: it should return a Vec<XYXY>, not a JSON String, this change will improve performance a bit
#[flutter_rust_bridge::frb(dart_async)] 
pub fn detect(file_path: String) -> String {
    let buf = std::fs::read(file_path).unwrap_or(vec![]);

    // Hardcoded, it shouldn't be.
    // TODO: fix
    let input_width = *INPUT_HEIGHT.lock().unwrap();
    let input_height = *INPUT_HEIGHT.lock().unwrap();
    
    let (input, img_width, img_height) = prepare_input(buf,input_width,input_height);
    let output = run_model(input);
    let boxes = process_output(output, img_width, img_height,input_width,input_height);

    return serde_json::to_string(&boxes).unwrap_or_default();
}

fn import_model(model_path: &str) -> Session {
    // let cuda = CUDAExecutionProvider::default();
    let model = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        // .with_execution_providers([cuda.build()]).unwrap()
        .commit_from_file(model_path).unwrap();

    return model;
}

// YOLO example
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
