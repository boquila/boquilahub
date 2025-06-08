#![allow(dead_code)]
use super::abstractions::{XYXYc, AI};
use super::bq::import_bq;
use super::eps::EP;
use super::models::{AIOutputs, Task, Yolo};
use image::{open, ImageBuffer, Rgb};
use once_cell::sync::Lazy;
use ort::session::builder::GraphOptimizationLevel;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use std::sync::Mutex;

pub fn init_app() {
    std::fs::create_dir_all("output_feed").unwrap();
    std::fs::create_dir_all("export").unwrap();
}

// Lazily initialized global variables for the MODEL
static CURRENT_AI: Lazy<Mutex<Yolo>> = Lazy::new(|| Mutex::new(Yolo::default())); //

fn import_model(model_data: &Vec<u8>, ep: EP) -> Session {
    let mut builder = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    if ep.name == "CUDA" {
        builder = builder
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .unwrap();
    }

    builder.commit_from_memory(model_data).unwrap()
}

pub fn set_model(value: &String, ep: EP) {
    let (model_metadata, data): (AI, Vec<u8>) = import_bq(value).unwrap();

    let len = model_metadata.classes.len() as u32;
    let aimodel = Yolo::new(
        model_metadata.name,
        model_metadata.description,
        model_metadata.version,
        model_metadata.classes,
        model_metadata.input_width,
        model_metadata.input_height,
        0.45,
        0.5,
        len,
        0,
        Task::from(model_metadata.task.as_str()),
        import_model(&data, ep),
    );

    *CURRENT_AI.lock().unwrap() = aimodel;
}

#[inline(always)]
pub fn detect_bbox_from_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<XYXYc> {
    match CURRENT_AI.lock().unwrap().run(&img) {
        AIOutputs::ObjectDetection(boxes) => return boxes,
        _ => {
            panic!("Expected ObjectDetection output");
        }
    }
}