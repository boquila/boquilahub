#![allow(dead_code)]
use crate::api::models::processing::inference::AIOutputs;
use super::abstractions::{AI};
use super::bq::import_bq;
use super::eps::EP;
use super::models::{Task, Yolo};
use egui::mutex::RwLock;
use image::{ImageBuffer, Rgb};
use ort::session::builder::GraphOptimizationLevel;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use std::sync::OnceLock;

pub fn init_app() {
    std::fs::create_dir_all("output_feed").unwrap();
    std::fs::create_dir_all("export").unwrap();
}

// Lazily initialized global variables for the MODEL
static CURRENT_AI: OnceLock<RwLock<Yolo>> = OnceLock::new();

fn import_model(model_data: &Vec<u8>, ep: &EP) -> Session {
    let mut builder = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    if ep.name == "CUDA" {
        builder = builder
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .unwrap();
    }

    builder.commit_from_memory(model_data).unwrap()
}

pub fn set_model(value: &String, ep: &EP) {
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
    if CURRENT_AI.get().is_some() {
        *CURRENT_AI.get().unwrap().write() = aimodel;
    } else {
        let _ = CURRENT_AI.set(RwLock::new(aimodel));
    }
}

#[inline(always)]
pub fn detect_bbox_from_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
    CURRENT_AI.get().unwrap().read().run(&img)
    // match CURRENT_AI.get().unwrap().read().run(&img) {
    //     AIOutputs::ObjectDetection(boxes) => return boxes,
    //     _ => {
    //         panic!("Expected ObjectDetection output");
    //     }
    // }
}