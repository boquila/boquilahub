#![allow(dead_code)]
use super::abstractions::AI;
use super::bq::import_bq;
use super::eps::EP;
use super::models::{Task, Yolo};
use crate::api::models::processing::inference::AIOutputs;
use image::{ImageBuffer, Rgb};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::ValueType;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use std::sync::{OnceLock, RwLock};

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
    let session = import_model(&data, ep);
    let input_shape = match &session.inputs[0].input_type {
        ValueType::Tensor { dimensions, .. } => dimensions,
        _ => {
            panic!("Not supported");
        }
    };
    let _batch_size = input_shape[0];
    let _input_depth = input_shape[1];
    let input_width = input_shape[2];
    let input_height = input_shape[3];

    let output_shape = match &session.outputs[0].output_type {
        ValueType::Tensor { dimensions, .. } => dimensions,
        _ => {
            panic!("Not supported");
        }
    };

    let output_width = output_shape[1];
    let output_height = output_shape[2];

    // at some point we might need this for accurate segmentation:
    let (num_masks, masks_width, masks_height) = if let Some(output) = session.outputs.get(1) {
        match &output.output_type {
            ValueType::Tensor { dimensions, .. } => {
                let num_masks = dimensions[1];
                let masks_width = dimensions[2];
                let masks_height = dimensions[3];
                (num_masks, masks_width, masks_height)
            }
            _ => {
                panic!("Not supported output type at index 1");
            }
        }
    } else {
        (0, 0, 0)
    };    

    let len = model_metadata.classes.len() as u32;
    let aimodel = Yolo::new(
        model_metadata.classes,
        input_width as u32,
        input_height as u32,
        output_width as u32,
        output_height as u32,
        0.45,
        0.5,
        len,
        num_masks as u32,
        masks_height as u32,
        masks_width as u32,
        Task::from(model_metadata.task.as_str()),
        session,
    );
    if CURRENT_AI.get().is_some() {
        *CURRENT_AI.get().unwrap().write().unwrap() = aimodel;
    } else {
        let _ = CURRENT_AI.set(RwLock::new(aimodel));
    }
    
}

#[inline(always)]
pub fn detect_bbox_from_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
    CURRENT_AI.get().unwrap().read().unwrap().run(&img)
}
