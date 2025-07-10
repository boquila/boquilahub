#![allow(dead_code)]
use super::abstractions::AI;
use super::bq::import_bq;
use super::eps::EP;
use super::models::{Task, Yolo};
use crate::api::import::import_model;
use crate::api::models::processing::inference::AIOutputs;
use crate::api::models::processing::post_processing::PostProcessing;
use crate::api::models::ModelTrait;
use image::{ImageBuffer, Rgb};
use std::sync::{OnceLock, RwLock};

static CURRENT_AI: OnceLock<RwLock<Yolo>> = OnceLock::new();

pub fn set_model(value: &String, ep: &EP) {
    let (model_metadata, data): (AI, Vec<u8>) = import_bq(value).unwrap();
    let session = import_model(&data, ep);
    let post: Vec<PostProcessing> = model_metadata
        .post_processing
        .iter()
        .map(|s| PostProcessing::from(s.as_str()))
        .filter(|t| !matches!(t, PostProcessing::None))
        .collect();
    let aimodel = Yolo::new(
        model_metadata.classes,
        0.45,
        0.5,
        Task::from(model_metadata.task.as_str()),
        post,
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
