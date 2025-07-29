#![allow(dead_code)]
use super::abstractions::AI;
use super::bq::import_bq;
use super::eps::EP;
use super::models::Task;
use crate::api::models::processing::inference::AIOutputs;
use crate::api::models::processing::post_processing::PostProcessing;
use crate::api::models::Model;
use crate::api::{import::import_model, models::processing::pre_processing::slice_image};
use image::{ImageBuffer, Rgb};
use std::sync::{OnceLock, RwLock};

static CURRENT_AI: OnceLock<RwLock<Model>> = OnceLock::new();
static CURRENT_AI2: OnceLock<RwLock<Option<Model>>> = OnceLock::new();

fn clear_current_ai2_simple() {
    let rw_lock = CURRENT_AI2.get_or_init(|| RwLock::new(None));
    let mut guard = rw_lock.write().unwrap();
    *guard = None;
}

pub fn set_model(value: &String, ep: &EP) {
    let (model_metadata, data): (AI, Vec<u8>) = import_bq(value).unwrap();
    let session = import_model(&data, ep);
    let post: Vec<PostProcessing> = model_metadata
        .post_processing
        .iter()
        .map(|s| PostProcessing::from(s.as_str()))
        .filter(|t| !matches!(t, PostProcessing::None))
        .collect();
    let aimodel: Model = Model::new(
        model_metadata.classes,
        0.45,
        0.5,
        Task::from(model_metadata.task.as_str()),
        post,
        session,
        model_metadata.architecture,
    );
    if CURRENT_AI.get().is_some() {
        *CURRENT_AI.get().unwrap().write().unwrap() = aimodel;
    } else {
        let _ = CURRENT_AI.set(RwLock::new(aimodel));
    }
}

pub fn set_model2(value: &String, ep: &EP) {
    let (model_metadata, data): (AI, Vec<u8>) = import_bq(value).unwrap();
    let session = import_model(&data, ep);
    let post: Vec<PostProcessing> = model_metadata
        .post_processing
        .iter()
        .map(|s| PostProcessing::from(s.as_str()))
        .filter(|t| !matches!(t, PostProcessing::None))
        .collect();

    let aimodel: Model = Model::new(
        model_metadata.classes,
        0.8,
        0.0,
        Task::from(model_metadata.task.as_str()),
        post,
        session,
        model_metadata.architecture,
    );

    if CURRENT_AI2.get().is_some() {
        *CURRENT_AI2.get().unwrap().write().unwrap() = Some(aimodel);
    } else {
        let _ = CURRENT_AI2.set(RwLock::new(Some(aimodel)));
    }
}

#[inline(always)]
pub fn process_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, two_steps: bool) -> AIOutputs {
    let mut outputs: AIOutputs = CURRENT_AI.get().unwrap().read().unwrap().run(&img);

    if !two_steps {
        return outputs;
    }

    if let Some(ai2) = CURRENT_AI2.get() {
        let ai2_guard = ai2.read().unwrap();
        let ai2_ref = ai2_guard.as_ref().unwrap();

        match &mut outputs {
            AIOutputs::ObjectDetection(detections) => {
                for xyxyc in detections.iter_mut() {
                    let sliced_img = slice_image(img, &xyxyc.xyxy);
                    let cls_output = ai2_ref.run(&sliced_img);

                    if let AIOutputs::Classification(prob_space) = cls_output {
                        xyxyc.extra_cls = Some(prob_space);
                    }
                }
            }
            AIOutputs::Segmentation(segmentations) => {
                for segc in segmentations {
                    let xyxyc = &mut segc.bbox;
                    let sliced_img = slice_image(img, &xyxyc.xyxy);
                    let cls_output = ai2_ref.run(&sliced_img);

                    if let AIOutputs::Classification(prob_space) = cls_output {
                        xyxyc.extra_cls = Some(prob_space);
                    }
                }
            }
            _ => {
                println!("Unsupported AIOutputs variant.");
            }
        }
    }

    return outputs;
}
