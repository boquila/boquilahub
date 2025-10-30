#![allow(dead_code)]
use super::abstractions::{AIOutputs, ModelConfig, AI};
use super::bq::import_bq;
use super::eps::EP;
use super::import::import_model;
use super::models::Model;
use super::models::Task;
use super::processing::post::PostProcessing;
use super::processing::pre::slice_image;
use eframe::Result;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

pub static GEOFENCE_DATA: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

pub fn init_geofence_data() -> Result<(), Box<dyn std::error::Error>> {
    // Check if already initialized, return early if so
    if GEOFENCE_DATA.get().is_some() {
        return Ok(());
    }

    let json_content = std::fs::read_to_string("assets/geofence.json")?;
    let geofence_map: HashMap<String, Vec<String>> = serde_json::from_str(&json_content)?;
    GEOFENCE_DATA
        .set(geofence_map)
        .map_err(|_| "Failed to initialize")?;
    Ok(())
}

pub static CURRENT_AI: OnceLock<RwLock<Model>> = OnceLock::new();
static CURRENT_AI2: OnceLock<RwLock<Option<Model>>> = OnceLock::new();

pub fn clear_current_ai2_simple() {
    let rw_lock = CURRENT_AI2.get_or_init(|| RwLock::new(None));
    let mut guard = rw_lock.write().unwrap();
    *guard = None;
}

pub fn set_model(value: &String, ep: &EP, config: Option<ModelConfig>) -> Result<(), String> {
    let config = config.unwrap_or_default();

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
        Task::from(model_metadata.task.as_str()),
        post,
        session,
        model_metadata.architecture,
        config,
    )?;
    if CURRENT_AI.get().is_some() {
        *CURRENT_AI.get().unwrap().write().unwrap() = aimodel;
    } else {
        let _ = CURRENT_AI.set(RwLock::new(aimodel));
    }
    Ok(())
}

pub fn set_model2(value: &String, ep: &EP, config: Option<ModelConfig>) -> Result<(), String> {
    let config = config.unwrap_or_default();

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
        Task::from(model_metadata.task.as_str()),
        post,
        session,
        model_metadata.architecture,
        config,
    )?;

    if CURRENT_AI2.get().is_some() {
        *CURRENT_AI2.get().unwrap().write().unwrap() = Some(aimodel);
    } else {
        let _ = CURRENT_AI2.set(RwLock::new(Some(aimodel)));
    }
    Ok(())
}

#[inline(always)]
pub fn process_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
    let mut outputs: AIOutputs = CURRENT_AI.get().unwrap().read().unwrap().run(&img);
    process_with_ai2(&mut outputs, img);
    return outputs;
}

fn process_with_ai2(outputs: &mut AIOutputs, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Option<()> {
    let ai2 = CURRENT_AI2.get()?;
    let ai2_guard = ai2.read().ok()?;
    let ai2_ref = ai2_guard.as_ref()?;

    match outputs {
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
        _ => {}
    }

    Some(())
}

pub fn update_config(new_config: ModelConfig) {
    let ai = CURRENT_AI.get().unwrap();
    let mut model = ai.write().unwrap();
    *model.config_mut() = new_config;
}

pub fn update_config2(new_config: ModelConfig) {
    let ai = CURRENT_AI2.get().unwrap();
    let mut model_opt = ai.write().unwrap();
    if let Some(ref mut model) = model_opt.as_mut() {
        *model.config_mut() = new_config;
    }
}
