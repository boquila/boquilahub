use crate::api::abstractions::{ProbSpace, SEGc, XYXYc};
use ndarray::{Array, Ix4};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub enum AIOutputs {
    ObjectDetection(Vec<XYXYc>),
    Classification(ProbSpace),
    Segmentation(Vec<SEGc>),
}

impl AIOutputs {
    pub fn is_empty(&self) -> bool {
        match self {
            AIOutputs::ObjectDetection(bboxes) => bboxes.is_empty(),
            AIOutputs::Classification(prob_space) => prob_space.classes.is_empty(),
            AIOutputs::Segmentation(segments) => segments.is_empty(),
        }
    }
}

pub fn inference<'a>(
    session: &'a ort::session::Session,
    input: &'a Array<f32, Ix4>,
    b: &'static str,
) -> ort::session::SessionOutputs<'a, 'a> {
    return session
        .run(ort::inputs![b => input.view()].unwrap())
        .unwrap();
}
