use crate::api::abstractions::{ProbSpace, SEGc, XYXYc};

#[derive(Clone)]
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