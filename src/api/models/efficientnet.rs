use crate::api::{
    abstractions::{AIOutputs, ProbSpace}, inference::init_geofence_data, models::{ModelTrait, Task}, processing::{
        inference::inference,
        post_processing::{
            apply_geofence_filter, apply_label_rollup, extract_output, process_class_output, transform_logits_to_probs, PostProcessing
        },
        pre_processing::{imgbuf_to_input_array, TensorFormat},
    }
};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

pub struct EfficientNetV2 {
    pub classes: Vec<String>,
    batch_size: i32,
    input_depth: u32, // 3, RGB or similar
    input_width: u32,
    input_height: u32,
    input_name: String,
    output_width: u32,
    output_height: u32,
    output_name: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
}

impl ModelTrait for EfficientNetV2 {
    fn new(
        classes: Vec<String>,
        confidence_threshold: f32,
        nms_threshold: f32,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
    ) -> Self {
        let (batch_size, input_width, input_height, input_depth) =
            match &session.inputs[0].input_type {
                ValueType::Tensor { dimensions, .. } => (
                    dimensions[0] as i32,
                    dimensions[1] as u32,
                    dimensions[2] as u32,
                    dimensions[3] as u32,
                ),
                _ => {
                    panic!("Not supported");
                }
            };

        let input_name = session.inputs[0].name.clone();

        let (output_width, output_height) = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => (dimensions[0] as u32, dimensions[1] as u32),
            _ => {
                panic!("Not supported");
            }
        };

        let output_name: String = session.outputs[0].name.clone();

        if post_processing.contains(&PostProcessing::GeoFence) {
            let _ = init_geofence_data();
        }

        EfficientNetV2 {
            classes,
            batch_size,
            input_depth,
            input_width,
            input_height,
            input_name,
            output_width,
            output_height,
            output_name,
            confidence_threshold,
            nms_threshold,
            task,
            post_processing,
            session,
        }
    }
    fn run(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (input, _img_width, _img_height) = imgbuf_to_input_array(
            1,
            3,
            self.input_height,
            self.input_width,
            img,
            TensorFormat::NHWC,
        );
        let outputs = inference(&self.session, &input, &self.input_name);
        let output = extract_output(&outputs, &self.output_name);
        let mut probs: ProbSpace =
            process_class_output(-1000.0, &self.classes, &output);

        // Usage in your original code:
        if self.post_processing.contains(&PostProcessing::GeoFence) {
            apply_geofence_filter(&mut probs, &crate::api::inference::GEOFENCE_DATA.get().unwrap(), "CHL");
            transform_logits_to_probs(&mut probs);
            apply_label_rollup(&mut probs, self.confidence_threshold);
        }
        return AIOutputs::Classification(probs);
    }
}
