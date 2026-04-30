use super::*;
use crate::api::{
    abstractions::{AIOutputs, ProbSpace},
    audio::AudioData,
    processing::{
        inference::inference,
        post::extract_output,
        pre::audio_to_input_array,
    },
};
use anyhow::{bail, Error, Result};
use ort::{session::Session, value::ValueType};

pub struct ResNet18 {
    pub classes: Vec<String>,

    // Input Tensor
    pub batch_size: i32,   // number of windows/clips
    pub channel: u32,      //  number of channel, 1 for single channel mel spectrogram
    pub input_height: u32, // number of mel freq bins
    pub input_width: i32,  // time steps, time frames, (width of the spectrogram)
    pub input_name: String,
    // Output Tensor
    pub output_width: i32,
    pub output_height: u32,
    pub output_name: String,
    pub task: Task,
    pub post_processing: Vec<PostProcessing>,
    pub session: Session,
    pub config: ModelConfig,
}

impl ModelTrait for ResNet18 {
    fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let (batch_size, channel, input_height, input_width) = match &session.inputs[0].input_type {
            ValueType::Tensor { dimensions, .. } => (
                dimensions[0] as i32,
                dimensions[1] as u32,
                dimensions[2] as u32,
                dimensions[3] as i32,
            ),
            _ => {
                bail!("expected tensor input for ResNet18");
            }
        };

        let input_name = session.inputs[0].name.clone();

        let (output_width, output_height) = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => (dimensions[0] as i32, dimensions[1] as u32),
            _ => {
                bail!("expected tensor output for ResNet18");
            }
        };

        let output_name: String = session.outputs[0].name.clone();

        Ok(ResNet18 {
            classes,
            batch_size,
            channel,
            input_width,
            input_height,
            input_name,
            output_width,
            output_height,
            output_name,
            task,
            post_processing,
            session,
            config,
        })
    }
}

impl ResNet18 {
    pub fn run_audio(&self, audio: &AudioData) -> AIOutputs {
        // HARDCODED PREPROCESSING PARAMS FROM MD_AUDIOBIRDS_V1 SPEC
        // TODO: PULL FROM MODEL METADATA (AI.n_fft, AI.hop_length, AI.n_mels, AI.top_db,
        //       AI.window_size, AI.stride) WHEN THE IMPORT PIPELINE PASSES THEM THROUGH
        let input = audio_to_input_array(audio, 2048, 512, 224, 80.0, 5.0, 1.0);

        let outputs = inference(&self.session, &input, &self.input_name);
        let output = extract_output(&outputs, &self.output_name);

        // output shape is [batch, 1]; take the highest probability across windows (OR logic)
        let max_logit = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let prob = 1.0 / (1.0 + (-max_logit).exp());

        let classes = if self.classes.is_empty() {
            vec!["Birds".to_string()]
        } else {
            self.classes.clone()
        };

        AIOutputs::Classification(ProbSpace::new(classes, vec![prob], vec![0]))
    }
}
