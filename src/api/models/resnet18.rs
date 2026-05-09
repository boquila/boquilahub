use super::*;
use crate::api::{
    abstractions::{AIOutputs, AudioProb},
    audio::AudioData,
    processing::{
        inference::inference,
        post::extract_output,
        pre::audio_to_input_array,
    },
};
use anyhow::{bail, Error, Result};
use ort::{session::Session, value::ValueType};

#[derive(Debug)]
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
    pub audio_config: AudioConfig
}

impl ResNet18 {
    pub fn new(
        classes: Vec<String>,
        task: Task,
        post_processing: Vec<PostProcessing>,
        session: Session,
        config: ModelConfig,
        audio_config: AudioConfig
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
            audio_config
        })
    }
}

impl ResNet18 {
    pub fn run_audio(&self, audio: &AudioData) -> AIOutputs {
        let audio = audio.resample(self.audio_config.sample_rate);
        let input = audio_to_input_array(
            &audio,
            self.audio_config.n_fft as usize,
            self.audio_config.hop_length as usize,
            self.input_height as usize,
            self.audio_config.top_db,
            self.audio_config.window_size as f64,
            self.audio_config.stride as f64,
        );

        let outputs = inference(&self.session, &input, &self.input_name);
        let output = extract_output(&outputs, &self.output_name);

        let audio_probs: Vec<AudioProb> = output
            .iter()
            .enumerate()
            .map(|(i, &logit)| {
                let prob = 1.0 / (1.0 + (-logit).exp());
                let start = i as f32 * self.audio_config.stride;
                let end = start + self.audio_config.window_size;
                let class_id = if prob >= self.config.confidence_threshold { 1 } else { 0 };
                AudioProb {
                    start,
                    end,
                    class_id,
                    prob,
                    label: self.classes[class_id as usize].clone(),
                }
            })
            .collect();

        AIOutputs::AudioClassification(audio_probs)
    }
}
