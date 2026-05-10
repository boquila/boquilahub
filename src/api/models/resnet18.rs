use super::*;
use crate::api::{
    abstractions::{AIOutputs, AudioProb},
    audio::AudioData,
    processing::{
        inference::inference,
        post::extract_output,
        pre::{compute_mel, mels_to_batch},
    },
};
use anyhow::{bail, Error, Result};
use ndarray::Array2;
use ort::{session::Session, value::ValueType};

const SUB_BATCH: usize = 64;

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
        let mono = if audio.channels <= 1 {
            audio.clone()
        } else {
            audio.to_mono()
        };

        let window_secs = self.audio_config.window_size as f64;
        let hop_secs = self.audio_config.stride as f64;
        let target_rate = self.audio_config.sample_rate;
        let n_fft = self.audio_config.n_fft as usize;
        let hop_length = self.audio_config.hop_length as usize;
        let n_mels = self.input_height as usize;
        let top_db = self.audio_config.top_db;

        let mut all_probs: Vec<AudioProb> = Vec::new();
        let mut batch_mels: Vec<Array2<f32>> = Vec::with_capacity(SUB_BATCH);
        let mut batch_indices: Vec<usize> = Vec::with_capacity(SUB_BATCH);

        if mono.duration() < window_secs {
            let padded = mono.padded_to(window_secs);
            let resampled = padded.resample(target_rate);
            let mel = compute_mel(&resampled, n_fft, hop_length, n_mels, top_db);
            batch_mels.push(mel);
            batch_indices.push(0);
            self.flush_batch(&batch_mels, &batch_indices, &mut all_probs);
        } else {
            for (i, window) in mono.chunks(window_secs, hop_secs).enumerate() {
                let resampled = window.resample(target_rate);
                let mel = compute_mel(&resampled, n_fft, hop_length, n_mels, top_db);
                batch_mels.push(mel);
                batch_indices.push(i);

                if batch_mels.len() == SUB_BATCH {
                    self.flush_batch(&batch_mels, &batch_indices, &mut all_probs);
                    batch_mels.clear();
                    batch_indices.clear();
                }
            }

            if !batch_mels.is_empty() {
                self.flush_batch(&batch_mels, &batch_indices, &mut all_probs);
            }
        }

        AIOutputs::AudioClassification(all_probs)
    }

    fn flush_batch(
        &self,
        batch_mels: &[Array2<f32>],
        batch_indices: &[usize],
        all_probs: &mut Vec<AudioProb>,
    ) {
        let n_mels = self.input_height as usize;
        let input = mels_to_batch(batch_mels, n_mels);
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();
        let output = extract_output(&outputs, &self.output_name);

        for (j, &logit) in output.iter().take(batch_indices.len()).enumerate() {
            let global_i = batch_indices[j];
            let prob = 1.0 / (1.0 + (-logit).exp());
            let start = global_i as f32 * self.audio_config.stride;
            let end = start + self.audio_config.window_size;
            let class_id = if prob >= self.config.confidence_threshold { 1 } else { 0 };
            all_probs.push(AudioProb {
                start,
                end,
                class_id,
                prob,
                positive: class_id == 1,
                label: self.classes[class_id as usize].clone(),
            });
        }
    }
}
