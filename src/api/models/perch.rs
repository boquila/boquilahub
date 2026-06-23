use super::*;
use crate::api::{
    abstractions::{AIOutputs, AudioProb, Prob, ProbSugar},
    audio::AudioData,
};
use anyhow::{bail, Error, Result};
use ndarray::Array2;
use ort::session::Session;

const SUB_BATCH: usize = 8;

pub struct PerchV2 {
    pub classes: Vec<String>,
    pub session: Session,
    pub config: ModelConfig,
    pub audio_config: AudioConfig,
    pub input_name: String,
    pub label_output_name: String,
    pub window_samples: usize,
    pub stride_samples: usize,
}

impl PerchV2 {
    pub fn new(metadata: AIMetadata, session: Session, config: ModelConfig) -> Result<Self, Error> {
        let Some(audio_config) = metadata.audio_config.clone() else {
            bail!("PerchV2 requires audio_config in metadata");
        };
        if metadata.classes.is_empty() {
            bail!("PerchV2 requires non-empty classes");
        }

        // Perch has 4 output heads (embedding, spatial_embedding, spectrogram,
        // label). outputs[0] would land on "embedding" — wrong tensor.
        let label_output_name = session
            .outputs()
            .iter()
            .find(|o| o.name() == "label")
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| session.outputs()[0].name().to_string());

        let sr = audio_config.sample_rate as f64;
        let window_samples = (audio_config.window_size as f64 * sr).round() as usize;
        let stride_samples = ((audio_config.stride as f64 * sr).round() as usize).max(1);

        Ok(Self {
            classes: metadata.classes,
            input_name: session.inputs()[0].name().to_string(),
            session,
            config,
            audio_config,
            label_output_name,
            window_samples,
            stride_samples,
        })
    }

    pub fn run_audio(&self, audio: &AudioData) -> AIOutputs {
        let prepared = audio.to_mono().resample(self.audio_config.sample_rate);
        let total = prepared.samples.len();
        let starts: Vec<usize> = (0..total).step_by(self.stride_samples).collect();
        let mut out = Vec::with_capacity(starts.len());

        for batch in starts.chunks(SUB_BATCH) {
            let mut data = Vec::with_capacity(batch.len() * self.window_samples);
            for &start in batch {
                let end = (start + self.window_samples).min(total);
                data.extend_from_slice(&prepared.samples[start..end]);
                data.resize(data.len() + (start + self.window_samples - end), 0.0);
            }
            let input = Array2::from_shape_vec((batch.len(), self.window_samples), data).unwrap();
            #[allow(mutable_transmutes)]
            let session: &mut Session = unsafe { std::mem::transmute(&self.session) };
            let input = ort::value::TensorRef::from_array_view(input.view()).unwrap();
            let outputs = session
                .run(ort::inputs![&*self.input_name => input])
                .unwrap();
            let logits = outputs[&*self.label_output_name]
                .try_extract_array::<f32>()
                .unwrap()
                .into_owned();

            for (i, &start) in batch.iter().enumerate() {
                let mut probs: Vec<Prob> = self.classes.iter().enumerate()
                    .map(|(c, label)| Prob::new(label.clone(), logits[[i, c]], c as u32))
                    .collect();
                probs.logits_to_probs();
                let top = probs.into_iter()
                    .max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap())
                    .unwrap();
                let start_s = start as f32 / self.audio_config.sample_rate as f32;
                out.push(AudioProb {
                    start: start_s,
                    end: start_s + self.audio_config.window_size,
                    prediction: top,
                });
            }
        }

        AIOutputs::AudioClassification(out)
    }
}
