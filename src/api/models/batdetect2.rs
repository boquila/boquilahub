use super::*;
use crate::api::{
    abstractions::{AIOutputs, Prob, XYXY, XYXYc},
    audio::AudioData,
};
use anyhow::{bail, Error, Result};
use ndarray::Array2;
use ort::session::Session;

/// UK bat-call detector; the ONNX graph bakes in preprocessing and box decoding (waveform in, detections out).
pub struct BatDetect2 {
    pub classes: Vec<String>,
    pub session: Session,
    pub config: ModelConfig,
    pub audio_config: AudioConfig,
    pub input_name: String,
    pub window_samples: usize,
    pub stride_samples: usize,
}

impl BatDetect2 {
    pub fn new(metadata: AIMetadata, session: Session, config: ModelConfig) -> Result<Self, Error> {
        let Some(audio_config) = metadata.audio_config.clone() else {
            bail!("BatDetect2 requires audio_config in metadata");
        };
        if metadata.classes.is_empty() {
            bail!("BatDetect2 requires non-empty classes");
        }

        let sr = audio_config.sample_rate as f64;
        let window_samples = (audio_config.window_size as f64 * sr).round() as usize;
        let stride_samples = ((audio_config.stride as f64 * sr).round() as usize).max(1);

        Ok(Self {
            classes: metadata.classes,
            input_name: session.inputs()[0].name().to_string(),
            session,
            config,
            audio_config,
            window_samples,
            stride_samples,
        })
    }

    pub fn run_audio(&self, audio: &AudioData) -> AIOutputs {
        let prepared = audio.to_mono().resample(self.audio_config.sample_rate);
        let total = prepared.samples.len();
        let sample_rate = self.audio_config.sample_rate as f32;
        let mut boxes: Vec<XYXYc> = Vec::new();

        for start in (0..total).step_by(self.stride_samples) {
            let end = (start + self.window_samples).min(total);
            let mut window = prepared.samples[start..end].to_vec();
            window.resize(self.window_samples, 0.0);
            let input = Array2::from_shape_vec((1, self.window_samples), window).unwrap();

            #[allow(mutable_transmutes)]
            let session: &mut Session = unsafe { std::mem::transmute(&self.session) };
            let input = ort::value::TensorRef::from_array_view(input.view()).unwrap();
            let outputs = session.run(ort::inputs![&*self.input_name => input]).unwrap();

            let scores = outputs["scores"].try_extract_array::<f32>().unwrap();
            let start_time = outputs["start_time"].try_extract_array::<f32>().unwrap();
            let end_time = outputs["end_time"].try_extract_array::<f32>().unwrap();
            let low_freq = outputs["low_freq"].try_extract_array::<f32>().unwrap();
            let high_freq = outputs["high_freq"].try_extract_array::<f32>().unwrap();
            let class_id = outputs["class_id"].try_extract_array::<i64>().unwrap();
            let class_scores = outputs["class_scores"].try_extract_array::<f32>().unwrap();

            let offset = start as f32 / sample_rate;
            for i in 0..scores.shape()[1] {
                let score = scores[[0, i]];
                // Sorted by descending score, so the first sub-threshold row ends it.
                if score < self.config.confidence_threshold {
                    break;
                }
                let cid = class_id[[0, i]].max(0) as usize;
                let extra_cls = self
                    .classes
                    .iter()
                    .enumerate()
                    .map(|(c, name)| Prob::new(name.clone(), class_scores[[0, i, c]], c as u32))
                    .collect();
                boxes.push(XYXYc {
                    xyxy: XYXY::new(
                        start_time[[0, i]] + offset,
                        low_freq[[0, i]],
                        end_time[[0, i]] + offset,
                        high_freq[[0, i]],
                        score,
                        cid as u32,
                    ),
                    label: self.classes.get(cid).cloned().unwrap_or_default(),
                    extra_cls: Some(extra_cls),
                });
            }
        }

        AIOutputs::ObjectDetection(boxes)
    }
}
