use crate::api::{
    abstractions::{AIOutputs, Embedding, ModelConfig},
    bq::AIMetadata,
    processing::{inference::inference, pre::imgbuf_to_clip_input},
};
use anyhow::{bail, Error, Result};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

pub struct Clip {
    pub input_width: u32,
    pub input_height: u32,
    pub input_name: String,
    pub output_name: String,
    pub model_name: String,
    pub session: Session,
    pub config: ModelConfig,
}

impl Clip {
    pub fn new(
        metadata: AIMetadata,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let (input_height, input_width) = match &session.inputs()[0].dtype() {
            ValueType::Tensor { shape: dimensions, .. } => {
                (dimensions[2] as u32, dimensions[3] as u32)
            }
            _ => bail!("expected tensor input for Clip"),
        };

        let out_dims: Vec<i64> = match session.outputs()[0].dtype() {
            ValueType::Tensor { shape: dimensions, .. } => dimensions.to_vec(),
            _ => bail!("expected tensor output for Clip"),
        };

        if out_dims.len() != 2 {
            bail!("Clip output must be flat [batch_size, N], got rank {}", out_dims.len());
        }

        Ok(Clip {
            input_width,
            input_height,
            input_name: session.inputs()[0].name().to_string(),
            output_name: session.outputs()[0].name().to_string(),
            model_name: metadata.name,
            session,
            config,
        })
    }

    pub fn run_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let input = imgbuf_to_clip_input(self.input_height, self.input_width, img);
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();
        let tensor = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .unwrap()
            .into_owned();
        let raw = tensor.as_slice().expect("non-contiguous embedding tensor");

        AIOutputs::Embed(Embedding::from_raw(raw, self.model_name.clone()))
    }
}
