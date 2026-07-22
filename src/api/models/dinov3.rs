use crate::api::{
    abstractions::{AIOutputs, Embedding, ModelConfig},
    bq::AIMetadata,
    processing::{inference::inference, pre::imgbuf_to_dinov3_input},
};
use anyhow::{bail, Error, Result};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

const DEFAULT_INPUT_SIZE: u32 = 224;

pub struct Dinov3 {
    pub input_width: u32,
    pub input_height: u32,
    pub input_name: String,
    pub output_name: String,
    pub model_name: String,
    pub session: Session,
    pub config: ModelConfig,
}

impl Dinov3 {
    pub fn new(
        metadata: AIMetadata,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let (input_height, input_width) = match &session.inputs()[0].dtype() {
            ValueType::Tensor { shape: dimensions, .. } => {
                let resolve = |d: i64| if d > 0 { d as u32 } else { DEFAULT_INPUT_SIZE };
                (resolve(dimensions[2]), resolve(dimensions[3]))
            }
            _ => bail!("expected tensor input for Dinov3"),
        };

        let output = &session.outputs()[0];
        let out_dims: Vec<i64> = match &output.dtype() {
            ValueType::Tensor { shape: dimensions, .. } => dimensions.to_vec(),
            _ => bail!("dinov3 output must be a tensor"),
        };

        if out_dims.len() != 2 {
            bail!("dinov3 output must be flat [batch_size, N], got rank {}", out_dims.len());
        }

        Ok(Dinov3 {
            input_width,
            input_height,
            input_name: session.inputs()[0].name().to_string(),
            output_name: output.name().to_string(),
            model_name: metadata.name,
            session,
            config,
        })
    }

    pub fn run_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let input = imgbuf_to_dinov3_input(self.input_height, self.input_width, img);
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();
        let tensor = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .unwrap()
            .into_owned();
        let raw = tensor.as_slice().expect("non-contiguous dinov3 tensor");

        AIOutputs::Embed(Embedding::from_raw(raw, self.model_name.clone()))
    }
}
