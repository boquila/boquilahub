use crate::api::{
    abstractions::{AIOutputs, Embedding, ModelConfig},
    bq::AIMetadata,
    processing::{
        inference::inference,
        pre::{imgbuf_to_input_array, TensorFormat},
    },
};
use anyhow::{bail, Error, Result};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

const INPUT_SIZE: u32 = 448;

pub struct Dinov3 {
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
        let output = &session.outputs()[0];

        let out_dims: Vec<i64> = match &output.dtype() {
            ValueType::Tensor { shape: dimensions, .. } => dimensions.to_vec(),
            _ => bail!("dinov3 output must be a tensor"),
        };

        if out_dims.len() != 2 {
            bail!("dinov3 output must be flat [batch_size, N], got rank {}", out_dims.len());
        }

        Ok(Dinov3 {
            input_name: session.inputs()[0].name().to_string(),
            output_name: output.name().to_string(),
            model_name: metadata.name,
            session,
            config,
        })
    }

    pub fn run_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (input, _w, _h) = imgbuf_to_input_array(
            1,
            3,
            INPUT_SIZE,
            INPUT_SIZE,
            img,
            &TensorFormat::NCHW,
        );
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();
        let tensor = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .unwrap()
            .into_owned();
        let raw = tensor.as_slice().expect("non-contiguous dinov3 tensor");

        AIOutputs::Embed(Embedding::from_raw(raw, self.model_name.clone()))
    }
}
