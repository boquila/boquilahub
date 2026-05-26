use crate::api::{
    abstractions::{AIOutputs, Embedding, ModelConfig},
    bq::AIMetadata,
    processing::{
        inference::inference,
        post::extract_output,
        pre::{imgbuf_to_input_array, TensorFormat},
    },
};
use anyhow::{bail, Error, Result};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

pub struct Clip {
    pub input_width: u32,
    pub input_height: u32,
    pub input_name: String,
    pub output_name: String,
    pub embedding_dim: usize,
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
        let (input_height, input_width) = match &session.inputs[0].input_type {
            ValueType::Tensor { dimensions, .. } => {
                // Expected NCHW: [batch, 3, H, W]
                (dimensions[2] as u32, dimensions[3] as u32)
            }
            _ => bail!("expected tensor input for Clip"),
        };

        let input_name = session.inputs[0].name.clone();

        let embedding_dim = match &session.outputs[0].output_type {
            ValueType::Tensor { dimensions, .. } => {
                // Expected [1, D]; take the last dim so [D] or [1, D] both work.
                *dimensions.last().unwrap_or(&0) as usize
            }
            _ => bail!("expected tensor output for Clip"),
        };

        let output_name = session.outputs[0].name.clone();

        Ok(Clip {
            input_width,
            input_height,
            input_name,
            output_name,
            embedding_dim,
            model_name: metadata.name,
            session,
            config,
        })
    }

    pub fn run_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (input, _w, _h) = imgbuf_to_input_array(
            1,
            3,
            self.input_height,
            self.input_width,
            img,
            &TensorFormat::NCHW,
        );
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();
        let output = extract_output(&outputs, &self.output_name);
        let values: Vec<f32> = output.iter().copied().collect();
        AIOutputs::Embed(Embedding {
            values,
            model: self.model_name.clone(),
        })
    }
}
