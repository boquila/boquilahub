use crate::api::{
    abstractions::{AIOutputs, Embedding, ModelConfig},
    bq::AIMetadata,
    processing::{
        inference::inference,
        pre::{imgbuf_to_input_array, TensorFormat},
    },
};
use anyhow::{bail, Context, Error, Result};
use image::{ImageBuffer, Rgb};
use ort::{session::Session, value::ValueType};

const INPUT_SIZE: u32 = 448;
const PATCH_SIZE: u32 = 16;
/// CLS (1) + register tokens (4).
const SKIP_TOKENS: usize = 5;
const OUTPUT_NAME: &str = "last_hidden_state";

pub struct Dinov3 {
    pub input_name: String,
    pub output_name: String,
    pub grid_side: u32,
    pub embed_dim: u32,
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
        let input_name = session.inputs()[0].name().to_string();

        let output = &session
            .outputs()
            .iter()
            .find(|o| o.name() == OUTPUT_NAME)
            .with_context(|| format!("missing `{OUTPUT_NAME}` output"))?;

        let embed_dim = match &output.dtype() {
            ValueType::Tensor { shape: dimensions, .. } => {
                let d = *dimensions.last().unwrap_or(&0);
                if d <= 0 {
                    bail!("dinov3 output last dim must be static, got {d}");
                }
                d as u32
            }
            _ => bail!("dinov3 last_hidden_state must be a tensor"),
        };

        Ok(Dinov3 {
            input_name,
            output_name: OUTPUT_NAME.to_string(),
            grid_side: INPUT_SIZE / PATCH_SIZE,
            embed_dim,
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

        let d = self.embed_dim as usize;
        let n_patches = (self.grid_side * self.grid_side) as usize;

        let mut values = Vec::with_capacity(n_patches * d);
        for t in 0..n_patches {
            let src = SKIP_TOKENS + t;
            let token = &raw[src * d..(src + 1) * d];
            let norm = token
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            values.extend(token.iter().map(|&v| v / norm));
        }

        AIOutputs::Embed(Embedding::square(
            values,
            self.model_name.clone(),
            self.grid_side,
        ))
    }
}
