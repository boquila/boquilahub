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
    pub grid_h: u32,
    pub grid_w: u32,
    pub embed_dim: u32,
    pub drop_cls: bool,
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

        let input_name = session.inputs()[0].name().to_string();

        let out_dims: Vec<i64> = match session.outputs()[0].dtype() {
            ValueType::Tensor { shape: dimensions, .. } => dimensions.to_vec(),
            _ => bail!("expected tensor output for Clip"),
        };

        let (grid_h, grid_w, embed_dim, drop_cls) = parse_embed_shape(&out_dims)?;

        Ok(Clip {
            input_width,
            input_height,
            input_name,
            output_name: session.outputs()[0].name().to_string(),
            grid_h,
            grid_w,
            embed_dim,
            drop_cls,
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

        let d = self.embed_dim as usize;
        let n_tokens = (self.grid_h * self.grid_w) as usize;
        let start_token = if self.drop_cls { 1 } else { 0 };

        let mut values = Vec::with_capacity(n_tokens * d);
        for t in 0..n_tokens {
            let src = start_token + t;
            let token = &raw[src * d..(src + 1) * d];
            let norm = token
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            values.extend(token.iter().map(|&v| v / norm));
        }

        let model = self.model_name.clone();
        AIOutputs::Embed(if self.grid_h == 1 && self.grid_w == 1 {
            Embedding::pooled(values, model)
        } else {
            Embedding::square(values, model, self.grid_h)
        })
    }
}

fn parse_embed_shape(dims: &[i64]) -> Result<(u32, u32, u32, bool), Error> {
    match dims.len() {
        2 => Ok((1, 1, dims[1] as u32, false)),
        3 => {
            let n = dims[1] as usize;
            let d = dims[2] as u32;
            if is_perfect_square(n) {
                let side = (n as f64).sqrt().round() as u32;
                Ok((side, side, d, false))
            } else if n > 0 && is_perfect_square(n - 1) {
                let side = ((n - 1) as f64).sqrt().round() as u32;
                Ok((side, side, d, true))
            } else {
                bail!(
                    "Clip output token count {} is neither a perfect square \
                     nor square+1; can't infer patch grid",
                    n
                );
            }
        }
        r => bail!("Clip output rank {} not supported (need 2 or 3)", r),
    }
}

fn is_perfect_square(n: usize) -> bool {
    let s = (n as f64).sqrt().round() as usize;
    s * s == n
}
