use super::utils::{image_buffer_to_ndarray, ndarray_to_image_buffer};
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::{
    iter::Iterator,
    path::{Path, PathBuf},
};
use video_rs::encode::Settings;
use video_rs::{Decoder, DecoderBuilder, Encoder, Time, WriterBuilder};

pub fn get_output_path(file_path: &str) -> PathBuf {
    let path = Path::new(file_path);
    let file_name = path
        .file_name()
        .map(|name| format!("predict_{}", name.to_string_lossy()))
        .unwrap_or_else(|| "predict_output".to_string());

    match path.parent() {
        Some(parent) => parent.join(file_name),
        None => PathBuf::from(file_name),
    }
}

pub struct VideofileProcessor {
    decoder: Decoder,
    pub encoder: Encoder,
}

impl VideofileProcessor {
    pub fn new(file_path: &str) -> Self {
        video_rs::init().unwrap();
        let decoder = DecoderBuilder::new(Path::new(file_path)).build().unwrap();

        let (w, h) = decoder.size();

        let mut options = HashMap::new();
        options.insert(
            "movflags".to_string(),
            "frag_keyframe+empty_moov".to_string(),
        );
        let output_path = get_output_path(file_path);

        let _writer = WriterBuilder::new(output_path.as_path())
            .with_options(&options.into())
            .build()
            .unwrap();

        let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
        let encoder = Encoder::new(output_path.as_path(), settings).unwrap();

        Self { decoder, encoder }
    }

    pub fn get_n_frames(&self) -> u64 {
        self.decoder.frames().unwrap()
    }

    pub fn encode(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>, time: Time) {
        let final_frame = image_buffer_to_ndarray(&img);
        self.encoder.encode(&final_frame, time).unwrap();
    }

    pub fn first_frame(
        file_path: &str,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        video_rs::init()?;
        let mut decoder = DecoderBuilder::new(Path::new(file_path)).build()?;
        let (_, frame) = decoder
            .decode_iter()
            .next()
            .ok_or("No frames found")?
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(ndarray_to_image_buffer(&frame))
    }
}

impl Iterator for VideofileProcessor {
    type Item = (Time, ImageBuffer<Rgb<u8>, Vec<u8>>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((time, frame))) => Some((time, ndarray_to_image_buffer(&frame))),
            _ => None,
        }
    }
}
