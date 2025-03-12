use super::abstractions::BBox;
use super::inference::detect_bbox_from_imgbuf;
use super::render::draw_bbox_from_imgbuf;
use super::rest::detect_bbox_from_buf_remotely;
use super::utils::{image_buffer_to_jpg_buffer, ndarray_to_image_buffer};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::collections::HashMap;
use std::{iter::Iterator, path::Path};
use video_rs::encode::Settings;
use video_rs::{Decoder, DecoderBuilder, Encoder, Time, WriterBuilder};

struct VideofileProcessor {
    decoder: Decoder,
    encoder: Encoder,
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

        let output_path = format!("predicted_{}", file_path);

        let _writer = WriterBuilder::new(Path::new(&output_path))
            .with_options(&options.into())
            .build()
            .unwrap();

        let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
        let encoder = Encoder::new(Path::new(&output_path), settings).unwrap();

        Self { decoder, encoder }
    }

    fn process_frame<F>(&mut self, prediction_fn: F) -> (Vec<u8>, Vec<BBox>)
    where
        F: Fn(&image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<BBox>,
    {
        let (time, frame) = self.next().unwrap();
        let mut img = ndarray_to_image_buffer(&frame);
        let predictions = prediction_fn(&img);
        draw_bbox_from_imgbuf(&mut img, &predictions);
        self.encoder.encode(&frame, time).unwrap();
        let jpg_buffer = image_buffer_to_jpg_buffer(img);
        (jpg_buffer, predictions)
    }

    pub fn run(&mut self) -> (Vec<u8>, Vec<BBox>){
        self.process_frame(|img| detect_bbox_from_imgbuf(img))
    }
    
    pub fn run_remotely(&mut self, url: &str) -> (Vec<u8>, Vec<BBox>) {
        self.process_frame(|img| detect_bbox_from_buf_remotely(url.to_string(), img.to_vec()))
    }
    
    
}

impl Iterator for VideofileProcessor {
    type Item = (Time, ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((time, frame))) => Some((time, frame)),
            _ => None,
        }
    }
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_videofile(file_path: &str) {
    let mut frame_processor = VideofileProcessor::new(file_path);
    while let _data = frame_processor.run() {}
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_videofile_remotely(file_path: &str, url: &str) {
    let mut frame_processor = VideofileProcessor::new(file_path);
    while let _data = frame_processor.run_remotely(url) {}
}
