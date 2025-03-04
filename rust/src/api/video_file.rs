use super::inference::{detect_from_imgbuf, simple_xyxy_to_bbox};
use super::render::draw_bbox_from_imgbuf;
use super::rest::detect_bbox_from_buf_remotely;
use super::utils::image_buffer_to_ndarray;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::{iter::Iterator, path::Path};
use video_rs::encode::Settings;
use video_rs::{Decoder, DecoderBuilder, Encoder, Time, WriterBuilder};

struct VideofileProcessor {
    decoder: Decoder,
    encoder: Encoder,
}

impl VideofileProcessor {
    fn new(file_path: &str) -> Self {
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
}

impl Iterator for VideofileProcessor {
    type Item = (Time, ImageBuffer<Rgb<u8>, Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((time, frame))) => {
                let dims = frame.dim();
                let height = dims.0;
                let width = dims.1;

                let mut img = ImageBuffer::new(width as u32, height as u32);

                for y in 0..height {
                    for x in 0..width {
                        let b = frame[[y, x, 2]];
                        let g = frame[[y, x, 1]];
                        let r = frame[[y, x, 0]];
                        img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
                Some((time, img))
            }
            _ => None,
        }
    }
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_videofile(file_path: &str) {
    let mut frame_processor = VideofileProcessor::new(file_path);
    while let Some((time, mut frame)) = frame_processor.next() {
        let array = image_buffer_to_ndarray(&frame);
        let predictions = simple_xyxy_to_bbox(detect_from_imgbuf(&frame));
        draw_bbox_from_imgbuf(&mut frame, &predictions);        
        frame_processor.encoder.encode(&array, time).unwrap();
    }
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_videofile_remotely(file_path: &str, url: &str) {
    let mut frame_processor = VideofileProcessor::new(file_path);
    while let Some((time, mut frame)) = frame_processor.next() {
        let array = image_buffer_to_ndarray(&frame);
        let predictions = detect_bbox_from_buf_remotely(url.to_string(),frame.to_vec());
        draw_bbox_from_imgbuf(&mut frame, &predictions);
        frame_processor.encoder.encode(&array, time).unwrap();
    }
}
