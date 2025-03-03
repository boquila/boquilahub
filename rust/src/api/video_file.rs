use super::inference::{detect_from_imgbuf, simple_xyxy_to_bbox};
use super::render::draw_bbox_from_imgbuf;
use super::utils::image_buffer_to_ndarray;
use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::{iter::Iterator, path::Path};
use video_rs::encode::Settings;
use video_rs::{Decoder, DecoderBuilder, Encoder, Time, WriterBuilder};

struct FileVideoFrameIterator {
    decoder: Decoder,
}

impl FileVideoFrameIterator {
    fn new(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        video_rs::init()?;
        let decoder = DecoderBuilder::new(Path::new(file_path)).build()?;
        Ok(FileVideoFrameIterator { decoder })
    }
}

impl Iterator for FileVideoFrameIterator {
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

#[frb(unignore)]
pub struct FrameProcessor {
    frame_iterator: FileVideoFrameIterator,
    encoder: Encoder,
    frame_count: usize,
}

impl FrameProcessor {
    fn new(file_path: &str) -> Self {
        let frame_iterator = FileVideoFrameIterator::new(file_path).unwrap();
        let (w, h) = frame_iterator.decoder.size();

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

        FrameProcessor {
            frame_iterator,
            encoder,
            frame_count: 0,
        }
    }
}

impl Iterator for FrameProcessor {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((time, mut frame)) = self.frame_iterator.next() {
            let predictions = detect_from_imgbuf(&frame);
            let predictions_bbox = simple_xyxy_to_bbox(predictions);
            draw_bbox_from_imgbuf(&mut frame, &predictions_bbox);
            let array = image_buffer_to_ndarray(&frame);

            self.encoder.encode(&array, time).unwrap();
            self.frame_count += 1;
            Some(self.frame_count)
        } else {
            None
        }
    }
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_video_file(file_path: &str) {
    let mut frame_processor = FrameProcessor::new(file_path);
    while let Some(_frame_count) = frame_processor.next(){}
}
