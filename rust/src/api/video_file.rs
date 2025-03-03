use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use ndarray::{Array3, ArrayBase, Dim, OwnedRepr};
use video_rs::encode::Settings;
use std::collections::HashMap;
use std::{iter::Iterator, path::Path};
use video_rs::{Decoder, DecoderBuilder, Encoder, Time, WriterBuilder};
use super::inference::{detect_from_imgbuf, simple_xyxy_to_bbox};
use super::render::draw_bbox_from_imgbuf;

#[frb(ignore)]
pub struct FileVideoFrameIterator {
    decoder: Decoder,
}

impl FileVideoFrameIterator {
    pub fn new(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
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
                        let r = frame[[y, x, 0]];
                        let g = frame[[y, x, 1]];
                        let b = frame[[y, x, 2]];
                        img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
                Some((time, img))
            },
            _ => None,
        }
    }
}

#[frb(ignore)]
pub fn process_video_file(file_path: &str) -> Result<FileVideoFrameIterator, Box<dyn std::error::Error>> {
    FileVideoFrameIterator::new(file_path)
}

fn image_buffer_to_ndarray(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>> {
    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;
    
    // Create a new 3D array with dimensions [height, width, 3]
    let mut array = Array3::<u8>::zeros((height, width, 3));
    
    // Fill the array with pixel data from the image
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x as u32, y as u32);
            array[[y, x, 0]] = pixel[0]; // R
            array[[y, x, 1]] = pixel[1]; // G
            array[[y, x, 2]] = pixel[2]; // B
        }
    }
    
    array
}

// Given a video file_path
// We run inference for each frame then create a new videofile displayingthe predictions
#[flutter_rust_bridge::frb(dart_async)]
pub fn predict_video(file_path: &str){
    let frame_iterator = FileVideoFrameIterator::new(file_path).unwrap();
    let (w, h) = frame_iterator.decoder.size();

    let mut options = HashMap::new();
    options.insert(
        "movflags".to_string(),
        "frag_keyframe+empty_moov".to_string(),
    );
    
    let output_path = format!("predicted_{}",file_path);

    let mut _writer = WriterBuilder::new(Path::new(&output_path))
        .with_options(&options.into())
        .build().unwrap();

    let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
    let mut enc = Encoder::new(Path::new(&output_path), settings).unwrap();

    for (time, mut frame) in frame_iterator {
        let predictions = detect_from_imgbuf(&frame);
        let predictions_bbox = simple_xyxy_to_bbox(predictions);
        draw_bbox_from_imgbuf(&mut frame, &predictions_bbox);
        let array = image_buffer_to_ndarray(&frame);

        enc.encode(&array, time).unwrap();        
    }
}

