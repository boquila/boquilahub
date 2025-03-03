use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use std::{iter::Iterator, path::Path};
use video_rs::{Decoder, DecoderBuilder};

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
    type Item = ImageBuffer<Rgb<u8>, Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((_, frame))) => {
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
                Some(img)
            },
            _ => None,
        }
    }
}

#[frb(ignore)]
pub fn process_video_file(file_path: &str) -> Result<FileVideoFrameIterator, Box<dyn std::error::Error>> {
    FileVideoFrameIterator::new(file_path)
}

// Given a video file_path
// We run inference for each frame then create a new videofile with the predictions
fn predict_video(file_path: &str){
    let a = FileVideoFrameIterator::new(file_path).unwrap();
    for frame in a {
        
    }
}