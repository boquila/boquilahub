use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use std::iter::Iterator;
use video_rs::{Decoder, DecoderBuilder, Url};

#[frb(ignore)]
pub struct FileVideoFrameIterator {
    decoder: Decoder,
}

impl FileVideoFrameIterator {
    pub fn new(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        video_rs::init()?;
        let url = Url::from_file_path(file_path)
            .map_err(|_| "Invalid file path")?;
        let decoder = DecoderBuilder::new(url).build()?;
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

// Example
// fn main() {
//     let frame_iterator = process_rtsp("rtsp://1.1.1.1:8000/101").unwrap();

//     for frame in frame_iterator {
//         // frame is now just the ndarray frame data
//         println!("Got frame with shape: {:?}", frame.shape());
//     }
// }