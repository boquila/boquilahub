use flutter_rust_bridge::frb;
use image::{ImageBuffer, Rgb};
use std::iter::Iterator;
use video_rs::{Decoder, DecoderBuilder, Options, Time, Url};
use super::utils::ndarray_to_image_buffer;

struct VideoFrameIterator {
    decoder: Decoder,
}

impl VideoFrameIterator {
    pub fn new(url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        video_rs::init()?;
        let options = Options::preset_rtsp_transport_tcp();
        let source = url.parse::<Url>()?;

        let decoder = DecoderBuilder::new(source).with_options(&options).build()?;

        Ok(VideoFrameIterator { decoder })
    }
}

impl Iterator for VideoFrameIterator {
    type Item = (Time, ImageBuffer<Rgb<u8>, Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((time, frame))) => Some((time, ndarray_to_image_buffer(&frame))),
            Some(Err(_)) => None, // Skip frames with errors
            None => None,
        }
    }
}

#[frb(ignore)]
pub fn process_rtsp(url: &str) -> Result<VideoFrameIterator, Box<dyn std::error::Error>> {
    VideoFrameIterator::new(url)
}
