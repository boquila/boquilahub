use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::iter::Iterator;
use video_rs::{Decoder, DecoderBuilder, Options, Url};

pub struct VideoFrameIterator {
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
    type Item = ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((_, frame))) => Some(frame),
            Some(Err(_)) => None, // Skip frames with errors
            None => None,
        }
    }
}

pub fn process_rtsp(url: &str) -> Result<VideoFrameIterator, Box<dyn std::error::Error>> {
    VideoFrameIterator::new(url)
}
