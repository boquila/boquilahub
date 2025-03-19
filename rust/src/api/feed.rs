use super::{
    abstractions::BBox,
    inference::detect_bbox_from_imgbuf,
    render::draw_bbox_from_imgbuf,
    rest::detect_bbox_from_buf_remotely,
    utils::{image_buffer_to_jpg_buffer, ndarray_to_image_buffer},
};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::{error::Error, iter::Iterator};
use video_rs::{Decoder, DecoderBuilder, Options, Time, Url};

pub struct RTSPFrameIterator {
    decoder: Decoder,
}

impl Iterator for RTSPFrameIterator {
    type Item = (Time, ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>);
    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.decode_iter().next() {
            Some(Ok((time, frame))) => Some((time, frame)),
            _ => None,
        }
    }
}

impl RTSPFrameIterator {
    #[flutter_rust_bridge::frb(sync)]
    pub fn new(url: &str) -> Self {
        video_rs::init().unwrap();
        let options = Options::preset_rtsp_transport_tcp();
        let source = url.parse::<Url>().unwrap();
        let decoder = DecoderBuilder::new(source)
            .with_options(&options)
            .build()
            .unwrap();
        Self { decoder }
    }

    // If the annotation is provided, it will just use that instead of computing it.
    fn process_frame<F>(&mut self, prediction_fn: F) -> Result<(Vec<u8>, Vec<BBox>), Box<dyn Error>>
    where
        F: Fn(&image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<BBox>,
    {
        match self.next() {
            Some((_, frame)) => {
                let mut img = ndarray_to_image_buffer(&frame);
                let predictions;
                predictions = prediction_fn(&img);
                draw_bbox_from_imgbuf(&mut img, &predictions);
                let jpg_buffer = image_buffer_to_jpg_buffer(img);
                Ok((jpg_buffer, predictions))
            }
            None => Err("Failed to retrieve the next frame.".into()), // Handle None case by returning a descriptive error
        }
    }

    fn run(&mut self) -> Result<(Vec<u8>, Vec<BBox>), Box<dyn Error>> {
        self.process_frame(|img| detect_bbox_from_imgbuf(img))
    }

    fn run_remotely(&mut self, url: &str) -> Result<(Vec<u8>, Vec<BBox>), Box<dyn Error>> {
        self.process_frame(|img| detect_bbox_from_buf_remotely(url.to_string(), img.to_vec()))
    }

    pub fn run_exp(&mut self) -> (Vec<u8>, Vec<BBox>) {
        self.run().unwrap()
    }

    pub fn run_remotely_exp(&mut self, url: &str) -> (Vec<u8>, Vec<BBox>) {
        self.run_remotely(url).unwrap()
    }

    pub fn ignore_frame(&mut self) {
        self.next();
    }

    pub fn get_jpg_frame(&mut self) -> Vec<u8> {
        let (a,frame) = self.next().unwrap();
        let img = ndarray_to_image_buffer(&frame);
        let jpg_buffer = image_buffer_to_jpg_buffer(img);
        jpg_buffer
    }
}