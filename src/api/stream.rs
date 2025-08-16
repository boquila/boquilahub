use chrono::Local;
use ffmpeg_next as ffmpeg;
use image::{ImageBuffer, Rgb};
use std::iter::Iterator;

use crate::api::utils::extract_img;

pub struct Feed {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    index: usize,
    decoded: ffmpeg::frame::Video,
    pub frames: i64,
}

impl Feed {
    pub fn new(url: &str) -> Result<Self, ffmpeg::Error> {
        // Initialize FFmpeg
        ffmpeg::init()?;
        std::fs::create_dir_all("export/feed").unwrap();

        // Open the RTSP stream with options for better RTSP handling
        let mut opts: ffmpeg::Dictionary<'_> = ffmpeg::Dictionary::new();
        opts.set("rtsp_transport", "tcp"); // Use TCP instead of UDP for more reliable streaming
        opts.set("stimeout", "5000000"); // Set socket timeout (in microseconds)
        opts.set("timeout", "5000000"); // General timeout value

        // Open input with options
        let input_ctx = ffmpeg::format::input_with_dictionary(url, opts)?;

        // Find the first video stream
        let video_stream: ffmpeg::Stream<'_> = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| ffmpeg::Error::StreamNotFound)?;

        let frames = video_stream.frames();

        let index = video_stream.index();

        let decoder = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?
            .decoder()
            .video()?;

        let decoded = ffmpeg::frame::Video::empty();

        Ok(Self {
            input_ctx,
            decoder,
            index,
            decoded,
            frames,
        })
    }
}

impl Iterator for Feed {
    // The iterator yields image buffers
    type Item = ImageBuffer<Rgb<u8>, Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Process packets until we get a frame
        for (stream, packet) in &mut self.input_ctx.packets() {
            // Only process video stream packets
            if stream.index() != self.index {
                continue;
            }

            // Try to decode the packet
            if self.decoder.send_packet(&packet).is_err() {
                continue;
            }

            // If we got a frame, return it
            if self.decoder.receive_frame(&mut self.decoded).is_ok() {
                return Some(extract_img(&self.decoded));
            }
        }

        // Try to flush any remaining frames
        self.decoder.send_eof().ok();
        if self.decoder.receive_frame(&mut self.decoded).is_ok() {
            return Some(extract_img(&self.decoded));
        }

        None
    }
}

pub fn save_frame(frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) {
    let now = Local::now();
    let date_str = now.format("%Y-%m-%d_%H-%M-%S.%3f").to_string();
    let filename = format!("output_feed/detection_{}.jpg", date_str);
    let _ = frame.save(filename);
}
