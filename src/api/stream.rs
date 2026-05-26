use crate::api::utils::{rgb_frame_to_imgbuf, SendScaler};
use ffmpeg_next as ffmpeg;
use image::{ImageBuffer, Rgb};
use std::iter::Iterator;

pub struct Feed {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: SendScaler,
    index: usize,
    decoded: ffmpeg::frame::Video,
    pub frames: i64,
}

impl Feed {
    pub fn new(url: &str) -> Result<Self, ffmpeg::Error> {
        // Initialize FFmpeg
        ffmpeg::init()?;
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
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

        let scaler = SendScaler(ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?);

        let decoded = ffmpeg::frame::Video::empty();

        Ok(Self {
            input_ctx,
            decoder,
            scaler,
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
        for (stream, packet) in &mut self.input_ctx.packets() {
            if stream.index() != self.index {
                continue;
            }

            if self.decoder.send_packet(&packet).is_err() {
                continue;
            }

            if self.decoder.receive_frame(&mut self.decoded).is_ok() {
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                self.scaler.run(&self.decoded, &mut rgb_frame).unwrap();
                return Some(rgb_frame_to_imgbuf(&rgb_frame));
            }
        }

        self.decoder.send_eof().ok();
        if self.decoder.receive_frame(&mut self.decoded).is_ok() {
            let mut rgb_frame = ffmpeg::frame::Video::empty();
            self.scaler.run(&self.decoded, &mut rgb_frame).unwrap();
            return Some(rgb_frame_to_imgbuf(&rgb_frame));
        }

        None
    }
}