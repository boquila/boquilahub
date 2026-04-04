use super::utils::extract_img;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use image::{ImageBuffer, Rgb};
use std::{
    iter::Iterator,
    path::{Path, PathBuf},
};

pub type Time = i64;

struct SendScaler(ffmpeg::software::scaling::Context);
unsafe impl Send for SendScaler {}

impl std::ops::Deref for SendScaler {
    type Target = ffmpeg::software::scaling::Context;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SendScaler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub fn get_output_path(file_path: &str) -> PathBuf {
    let path = Path::new(file_path);
    let file_name = path
        .file_name()
        .map(|name| format!("predict_{}", name.to_string_lossy()))
        .unwrap_or_else(|| "predict_output".to_string());

    match path.parent() {
        Some(parent) => parent.join(file_name),
        None => PathBuf::from(file_name),
    }
}

pub struct VideoEncoder {
    encoder: ffmpeg::encoder::Video,
    output_ctx: ffmpeg::format::context::Output,
    scaler: SendScaler,
    stream_index: usize,
    width: u32,
    height: u32,
    finished: bool,
    input_time_base: ffmpeg::Rational,
    encoder_time_base: ffmpeg::Rational,
    output_time_base: ffmpeg::Rational,
}

impl VideoEncoder {
    fn encode_frame(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>, pts: Time) {
        // Rescale PTS from input stream time_base to encoder time_base
        let enc_pts = pts.rescale(self.input_time_base, self.encoder_time_base);

        let mut rgb_frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, self.width, self.height);

        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data_mut(0);
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let pixel = img.get_pixel(x as u32, y as u32);
                let offset = y * stride + x * 3;
                data[offset] = pixel[0];
                data[offset + 1] = pixel[1];
                data[offset + 2] = pixel[2];
            }
        }

        let mut yuv_frame = ffmpeg::frame::Video::empty();
        self.scaler.run(&rgb_frame, &mut yuv_frame).unwrap();
        yuv_frame.set_pts(Some(enc_pts));

        self.encoder.send_frame(&yuv_frame).unwrap();
        self.receive_and_write_packets();
    }

    fn receive_and_write_packets(&mut self) {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.stream_index);
            packet.rescale_ts(self.encoder_time_base, self.output_time_base);
            packet.write_interleaved(&mut self.output_ctx).unwrap();
        }
    }

    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let _ = self.encoder.send_eof();
        self.receive_and_write_packets();
        self.output_ctx.write_trailer().unwrap();
    }
}

pub struct VideofileProcessor {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    decoded: ffmpeg::frame::Video,
    pub encoder: VideoEncoder,
    frames: i64,
}

impl VideofileProcessor {
    pub fn new(file_path: &str) -> Self {
        ffmpeg::init().unwrap();

        let input_ctx = ffmpeg::format::input(&Path::new(file_path)).unwrap();

        let (stream_index, time_base, frames, decoder) = {
            let video_stream = input_ctx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .unwrap();
            let idx = video_stream.index();
            let tb = video_stream.time_base();
            let fr = video_stream.frames();
            let dec = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())
                .unwrap()
                .decoder()
                .video()
                .unwrap();
            (idx, tb, fr, dec)
        };

        let width = decoder.width();
        let height = decoder.height();

        let output_path = get_output_path(file_path);
        let mut output_ctx = ffmpeg::format::output(&output_path).unwrap();

        let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::H264)
            .expect("H264 encoder not found");

        let global_header = output_ctx
            .format()
            .flags()
            .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

        let mut ost = output_ctx.add_stream(codec).unwrap();
        let enc_stream_index = ost.index();

        let ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
        let mut enc = ctx.encoder().video().unwrap();
        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(ffmpeg::format::Pixel::YUV420P);
        enc.set_time_base(time_base);
        enc.set_frame_rate(decoder.frame_rate());

        if global_header {
            enc.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }

        let encoder = enc.open_as(codec).unwrap();
        ost.set_parameters(&encoder);

        output_ctx.write_header().unwrap();

        // Must read output stream time_base AFTER write_header (ffmpeg may change it)
        let output_time_base = output_ctx.stream(enc_stream_index).unwrap().time_base();
        let encoder_time_base = encoder.time_base();

        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24,
            width,
            height,
            ffmpeg::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .unwrap();

        let video_encoder = VideoEncoder {
            encoder,
            output_ctx,
            scaler: SendScaler(scaler),
            stream_index: enc_stream_index,
            width,
            height,
            finished: false,
            input_time_base: time_base,
            encoder_time_base,
            output_time_base,
        };

        Self {
            input_ctx,
            decoder,
            stream_index,
            decoded: ffmpeg::frame::Video::empty(),
            encoder: video_encoder,
            frames,
        }
    }

    pub fn get_n_frames(&self) -> u64 {
        self.frames as u64
    }

    pub fn encode(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>, time: Time) {
        self.encoder.encode_frame(img, time);
    }

    pub fn first_frame(
        file_path: &str,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        ffmpeg::init()?;

        let mut input_ctx = ffmpeg::format::input(&Path::new(file_path))?;

        let (stream_index, mut decoder) = {
            let video_stream = input_ctx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or("No video stream found")?;
            let idx = video_stream.index();
            let dec =
                ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?
                    .decoder()
                    .video()?;
            (idx, dec)
        };

        let mut decoded = ffmpeg::frame::Video::empty();

        for (stream, packet) in input_ctx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            if decoder.send_packet(&packet).is_err() {
                continue;
            }
            if decoder.receive_frame(&mut decoded).is_ok() {
                return Ok(extract_img(&decoded));
            }
        }

        Err("No frames found".into())
    }
}

impl Iterator for VideofileProcessor {
    type Item = (Time, ImageBuffer<Rgb<u8>, Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        for (stream, packet) in &mut self.input_ctx.packets() {
            if stream.index() != self.stream_index {
                continue;
            }
            if self.decoder.send_packet(&packet).is_err() {
                continue;
            }
            if self.decoder.receive_frame(&mut self.decoded).is_ok() {
                let pts = self.decoded.pts().unwrap_or(0);
                return Some((pts, extract_img(&self.decoded)));
            }
        }

        self.decoder.send_eof().ok();
        if self.decoder.receive_frame(&mut self.decoded).is_ok() {
            let pts = self.decoded.pts().unwrap_or(0);
            return Some((pts, extract_img(&self.decoded)));
        }

        None
    }
}
