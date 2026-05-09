use super::utils::rgb_frame_to_imgbuf;
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

enum DecodedItem {
    VideoFrame(Time, ImageBuffer<Rgb<u8>, Vec<u8>>),
    AudioPacket(ffmpeg::Packet),
}

// SAFETY: We transfer exclusive ownership of ffmpeg types across threads.
// No concurrent access occurs.
unsafe impl Send for DecodedItem {}

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
    audio_output_stream_index: Option<usize>,
    audio_input_time_base: Option<ffmpeg::Rational>,
    audio_output_time_base: Option<ffmpeg::Rational>,
}

impl VideoEncoder {
    fn encode_frame(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>, pts: Time) {
        let enc_pts = pts.rescale(self.input_time_base, self.encoder_time_base);

        let mut rgb_frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, self.width, self.height);

        // Row-by-row memcpy instead of pixel-by-pixel copy
        let raw = img.as_raw();
        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data_mut(0);
        let row_bytes = self.width as usize * 3;
        for y in 0..self.height as usize {
            let src_start = y * row_bytes;
            let dst_start = y * stride;
            data[dst_start..dst_start + row_bytes]
                .copy_from_slice(&raw[src_start..src_start + row_bytes]);
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

    pub fn write_audio_packet(&mut self, mut packet: ffmpeg::Packet) {
        if let (Some(out_idx), Some(in_tb), Some(out_tb)) = (
            self.audio_output_stream_index,
            self.audio_input_time_base,
            self.audio_output_time_base,
        ) {
            packet.set_stream(out_idx);
            packet.rescale_ts(in_tb, out_tb);
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
    receiver: std::sync::mpsc::Receiver<DecodedItem>,
    pub encoder: VideoEncoder,
    frames: i64,
}

impl VideofileProcessor {
    pub fn new(file_path: &str) -> Self {
        ffmpeg::init().unwrap();
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);

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

        let audio_info = {
            input_ctx
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .map(|s| (s.index(), s.time_base(), s.parameters()))
        };

        let width = decoder.width();
        let height = decoder.height();
        let decoder_format = decoder.format();

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

        let (audio_input_idx, audio_output_stream_index, audio_input_time_base) =
            if let Some((a_idx, a_tb, a_params)) = audio_info {
                let mut audio_ost =
                    output_ctx.add_stream(None::<ffmpeg::Codec>).unwrap();
                audio_ost.set_parameters(a_params);
                let a_out_idx = audio_ost.index();
                (Some(a_idx), Some(a_out_idx), Some(a_tb))
            } else {
                (None, None, None)
            };

        output_ctx.write_header().unwrap();

        // Must read output stream time_base AFTER write_header (ffmpeg may change it)
        let output_time_base = output_ctx.stream(enc_stream_index).unwrap().time_base();
        let encoder_time_base = encoder.time_base();

        let audio_output_time_base = audio_output_stream_index
            .map(|idx| output_ctx.stream(idx).unwrap().time_base());

        let encode_scaler = ffmpeg::software::scaling::Context::get(
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
            scaler: SendScaler(encode_scaler),
            stream_index: enc_stream_index,
            width,
            height,
            finished: false,
            input_time_base: time_base,
            encoder_time_base,
            output_time_base,
            audio_output_stream_index,
            audio_input_time_base,
            audio_output_time_base,
        };

        // Decode scaler: decoder pixel format -> RGB24 (SIMD-optimized)
        let mut decode_scaler = SendScaler(
            ffmpeg::software::scaling::Context::get(
                decoder_format,
                width,
                height,
                ffmpeg::format::Pixel::RGB24,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            )
            .unwrap(),
        );

        // Prefetch: decode ahead in a background thread with a bounded buffer
        let (tx, rx) = std::sync::mpsc::sync_channel::<DecodedItem>(8);

        std::thread::spawn(move || {
            let mut decoder = decoder;
            let mut input_ctx = input_ctx;
            let mut decoded = ffmpeg::frame::Video::empty();

            for (stream, packet) in input_ctx.packets() {
                let idx = stream.index();

                if Some(idx) == audio_input_idx {
                    if tx.send(DecodedItem::AudioPacket(packet)).is_err() {
                        return;
                    }
                    continue;
                }

                if idx != stream_index {
                    continue;
                }

                if decoder.send_packet(&packet).is_err() {
                    continue;
                }

                while decoder.receive_frame(&mut decoded).is_ok() {
                    let pts = decoded.pts().unwrap_or(0);
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
                    let img = rgb_frame_to_imgbuf(&rgb_frame);

                    if tx.send(DecodedItem::VideoFrame(pts, img)).is_err() {
                        return;
                    }
                }
            }

            // Flush buffered frames from decoder
            let _ = decoder.send_eof();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let pts = decoded.pts().unwrap_or(0);
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
                let img = rgb_frame_to_imgbuf(&rgb_frame);

                if tx.send(DecodedItem::VideoFrame(pts, img)).is_err() {
                    return;
                }
            }
        });

        Self {
            receiver: rx,
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

        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?;

        let mut decoded = ffmpeg::frame::Video::empty();

        for (stream, packet) in input_ctx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            if decoder.send_packet(&packet).is_err() {
                continue;
            }
            if decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;
                return Ok(rgb_frame_to_imgbuf(&rgb_frame));
            }
        }

        Err("No frames found".into())
    }
}

impl Iterator for VideofileProcessor {
    type Item = (Time, ImageBuffer<Rgb<u8>, Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.receiver.recv() {
                Ok(DecodedItem::VideoFrame(pts, img)) => return Some((pts, img)),
                Ok(DecodedItem::AudioPacket(packet)) => {
                    self.encoder.write_audio_packet(packet);
                }
                Err(_) => return None,
            }
        }
    }
}
