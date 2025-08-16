use ffmpeg_next as ffmpeg;
use image::{ImageBuffer, Rgb};

use crate::api::utils::extract_img;

pub fn get_output_path(file_path: &str) -> String {
    if let Some(pos) = file_path.rfind(['\\', '/']) {
        let (directory, file_name) = file_path.split_at(pos + 1);
        let new_file_name = format!("predict_{}", file_name);
        format!("{}{}", directory, new_file_name)
    } else {
        format!("predict_{}", file_path)
    }
}

pub struct VideofileProcessor {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    encoder: ffmpeg::encoder::Video,
    output_ctx: ffmpeg::format::context::Output,
    video_stream_index: usize,
    scaler: ffmpeg::software::scaling::Context,
    pub frame_count: i64,
    pub total_frames: i64,
    input_finished: bool,
}

unsafe impl Send for VideofileProcessor {}

impl VideofileProcessor {
    pub fn new(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        ffmpeg::init()?;

        // Open input file
        let input_ctx = ffmpeg::format::input(&std::path::Path::new(file_path))?;

        // Find video stream
        let video_stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream found")?;

        let video_stream_index = video_stream.index();
        let total_frames = video_stream.frames();

        // Get decoder context
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let decoder = context_decoder.decoder().video()?;

        let width = decoder.width();
        let height = decoder.height();
        let fps = video_stream.avg_frame_rate();

        // Create output file
        let output_path = get_output_path(file_path);
        let mut output_ctx = ffmpeg::format::output(&std::path::Path::new(&output_path))?;

        // Create encoder
        let codec =
            ffmpeg::encoder::find(ffmpeg::codec::Id::H264).ok_or("H264 encoder not found")?;

        let mut output_stream = output_ctx.add_stream(codec)?;

        let context_encoder = ffmpeg::codec::context::Context::new_with_codec(codec);
        let mut encoder = context_encoder.encoder().video()?;

        encoder.set_width(width);
        encoder.set_height(height);
        encoder.set_format(ffmpeg::format::Pixel::YUV420P);
        encoder.set_time_base(fps.invert());
        output_stream.set_time_base(fps.invert());
        encoder.set_bit_rate(2000000); // 2Mbps
        encoder.set_gop(12);
        encoder.set_max_b_frames(2);

        // Set encoder options for better compatibility
        let mut opts = ffmpeg::Dictionary::new();
        opts.set("preset", "medium");
        opts.set("crf", "23");

        let encoder = encoder.open_as_with(codec, opts)?;
        output_stream.set_parameters(&encoder);

        // Create scaler for RGB to YUV420P conversion
        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24,
            width,
            height,
            ffmpeg::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?;

        // Write output header
        output_ctx.write_header()?;

        Ok(Self {
            input_ctx,
            decoder,
            encoder,
            output_ctx,
            video_stream_index,
            scaler,
            frame_count: 0,
            total_frames,
            input_finished: false,
        })
    }

    pub fn encode(
        &mut self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let width = img.width() as usize;
        let height = img.height() as usize;

        // Create RGB frame from image buffer
        let mut rgb_frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, width as u32, height as u32);

        // Get stride first (immutable borrow)
        let stride = rgb_frame.stride(0);

        // Then get mutable data
        let data = rgb_frame.data_mut(0);

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                let offset = y * stride + x * 3;
                data[offset] = pixel[0]; // R
                data[offset + 1] = pixel[1]; // G
                data[offset + 2] = pixel[2]; // B
            }
        }

        // Create YUV frame for encoding
        let mut yuv_frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width as u32, height as u32);

        // Convert RGB to YUV420P using scaler
        self.scaler.run(&rgb_frame, &mut yuv_frame)?;

        // Set presentation timestamp
        yuv_frame.set_pts(Some(self.frame_count));
        self.frame_count += 1;

        // Send frame to encoder
        self.encoder.send_frame(&yuv_frame)?;

        // Receive and write encoded packets
        self.write_encoded_packets()?;

        Ok(())
    }

    fn write_encoded_packets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(0);
            packet.rescale_ts(
                self.encoder.time_base(),
                self.output_ctx.stream(0).unwrap().time_base(),
            );
            packet.write_interleaved(&mut self.output_ctx)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Flush encoder
        self.encoder.send_eof()?;
        self.write_encoded_packets()?;

        // Write trailer
        self.output_ctx.write_trailer()?;
        Ok(())
    }

    pub fn new_first_frame(
        file_path: std::path::PathBuf
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        ffmpeg::init()?;

        // Open input file
        let mut input_ctx = ffmpeg::format::input(&file_path)?;

        // Find video stream
        let video_stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream found")?;

        let video_stream_index = video_stream.index();

        // Get decoder context
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;

        let mut decoded = ffmpeg::frame::Video::empty();

        // Process packets until we get the first frame
        for (stream, packet) in input_ctx.packets() {
            if stream.index() != video_stream_index {
                continue;
            }

            if decoder.send_packet(&packet).is_ok() && decoder.receive_frame(&mut decoded).is_ok() {
                return Ok(extract_img(&decoded));
            }
        }

        Err("No frame found".into())
    }
}

impl Iterator for VideofileProcessor {
    type Item = ImageBuffer<Rgb<u8>, Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.input_finished {
            return None;
        }

        let mut decoded = ffmpeg::frame::Video::empty();

        // Process packets until we get a frame
        for (stream, packet) in &mut self.input_ctx.packets() {
            if stream.index() != self.video_stream_index {
                continue;
            }

            if self.decoder.send_packet(&packet).is_err() {
                continue;
            }

            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return Some(extract_img(&decoded));
            }
        }

        // Try to flush remaining frames
        if self.decoder.send_eof().is_ok() {
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return Some(extract_img(&decoded));
            }
        }

        self.input_finished = true;
        None
    }
}
