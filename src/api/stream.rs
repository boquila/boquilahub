use super::{
    abstractions::XYXYc, inference::detect_bbox_from_imgbuf, render::draw_bbox_from_imgbuf,
    rest::detect_bbox_from_buf_remotely, utils::image_buffer_to_jpg_buffer,
};
use ffmpeg_next as ffmpeg;
use image::{ImageBuffer, Rgb};
use std::error::Error;
use std::iter::Iterator;
use std::time::{Duration, Instant};
use std::{fs::File, io::Write};
use chrono::Local;

pub struct VideoStream {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    index: usize,   
    decoded: ffmpeg::frame::Video,
    frames: i64,
}

unsafe impl Sync for VideoStream {}

impl Iterator for VideoStream {
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

impl VideoStream {
    pub fn new(path_or_url: &str) -> Self {
        // Initialize FFmpeg
        ffmpeg::init().unwrap();

        // Open the RTSP stream with options for better RTSP handling
        let mut opts: ffmpeg::Dictionary<'_> = ffmpeg::Dictionary::new();
        opts.set("rtsp_transport", "tcp"); // Use TCP instead of UDP for more reliable streaming
        opts.set("stimeout", "5000000"); // Set socket timeout (in microseconds)
        opts.set("timeout", "5000000"); // General timeout value

        // Open input with options
        let input_ctx = ffmpeg::format::input_with_dictionary(path_or_url, opts).unwrap();

        // Find the first video stream
        let video_stream: ffmpeg::Stream<'_> = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream found")
            .unwrap();

        let frames = video_stream.frames();

        let index = video_stream.index();

        let decoder = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())
            .unwrap()
            .decoder()
            .video()
            .unwrap();
        let decoded = ffmpeg::frame::Video::empty();

        Self {
            input_ctx,
            decoder,
            index,
            decoded,
            frames,
        }
    }

    fn process_frame<F>(
        &mut self,
        prediction_fn: F,
        log: bool
    ) -> Result<(Vec<u8>, Vec<XYXYc>), Box<dyn Error>>
    where
        F: Fn(&image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<XYXYc>,
    {
        match self.next() {
            Some(mut img) => {
                let predictions = prediction_fn(&img);
                draw_bbox_from_imgbuf(&mut img, &predictions);
                let jpg_buffer = image_buffer_to_jpg_buffer(img);
                if log == true {
                    let jpg_buffer_clone = jpg_buffer.clone();
                    std::thread::spawn(move || {
                        let now = Local::now();
                        let date_str = now.format("%Y-%m-%d_%H-%M-%S.%3f").to_string();
                        let filename = format!("output_feed/detection_{}.jpg", date_str);
                        save_jpg(&jpg_buffer_clone, &filename);
                    });
                }
                Ok((jpg_buffer, predictions))
            }
            None => Err("Failed to retrieve the next frame.".into()), // Handle None case by returning a descriptive error
        }
    }

    fn run(&mut self, log: bool) -> Result<(Vec<u8>, Vec<XYXYc>), Box<dyn Error>> {
        self.process_frame(|img| detect_bbox_from_imgbuf(img), log)
    }

    fn run_remotely(&mut self, url: &str, log: bool) -> Result<(Vec<u8>, Vec<XYXYc>), Box<dyn Error>> {
        self.process_frame(|img: &ImageBuffer<Rgb<u8>, Vec<u8>>| detect_bbox_from_buf_remotely(url, img.to_vec()), log)
    }

    pub fn ignore_frame(&mut self) {
        self.next();
    }

    fn measure_method<F>(&mut self, method: F, iterations: u32) -> u32
    where
        F: Fn(&mut Self),
    {
        let mut total_duration = Duration::new(0, 0);

        for _ in 0..iterations {
            let start = Instant::now();
            method(self);
            total_duration += start.elapsed();
        }

        let avg_duration_nanos = total_duration.as_nanos() as f64 / iterations as f64;
        let avg_duration_secs = avg_duration_nanos / 1_000_000_000.0;
        let fps = (1.0 / avg_duration_secs).round() as u32;
        return fps;
    }

    pub fn measure_fps(&mut self, iterations: u32) -> u32 {
        self.measure_method(
            |s| {
                s.next();
            },
            iterations,
        )
    }

    pub fn measure_inference(&mut self, iterations: u32) -> u32 {
        self.measure_method(
            |s| {
                let _ = s.run(false);
            },
            iterations,
        )
    }

    pub fn measure_remote_inference(&mut self, iterations: u32, url: &str) -> u32 {
        self.measure_method(
            |s| {
                let _ = s.run_remotely(url,false);
            },
            iterations,
        )
    }

    pub fn get_n_frames(&self) -> i64 {
        self.frames
    }
}

/// Converts and saves a video frame as a PNG image
fn extract_img(frame: &ffmpeg::frame::Video) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;

    // Create a new RGB image buffer
    let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);

    // Convert the FFmpeg frame data to RGB format
    let data = frame.data(0);
    let linesize = frame.stride(0);

    // Copy data from FFmpeg frame to image buffer
    match frame.format() {
        ffmpeg::format::Pixel::RGB24 => {
            for y in 0..height {
                for x in 0..width {
                    let offset = y * linesize + x * 3;
                    let b = data[offset + 2];
                    let g = data[offset + 1];
                    let r = data[offset];
                    img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
        }
        ffmpeg::format::Pixel::YUV420P | ffmpeg::format::Pixel::YUVJ420P => {
            // For YUV formats, we need to convert from YUV to RGB
            // This requires accessing Y, U, and V planes separately
            let y_plane = frame.data(0);
            let y_stride = frame.stride(0);
            let u_plane = frame.data(1);
            let u_stride = frame.stride(1);
            let v_plane = frame.data(2);
            let v_stride = frame.stride(2);

            for y in 0..height {
                for x in 0..width {
                    let y_value = y_plane[y * y_stride + x] as f32;

                    // Subsample U and V (they are at quarter resolution in YUV420)
                    let u_x = x / 2;
                    let u_y = y / 2;
                    let v_x = x / 2;
                    let v_y = y / 2;

                    let u_value = u_plane[u_y * u_stride + u_x] as f32 - 128.0;
                    let v_value = v_plane[v_y * v_stride + v_x] as f32 - 128.0;

                    // YUV to RGB conversion
                    let r = (y_value + 1.402 * v_value).clamp(0.0, 255.0) as u8;
                    let g =
                        (y_value - 0.344136 * u_value - 0.714136 * v_value).clamp(0.0, 255.0) as u8;
                    let b = (y_value + 1.772 * u_value).clamp(0.0, 255.0) as u8;

                    img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
        }
        _ => {
            // For other formats, we would need to convert using ffmpeg's software scaler (SwsContext)
            // This is more complicated and would require additional code
        }
    }

    return img_buffer;
}

pub fn save_jpg(data: &[u8], filename: &str) {
    let full_filename = format!("{filename}.jpg");
    let mut file = File::create(full_filename).unwrap();
    file.write_all(&data).unwrap();    
}