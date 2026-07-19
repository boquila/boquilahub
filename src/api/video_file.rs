use super::abstractions::AIOutputs;
use super::utils::{rgb_frame_to_imgbuf, SendScaler};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use image::{ImageBuffer, Rgb};
use std::{iter::Iterator, path::Path};

pub type Time = i64;

/// Decoded video frame paired with its ordinal frame index (0-based).
struct DecodedFrame {
    index: u64,
    img: ImageBuffer<Rgb<u8>, Vec<u8>>,
}

// SAFETY: ImageBuffer<Rgb<u8>, Vec<u8>> is Send; the wrapper just bundles it
// with an index. Same posture as the prior `DecodedItem`.
unsafe impl Send for DecodedFrame {}

/// Streams decoded RGB frames from a video file in a background thread.
/// Pure decoder — no encoder, no audio passthrough.
pub struct VideofileProcessor {
    receiver: std::sync::mpsc::Receiver<DecodedFrame>,
    width: u32,
    height: u32,
    n_frames: u64,
    fps: f64,
}

impl VideofileProcessor {
    pub fn new(file_path: &str) -> Self {
        ffmpeg::init().unwrap();
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);

        let input_ctx = ffmpeg::format::input(&Path::new(file_path)).unwrap();

        let (stream_index, frames, fps, decoder) = {
            let video_stream = input_ctx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .unwrap();
            let idx = video_stream.index();
            let fr = video_stream.frames();
            let avg = video_stream.avg_frame_rate();
            let fps_from_stream = if avg.denominator() != 0 {
                avg.numerator() as f64 / avg.denominator() as f64
            } else {
                0.0
            };
            let dec = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())
                .unwrap()
                .decoder()
                .video()
                .unwrap();
            let fps = if fps_from_stream > 0.0 {
                fps_from_stream
            } else {
                let dec_fr = dec.frame_rate().unwrap_or(ffmpeg::Rational::new(0, 1));
                if dec_fr.denominator() != 0 {
                    dec_fr.numerator() as f64 / dec_fr.denominator() as f64
                } else {
                    30.0
                }
            };
            (idx, fr, fps, dec)
        };

        let width = decoder.width();
        let height = decoder.height();
        let decoder_format = decoder.format();

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

        let (tx, rx) = std::sync::mpsc::sync_channel::<DecodedFrame>(8);

        std::thread::spawn(move || {
            let mut decoder = decoder;
            let mut input_ctx = input_ctx;
            let mut decoded = ffmpeg::frame::Video::empty();
            let mut frame_idx: u64 = 0;

            for (stream, packet) in input_ctx.packets() {
                if stream.index() != stream_index {
                    continue;
                }
                if decoder.send_packet(&packet).is_err() {
                    continue;
                }
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
                    let img = rgb_frame_to_imgbuf(&rgb_frame);
                    if tx.send(DecodedFrame { index: frame_idx, img }).is_err() {
                        return;
                    }
                    frame_idx += 1;
                }
            }

            let _ = decoder.send_eof();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = ffmpeg::frame::Video::empty();
                decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
                let img = rgb_frame_to_imgbuf(&rgb_frame);
                if tx.send(DecodedFrame { index: frame_idx, img }).is_err() {
                    return;
                }
                frame_idx += 1;
            }
        });

        Self {
            receiver: rx,
            width,
            height,
            n_frames: frames.max(0) as u64,
            fps,
        }
    }

    /// Single-shot open: read metadata and decode the first frame, then close.
    /// Used at file-pick time so the GUI can show the first frame without also
    /// paying for the full streaming decoder + thread spawn that `new()` does
    /// — that gets built lazily when the user actually clicks Analyse.
    pub fn probe(file_path: &str) -> Result<VideoProbe, Box<dyn std::error::Error>> {
        ffmpeg::init()?;
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);

        let mut input_ctx = ffmpeg::format::input(&Path::new(file_path))?;

        let (stream_index, n_frames, fps, mut decoder) = {
            let video_stream = input_ctx
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or("No video stream found")?;
            let idx = video_stream.index();
            let n = video_stream.frames().max(0) as u64;
            let avg = video_stream.avg_frame_rate();
            let fps_from_stream = if avg.denominator() != 0 {
                avg.numerator() as f64 / avg.denominator() as f64
            } else {
                0.0
            };
            let dec =
                ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?
                    .decoder()
                    .video()?;
            let fps = if fps_from_stream > 0.0 {
                fps_from_stream
            } else {
                let dec_fr = dec.frame_rate().unwrap_or(ffmpeg::Rational::new(0, 1));
                if dec_fr.denominator() != 0 {
                    dec_fr.numerator() as f64 / dec_fr.denominator() as f64
                } else {
                    30.0
                }
            };
            (idx, n, fps, dec)
        };

        let width = decoder.width();
        let height = decoder.height();

        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGB24,
            width,
            height,
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
                return Ok(VideoProbe {
                    first_frame: rgb_frame_to_imgbuf(&rgb_frame),
                    width,
                    height,
                    fps,
                    n_frames,
                });
            }
        }

        Err("No frames found".into())
    }
}

pub struct VideoProbe {
    pub first_frame: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub n_frames: u64,
}

impl Iterator for VideofileProcessor {
    /// `(frame_index, image)` — frame_index is 0-based and increases monotonically.
    type Item = (u64, ImageBuffer<Rgb<u8>, Vec<u8>>);

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok().map(|f| (f.index, f.img))
    }
}

/// Re-encode a source video while letting `paint` decorate each decoded frame.
/// Audio (if present) is muxed through unchanged. Used only by the on-demand
/// export — never by the analysis pipeline.
///
/// `paint(frame_idx, &mut img)` is invoked on every decoded frame before encoding.
/// `progress(frame_idx, total)` is invoked after each frame is written.
pub fn export_annotated_video<P, R>(
    input_path: &str,
    output_path: &Path,
    total: u64,
    mut paint: P,
    mut progress: R,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    P: FnMut(u64, &mut ImageBuffer<Rgb<u8>, Vec<u8>>),
    R: FnMut(u64, u64),
{
    ffmpeg::init()?;
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);

    let mut input_ctx = ffmpeg::format::input(&Path::new(input_path))?;

    let (video_in_idx, input_time_base, mut decoder) = {
        let stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream found")?;
        let idx = stream.index();
        let tb = stream.time_base();
        let dec = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .video()?;
        (idx, tb, dec)
    };

    let audio_info = input_ctx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .map(|s| (s.index(), s.time_base(), s.parameters()));

    let width = decoder.width();
    let height = decoder.height();
    let decoder_format = decoder.format();
    let decoder_frame_rate = decoder.frame_rate();

    let mut output_ctx = ffmpeg::format::output(&output_path)?;

    let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::H264).ok_or("H264 encoder not found")?;
    let global_header = output_ctx
        .format()
        .flags()
        .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

    let mut ost = output_ctx.add_stream(codec)?;
    let video_out_idx = ost.index();

    let ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
    let mut enc = ctx.encoder().video()?;
    enc.set_width(width);
    enc.set_height(height);
    enc.set_format(ffmpeg::format::Pixel::YUV420P);
    enc.set_time_base(input_time_base);
    enc.set_frame_rate(decoder_frame_rate);
    if global_header {
        enc.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
    }
    let mut encoder = enc.open_as(codec)?;
    ost.set_parameters(&encoder);

    let (audio_in_idx, audio_out_idx, audio_in_tb) = if let Some((idx, tb, params)) = audio_info {
        let mut audio_ost = output_ctx.add_stream(None::<ffmpeg::Codec>)?;
        audio_ost.set_parameters(params);
        (Some(idx), Some(audio_ost.index()), Some(tb))
    } else {
        (None, None, None)
    };

    output_ctx.write_header()?;

    let output_time_base = output_ctx.stream(video_out_idx).ok_or("missing video stream")?.time_base();
    let encoder_time_base = encoder.time_base();
    let audio_out_tb = audio_out_idx.and_then(|i| output_ctx.stream(i).map(|s| s.time_base()));

    let mut encode_scaler = SendScaler(ffmpeg::software::scaling::Context::get(
        ffmpeg::format::Pixel::RGB24,
        width,
        height,
        ffmpeg::format::Pixel::YUV420P,
        width,
        height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?);
    let mut decode_scaler = SendScaler(ffmpeg::software::scaling::Context::get(
        decoder_format,
        width,
        height,
        ffmpeg::format::Pixel::RGB24,
        width,
        height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?);

    let mut decoded = ffmpeg::frame::Video::empty();
    let mut frame_idx: u64 = 0;

    let drain_packets = |encoder: &mut ffmpeg::encoder::Video,
                         output_ctx: &mut ffmpeg::format::context::Output| {
        let mut packet = ffmpeg::Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(video_out_idx);
            packet.rescale_ts(encoder_time_base, output_time_base);
            packet.write_interleaved(output_ctx).unwrap();
        }
    };

    let encode_one = |encoder: &mut ffmpeg::encoder::Video,
                      output_ctx: &mut ffmpeg::format::context::Output,
                      encode_scaler: &mut SendScaler,
                      img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
                      pts: Time| {
        let enc_pts = pts.rescale(input_time_base, encoder_time_base);
        let mut rgb_frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, width, height);
        let raw = img.as_raw();
        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data_mut(0);
        let row_bytes = width as usize * 3;
        for y in 0..height as usize {
            let src_start = y * row_bytes;
            let dst_start = y * stride;
            data[dst_start..dst_start + row_bytes]
                .copy_from_slice(&raw[src_start..src_start + row_bytes]);
        }
        let mut yuv_frame = ffmpeg::frame::Video::empty();
        encode_scaler.run(&rgb_frame, &mut yuv_frame).unwrap();
        yuv_frame.set_pts(Some(enc_pts));
        encoder.send_frame(&yuv_frame).unwrap();
        drain_packets(encoder, output_ctx);
    };

    let write_audio = |output_ctx: &mut ffmpeg::format::context::Output,
                       mut packet: ffmpeg::Packet| {
        if let (Some(out_idx), Some(in_tb), Some(out_tb)) =
            (audio_out_idx, audio_in_tb, audio_out_tb)
        {
            packet.set_stream(out_idx);
            packet.rescale_ts(in_tb, out_tb);
            packet.write_interleaved(output_ctx).unwrap();
        }
    };

    for (stream, packet) in input_ctx.packets() {
        let sidx = stream.index();
        if Some(sidx) == audio_in_idx {
            write_audio(&mut output_ctx, packet);
            continue;
        }
        if sidx != video_in_idx {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut decoded).is_ok() {
            let pts = decoded.pts().unwrap_or(0);
            let mut rgb_frame = ffmpeg::frame::Video::empty();
            decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
            let mut img = rgb_frame_to_imgbuf(&rgb_frame);
            paint(frame_idx, &mut img);
            encode_one(
                &mut encoder,
                &mut output_ctx,
                &mut encode_scaler,
                &img,
                pts,
            );
            progress(frame_idx, total);
            frame_idx += 1;
        }
    }

    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let pts = decoded.pts().unwrap_or(0);
        let mut rgb_frame = ffmpeg::frame::Video::empty();
        decode_scaler.run(&decoded, &mut rgb_frame).unwrap();
        let mut img = rgb_frame_to_imgbuf(&rgb_frame);
        paint(frame_idx, &mut img);
        encode_one(
            &mut encoder,
            &mut output_ctx,
            &mut encode_scaler,
            &img,
            pts,
        );
        progress(frame_idx, total);
        frame_idx += 1;
    }

    let _ = encoder.send_eof();
    drain_packets(&mut encoder, &mut output_ctx);
    output_ctx.write_trailer()?;
    Ok(())
}

/// Convenience: paint predictions from a `PredVideo` onto each frame during export.
/// Uses the "sticky" prediction (the most recent analyzed frame at or before the
/// current one), so unanalyzed frames inherit their neighbor's overlay.
pub fn export_video_with_predictions(
    pred_video: &super::abstractions::PredVideo,
    output_path: &Path,
    mut progress: impl FnMut(u64, u64),
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let input_str = pred_video
        .file_path
        .to_str()
        .ok_or("Non-UTF-8 input path")?;
    let total = pred_video.n_frames;

    let mut sticky: Vec<Option<AIOutputs>> = Vec::with_capacity(total as usize);
    let mut last: Option<AIOutputs> = None;
    for slot in pred_video.frames.iter() {
        if let Some(a) = slot {
            last = Some(a.clone());
        }
        sticky.push(last.clone());
    }

    export_annotated_video(
        input_str,
        output_path,
        total,
        |idx, img| {
            if let Some(Some(a)) = sticky.get(idx as usize) {
                if !a.is_empty() {
                    super::render::draw_aioutput(img, a);
                }
            }
        },
        |i, n| progress(i, n),
    )
}
