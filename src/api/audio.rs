use serde::{Deserialize, Serialize};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::format::sample::Type as SampleType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn load_audio(path: impl AsRef<std::path::Path>) -> Result<AudioData, ffmpeg::Error> {
    ffmpeg::init()?;
    let mut ictx = ffmpeg::format::input(&path.as_ref())?;

    let stream = ictx.streams().best(ffmpeg::media::Type::Audio)
        .ok_or(ffmpeg::Error::StreamNotFound)?;
    let stream_index = stream.index();

    let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
        .decoder().audio()?;

    let mut resampler = decoder.resampler(
        Sample::F32(SampleType::Packed),
        decoder.channel_layout(),
        decoder.rate(),
    )?;

    let mut samples: Vec<f32> = Vec::new();

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index { continue; }
        decoder.send_packet(&packet)?;
        let mut decoded = ffmpeg::frame::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = ffmpeg::frame::Audio::empty();
            resampler.run(&decoded, &mut resampled)?;
            samples.extend(
                resampled.data(0).chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            );
        }
    }

    decoder.send_eof()?;
    let mut decoded = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let mut resampled = ffmpeg::frame::Audio::empty();
        resampler.run(&decoded, &mut resampled)?;
        samples.extend(
            resampled.data(0).chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        );
    }

    Ok(AudioData {
        samples,
        sample_rate: decoder.rate(),
        channels: decoder.channels(),
    })
}

#[test]
fn smoke() {
    let audio = load_audio("assets/test/audio.mp3").unwrap();
    println!("{:?}",audio)
}