use crate::api::abstractions::AIOutputs;
use crate::api::utils::create_predictions_file_path;
use std::path::Path;
use std::{fs, io};

pub const IMAGE_FORMATS: [&'static str; 23] = [
    "bmp", "dib", "dds", "ff", "gif", "hdr", "ico", "cur", "jpg", "jpeg", "jpe", "jfif", "exr",
    "png", "pnm", "pbm", "pgm", "ppm", "qoi", "tga", "tiff", "tif", "webp",
];

pub const VIDEO_FORMATS: [&'static str; 35] = [
    "mp4", "m4v", "mkv", "avi", "mov", "qt", "wmv", "flv", "f4v", "webm", "mpg", "mpeg", "mpe",
    "m1v", "m2v", "3gp", "3g2", "ts", "mts", "m2ts", "mxf", "vob", "asf", "rm", "rmvb", "ogv",
    "ogg", "divx", "swf", "wtv", "dvr-ms", "amv", "hevc", "h265", "h264",
];

pub const AUDIO_FORMATS: [&'static str; 18] = [
    "mp3", "wav", "flac", "ogg", "opus", "aac", "m4a", "wma", "aiff", "aif", "au", "snd", "amr",
    "ac3", "mid", "midi", "wv", "ape",
];

fn is_supported(file_path: &str, formats: &[&str]) -> bool {
    file_path.rsplit('.').next().map_or(false, |ext| formats.contains(&ext.to_lowercase().as_str()))
}

pub fn is_supported_audio(file_path: &str) -> bool {
    is_supported(file_path, &AUDIO_FORMATS)
}

pub fn is_supported_img(file_path: &str) -> bool {
    is_supported(file_path, &IMAGE_FORMATS)
}

pub fn is_supported_videofile(file_path: &str) -> bool {
    is_supported(file_path, &VIDEO_FORMATS)
}

pub fn read_predictions_from_file(input_path: &Path) -> io::Result<AIOutputs> {
    // Create expected filename based on input filepath
    let prediction_path = create_predictions_file_path(input_path)?;

    // Check if file exists
    if !prediction_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Prediction file not found",
        ));
    }

    // Read and deserialize the file
    let data = fs::read_to_string(prediction_path)?;
    let deserialized: AIOutputs = serde_json::from_str(&data)?;
    Ok(deserialized)
}
