use crate::api::abstractions::AIOutputs;
use crate::api::eps::EP;
use crate::api::utils::create_predictions_file_path;
use ort::session::builder::GraphOptimizationLevel;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use std::path::Path;
use std::{fs, io};

// First formats
// then, some logic and checks
pub const IMAGE_FORMATS: [&'static str; 23] = [
    "bmp",  // Bitmap Image File
    "dib",  // Device Independent Bitmap (BMP alternative)
    "dds",  // DirectDraw Surface
    "ff",   // Farbfeld
    "gif",  // Graphics Interchange Format
    "hdr",  // High Dynamic Range
    "ico",  // Icon
    "cur",  // Cursor (similar to ICO)
    "jpg",  // JPEG
    "jpeg", // JPEG alternative extension
    "jpe",  // JPEG alternative extension
    "jfif", // JPEG File Interchange Format
    "exr",  // OpenEXR HDR
    "png",  // Portable Network Graphics
    "pnm",  // Portable Anymap
    "pbm",  // Portable Bitmap (PNM subset)
    "pgm",  // Portable Graymap (PNM subset)
    "ppm",  // Portable Pixmap (PNM subset)
    "qoi",  // Quite OK Image format
    "tga",  // Truevision Graphics Adapter
    "tiff", // Tagged Image File Format
    "tif",  // TIFF alternative extension
    "webp", // WebP
];

pub const VIDEO_FORMATS: [&'static str; 35] = [
    "mp4",    // MPEG-4 Part 14
    "m4v",    // MPEG-4 Video
    "mkv",    // Matroska Video
    "avi",    // Audio Video Interleave
    "mov",    // QuickTime Movie
    "qt",     // QuickTime alternative extension
    "wmv",    // Windows Media Video
    "flv",    // Flash Video
    "f4v",    // Flash MP4 Video
    "webm",   // WebM
    "mpg",    // MPEG-1 Video
    "mpeg",   // MPEG-1/2 Video
    "mpe",    // MPEG alternative extension
    "m1v",    // MPEG-1 Video
    "m2v",    // MPEG-2 Video
    "3gp",    // 3GPP Media
    "3g2",    // 3GPP2 Media
    "ts",     // MPEG Transport Stream
    "mts",    // AVCHD Video
    "m2ts",   // Blu-ray BDAV
    "mxf",    // Material Exchange Format
    "vob",    // DVD Video Object
    "asf",    // Advanced Systems Format
    "rm",     // RealMedia
    "rmvb",   // RealMedia Variable Bitrate
    "ogv",    // Ogg Video
    "ogg",    // Ogg container (can contain video)
    "divx",   // DivX Video
    "swf",    // Small Web Format (Flash)
    "wtv",    // Windows Recorded TV Show
    "dvr-ms", // Microsoft Digital Video Recording
    "amv",    // Anime Music Video
    "hevc",   // High Efficiency Video Coding
    "h265",   // H.265 Video
    "h264",   // H.264 Video
];

pub fn is_supported_img(file_path: &str) -> bool {
    if let Some(extension) = file_path.rsplit('.').next() {
        return IMAGE_FORMATS.contains(&extension.to_lowercase().as_str());
    }
    false
}

pub fn is_supported_videofile(file_path: &str) -> bool {
    if let Some(extension) = file_path.rsplit('.').next() {
        return VIDEO_FORMATS.contains(&extension.to_lowercase().as_str());
    }
    false
}

pub fn import_model(model_data: &[u8], ep: &EP) -> Result<Session, ort::Error> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;

    if ep.name == "CUDA" {
        builder = builder
            .with_execution_providers([CUDAExecutionProvider::default().build()])?;
    }

    builder.commit_from_memory(model_data)
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
