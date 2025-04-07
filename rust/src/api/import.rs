use super::file_formats::*;

#[flutter_rust_bridge::frb(sync)]
pub fn is_supported_img(file_path: &str) -> bool {
    if let Some(extension) = file_path.rsplit('.').next() {
        return IMAGE_FORMATS.contains(&extension.to_lowercase().as_str());
    }
    false
}

#[flutter_rust_bridge::frb(sync)]
pub fn is_supported_videofile(file_path: &str) -> bool {
    if let Some(extension) = file_path.rsplit('.').next() {
        return VIDEO_FORMATS.contains(&extension.to_lowercase().as_str());
    }
    false
}