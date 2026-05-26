use super::abstractions::PredImg;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub const EXPORT_DIR: &str = "export";

impl PredImg {
    pub fn save(&self) -> Result<()> {
        let img_data = self.draw()?;
        let filename = prepare_export_img(&self.file_path);
        img_data.save(&filename)?;
        Ok(())
    }
}

pub fn prepare_export_img(path: &PathBuf) -> String {
    std::fs::create_dir_all(EXPORT_DIR).expect("Failed to create export directory");
    return format!(
        "{}/exported_{}.jpg",
        EXPORT_DIR,
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
    );
}

/// `export/exported_<original-filename>.<ext>` for a given input file. Used for
/// annotated video exports so they land next to the image exports rather than
/// being scattered next to the source.
pub fn prepare_export_video(path: &Path) -> PathBuf {
    std::fs::create_dir_all(EXPORT_DIR).expect("Failed to create export directory");
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "exported_video".to_string());
    PathBuf::from(format!("{}/exported_{}", EXPORT_DIR, name))
}
