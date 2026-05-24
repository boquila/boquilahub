use super::abstractions::{PredAudio, PredImg, PredVideo};
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const EXPORT_DIR: &str = "export";

impl PredImg {
    pub fn save(&self) -> Result<()> {
        let img_data = self.draw()?;
        let filename = prepare_export_img(&self.file_path);
        img_data.save(&filename)?;
        Ok(())
    }

    // For file 'img.jpg', creates a file 'img_predictions.json' that contains the AI outputs
    pub async fn write_pred_img_to_file(&self) -> Result<()> {
        let output_path = self.predictions_file_path()?;
        let mut file = File::create(&output_path)?;
        let json_string = serde_json::to_string(&self.aioutput)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }
}

impl PredVideo {
    // For file 'video.mp4', creates a file 'video_predictions.json' containing this PredVideo.
    pub async fn write_pred_video_to_file(&self) -> Result<()> {
        let output_path = self.predictions_file_path()?;
        let mut file = File::create(&output_path)?;
        let json_string = serde_json::to_string(self)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }
}

impl PredAudio {
    // For file 'sound.wav', creates a file 'sound_predictions.json' containing the AI outputs.
    pub async fn write_pred_audio_to_file(&self) -> Result<()> {
        let output_path = self.predictions_file_path()?;
        let mut file = File::create(&output_path)?;
        let json_string = serde_json::to_string(&self.aioutput)?;
        file.write_all(json_string.as_bytes())?;
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
