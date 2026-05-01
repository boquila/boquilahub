use super::abstractions::{AIOutputs, ModelConfig, AI};
use super::audio::*;
use super::eps::Ep;
use super::import::import_model;
use super::models::{AIInput, Model, Task};
use super::processing::post::PostProcessing;
use super::processing::pre::slice_image;
use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::{OnceLock, RwLock};

// Just an interface for interacting with .bq models
pub struct BQModel;

// An interface for interacting with .bq models In Memory
pub enum GlobalBQ {
    First,
    Second,
}

impl GlobalBQ {
    fn get_lock(&self) -> &'static RwLock<Option<Model>> {
        match self {
            GlobalBQ::First => &CURRENT_AI,
            GlobalBQ::Second => &CURRENT_AI2,
        }
    }

    pub fn set_model(
        &self,
        value: impl AsRef<Path>,
        ep: Ep,
        config: Option<ModelConfig>,
    ) -> Result<()> {
        let config = config.unwrap_or_default();

        let (model_metadata, data): (AI, Vec<u8>) = BQModel::import_data(value).unwrap();
        debug_assert!(model_metadata.architecture.is_some());
        let session = import_model(&data, ep).unwrap();
        let post: Vec<PostProcessing> = model_metadata
            .post_processing
            .iter()
            .map(|s| PostProcessing::from(s.as_str()))
            .filter(|t| !matches!(t, PostProcessing::None))
            .collect();
        let aimodel: Model = Model::new(
            model_metadata.classes,
            Task::from(model_metadata.task.as_str()),
            post,
            session,
            model_metadata.architecture,
            config,
        )?;
        *self.get_lock().write().unwrap() = Some(aimodel);
        Ok(())
    }

    pub fn update_config(&self, new_config: ModelConfig) {
        if let Some(ref mut model) = self.get_lock().write().unwrap().as_mut() {
            *model.config_mut() = new_config;
        }
    }

    pub fn clear(&self) {
        let mut guard = self.get_lock().write().unwrap();
        *guard = None;
    }
}

impl BQModel {
    pub fn import_data(file_path: impl AsRef<Path>) -> io::Result<(AI, Vec<u8>)> {
        // Open the .bq file
        let mut file = File::open(&file_path)?;
        let mut file_content = Vec::new();
        file.read_to_end(&mut file_content)?;

        // Validate magic string
        if &file_content[..7] != b"BQMODEL" {
            panic!("Invalid file format: missing BQMODEL magic string");
        }

        // Read version (1 byte)
        let version = file_content[7];
        if version != 1 {
            panic!("Unsupported version: {}", version);
        }

        // Read JSON section length (4 bytes, little-endian)
        let json_length = u32::from_le_bytes(file_content[8..12].try_into().unwrap()) as usize;

        // Extract JSON section
        let json_start = 12;
        let json_end = json_start + json_length;
        let json_data = &file_content[json_start..json_end];
        let json_str = String::from_utf8(json_data.to_vec())
            .unwrap_or_else(|_| panic!("Failed to parse JSON content"));

        // Deserialize JSON into AImodel
        let mut ai_model: AI = serde_json::from_str(&json_str)
            .unwrap_or_else(|_| panic!("Failed to deserialize JSON into AImodel"));

        // Extract the name from the file path
        let name = file_path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_string();

        // Set the name field
        ai_model.name = name;

        // Read ONNX section length (4 bytes, little-endian)
        let onnx_length_start = json_end;
        let onnx_length = u32::from_le_bytes(
            file_content[onnx_length_start..onnx_length_start + 4]
                .try_into()
                .unwrap(),
        ) as usize;

        // Extract ONNX section
        let onnx_start = onnx_length_start + 4;
        let onnx_end = onnx_start + onnx_length;
        let onnx_data = file_content[onnx_start..onnx_end].to_vec();
        Ok((ai_model, onnx_data))
    }

    pub fn form_file_to_metadata(file_path: &str) -> io::Result<AI> {
        let mut file = File::open(file_path)?;

        // Read header (magic + version + length)
        let mut header = [0u8; 12];
        file.read_exact(&mut header)?;

        // Validate magic string
        if &header[..7] != b"BQMODEL" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid file format",
            ));
        }

        // Check version
        if header[7] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported version",
            ));
        }

        // Get JSON length
        let json_length = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

        // Read only the JSON section
        let mut json_data = vec![0u8; json_length];
        file.read_exact(&mut json_data)?;

        let json_str = String::from_utf8(json_data)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Failed to parse JSON"))?;

        // Deserialize JSON into AImodel
        let mut ai_model: AI = serde_json::from_str(&json_str)
            .unwrap_or_else(|_| panic!("Failed to deserialize JSON into AImodel"));

        // Extract the name from the file path
        let path = std::path::Path::new(file_path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_string();

        // Set the name field
        ai_model.name = name;

        return Ok(ai_model);
    }

    pub fn get_bqs() -> Vec<AI> {
        analyze_folder("models/").unwrap()
    }

    pub fn clear_second() {
        let mut guard = CURRENT_AI2.write().unwrap();
        *guard = None;
    }

    pub fn from_file_print_shape(model_path: impl AsRef<Path>) -> Result<()> {
        let path = model_path.as_ref();

        let model_data = match path.extension().and_then(|e| e.to_str()) {
            Some("onnx") => fs::read(path)?,
            Some("bq") => {
                let (_metadata, data) = BQModel::import_data(path)?;
                data
            }
            Some(ext) => return Err(anyhow::anyhow!("Unsupported extension: .{}", ext)),
            None => return Err(anyhow::anyhow!("No file extension found")),
        };

        let session = ort::session::Session::builder()?.commit_from_memory(&model_data)?;

        println!("Inputs:\n{:?}", session.inputs);
        println!("Outputs:\n{:?}", session.outputs);

        Ok(())
    }

    pub fn create_bq_file(name: String) -> Result<()> {
        let json_path = format!("{}.json", name);
        let onnx_path = format!("{}.onnx", name);
        let output_path = format!("{}.bq", name);

        let json_content = fs::read(&json_path).context(format!("Failed to open {}", json_path))?;
        let _ai: AI = serde_json::from_slice(&json_content).context(format!(
            "Failed to deserialize {} into required AI metadata",
            json_path
        ))?;
        let onnx_content = fs::read(&onnx_path).context(format!("Failed to open {}", onnx_path))?;

        if !matches!(onnx_content.get(0..2), Some(&[0x08, ir_ver]) if ir_ver > 0) {
            anyhow::bail!(
                "File {} does not appear to be a valid ONNX model",
                onnx_path
            );
        }

        let mut output_file = File::create(&output_path)
            .unwrap_or_else(|_| panic!("Failed to create output file: {}", output_path));

        output_file.write_all(b"BQMODEL")?; // Magic string
        output_file.write_all(&[1])?; // Version

        let json_length = json_content.len() as u32;
        output_file.write_all(&json_length.to_le_bytes())?;
        output_file.write_all(&json_content)?;

        let onnx_length = onnx_content.len() as u32;
        output_file.write_all(&onnx_length.to_le_bytes())?;
        output_file.write_all(&onnx_content)?;

        println!("New model: {}", output_path);
        Ok(())
    }
}

fn analyze_folder(folder_path: &str) -> io::Result<Vec<AI>> {
    // Validate the folder path
    let path = Path::new(folder_path);
    if !path.is_dir() || !path.exists() {
        if let Err(error) = fs::create_dir(folder_path) {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Could not create {folder_path} with error: {error}"),
            ));
        }
    }

    // Collect BQ files and process them
    let mut ai_models = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        // Check if the file has a .bq extension
        if let Some(extension) = file_path.extension() {
            if extension == "bq" {
                match BQModel::form_file_to_metadata(file_path.to_str().unwrap()) {
                    Ok(model) => ai_models.push(model),
                    Err(e) => eprintln!("Error processing file {:?}: {}", file_path, e),
                }
            }
        }
    }

    Ok(ai_models)
}

pub static GEOFENCE_DATA: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

pub fn init_geofence_data() -> Result<(), Box<dyn std::error::Error>> {
    if GEOFENCE_DATA.get().is_some() {
        return Ok(());
    }

    let json_content = std::fs::read_to_string("assets/geofence.json")?;
    let geofence_map: HashMap<String, Vec<String>> = serde_json::from_str(&json_content)?;
    GEOFENCE_DATA
        .set(geofence_map)
        .map_err(|_| "Failed to initialize")?;
    Ok(())
}

pub static CURRENT_AI: RwLock<Option<Model>> = RwLock::new(None);
pub static CURRENT_AI2: RwLock<Option<Model>> = RwLock::new(None);

#[inline(always)]
pub fn process_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
    let mut outputs = CURRENT_AI
        .read()
        .unwrap()
        .as_ref()
        .unwrap()
        .run(&AIInput::Image(img));
    process_with_ai2(&mut outputs, img);
    return outputs;
}

#[inline(always)]
pub fn process_audio(audio: &AudioData) -> AIOutputs {
    let outputs = CURRENT_AI
        .read()
        .unwrap()
        .as_ref()
        .unwrap()
        .run(&AIInput::Audio(audio));
    return outputs;
}

fn process_with_ai2(outputs: &mut AIOutputs, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Option<()> {
    let ai2 = CURRENT_AI2.read().ok()?;
    let ai2_ref = ai2.as_ref()?;

    match outputs {
        AIOutputs::ObjectDetection(detections) => {
            for xyxyc in detections.iter_mut() {
                let sliced_img = slice_image(img, &xyxyc.xyxy);
                let cls_output = ai2_ref.run(&AIInput::Image(&sliced_img));
                if let AIOutputs::Classification(prob_space) = cls_output {
                    xyxyc.extra_cls = Some(prob_space);
                }
            }
        }
        AIOutputs::Segmentation(segmentations) => {
            for segc in segmentations {
                let xyxyc = &mut segc.bbox;
                let sliced_img = slice_image(img, &xyxyc.xyxy);
                let cls_output = ai2_ref.run(&AIInput::Image(&sliced_img));
                if let AIOutputs::Classification(prob_space) = cls_output {
                    xyxyc.extra_cls = Some(prob_space);
                }
            }
        }
        _ => {}
    }

    Some(())
}
