use super::abstractions::*;
use super::audio::*;
use super::models::{AIInput, Model, Task};
use super::processing::post::PostProcessing;
use super::processing::pre::slice_image;
use anyhow::{bail, ensure, Context, Result};
use image::{ImageBuffer, Rgb};
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[cfg(feature = "webgpu")]
use ort::ep::WebGPU;
use ort::session::Session;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

pub(crate) fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

pub struct BQModel;

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

        let (model_metadata, data) = BQModel::import_data(value)?;
        
        let session = BQModel::session_from_memory(&data, ep)?;
        let aimodel: Model = Model::new(
            model_metadata,
            session,
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

    pub fn run(&self, input: &AIInput) -> AIOutputs {
        self.get_lock().read().unwrap().as_ref().unwrap().run(input)
    }
}

fn parse_bq_header(content: &[u8], file_stem: &str) -> Result<(AIMetadata, usize)> {
    ensure!(content.len() >= 7, "File too short to be a valid .bq file");
    ensure!(&content[..7] == b"BQMODEL", "Invalid file format: missing BQMODEL magic string");
    ensure!(content[7] == 1, "Unsupported .bq version: {}", content[7]);
    ensure!(content.len() >= 12, "File too short: missing JSON length");

    let json_length = u32::from_le_bytes(content[8..12].try_into().context("Failed to read JSON length")?) as usize;
    let json_end = 12 + json_length;
    ensure!(content.len() >= json_end, "File truncated: JSON section extends beyond file end");

    let json_str = String::from_utf8(content[12..json_end].to_vec())
        .context("Failed to parse JSON content in .bq file")?;
    let ai_model: AIMetadataRaw = serde_json::from_str(&json_str)
        .context("Failed to deserialize JSON into AI metadata")?;
    let ai_model = ai_model.cook(file_stem);
    Ok((ai_model, json_end))
}

impl BQModel {
    pub fn session_from_memory(model_data: &[u8], ep: Ep) -> Result<Session> {
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_err)?;
        match ep {
            #[cfg(feature = "cuda")]
            Ep::Cuda => builder = builder.with_execution_providers([CUDA::default().build().error_on_failure()]).map_err(ort_err)?,
            #[cfg(feature = "webgpu")]
            Ep::WebGPU => builder = builder.with_execution_providers([WebGPU::default().build().error_on_failure()]).map_err(ort_err)?,
            _ => {}
        }
        Ok(builder.commit_from_memory(model_data)?)
    }

    pub fn import_data(file_path: impl AsRef<Path>) -> Result<(AIMetadata, Vec<u8>)> {
        let content = fs::read(&file_path)
            .with_context(|| format!("Failed to read .bq file: {}", file_path.as_ref().display()))?;
        let name = file_path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let (ai_model, json_end) = parse_bq_header(&content, name)?;

        let onnx_length_start = json_end;
        ensure!(
            content.len() >= onnx_length_start + 4,
            "File truncated: missing ONNX length"
        );
        let onnx_length = u32::from_le_bytes(
            content[onnx_length_start..onnx_length_start + 4].try_into()?,
        ) as usize;
        let onnx_start = onnx_length_start + 4;
        let onnx_end = onnx_start + onnx_length;
        ensure!(
            content.len() >= onnx_end,
            "File truncated: ONNX section extends beyond file end"
        );
        let onnx_data = content[onnx_start..onnx_end].to_vec();

        Ok((ai_model, onnx_data))
    }

    pub fn from_file_to_metadata(file_path: impl AsRef<Path>) -> Result<AIMetadata> {
        let path = file_path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let mut file = File::open(path)
            .with_context(|| format!("Failed to open .bq file: {}", path.display()))?;

        let mut header = [0u8; 12];
        file.read_exact(&mut header)
            .with_context(|| format!("Failed to read .bq header: {}", path.display()))?;

        let json_length = u32::from_le_bytes(header[8..12].try_into().context("Failed to read JSON length")?) as usize;
        let mut buf = vec![0u8; 12 + json_length];
        buf[..12].copy_from_slice(&header);
        file.read_exact(&mut buf[12..])
            .with_context(|| format!("Failed to read JSON section from .bq file: {}", path.display()))?;

        let (ai_model, _) = parse_bq_header(&buf, name)?;
        Ok(ai_model)
    }

    pub fn get_list() -> Vec<AIMetadata> {
        analyze_folder("models/").unwrap_or_default()
    }

    pub fn from_file_print_shape(model_path: impl AsRef<Path>) -> Result<()> {
        let path = model_path.as_ref();

        let model_data = match path.extension().and_then(|e| e.to_str()) {
            Some("onnx") => fs::read(path)?,
            Some("bq") => {
                let (_metadata, data) = BQModel::import_data(path)?;
                data
            }
            Some(ext) => bail!("Unsupported extension: .{}", ext),
            None => bail!("No file extension found"),
        };

        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_memory(&model_data).map_err(ort_err)?;

        println!("Inputs:\n{:?}", session.inputs());
        println!("Outputs:\n{:?}", session.outputs());

        Ok(())
    }

    pub fn create_bq_file(name: String) -> Result<()> {
        let json_path = format!("{}.json", name);
        let onnx_path = format!("{}.onnx", name);
        let output_path = format!("{}.bq", name);

        let json_content = fs::read(&json_path).with_context(|| format!("Failed to open {}", json_path))?;
        let _ai: AIMetadataRaw = serde_json::from_slice(&json_content)
            .with_context(|| format!("Failed to deserialize {} into required AI metadata", json_path))?;
        let onnx_content = fs::read(&onnx_path).with_context(|| format!("Failed to open {}", onnx_path))?;

        if !matches!(onnx_content.get(0..2), Some(&[0x08, ir_ver]) if ir_ver > 0) {
            bail!("File {} does not appear to be a valid ONNX model", onnx_path);
        }

        let mut output = Vec::new();
        output.extend_from_slice(b"BQMODEL");
        output.push(1);

        let json_length = json_content.len() as u32;
        output.extend_from_slice(&json_length.to_le_bytes());
        output.extend_from_slice(&json_content);

        let onnx_length = onnx_content.len() as u32;
        output.extend_from_slice(&onnx_length.to_le_bytes());
        output.extend_from_slice(&onnx_content);

        fs::write(&output_path, output)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;

        println!("New model: {}", output_path);
        Ok(())
    }

    pub async fn get_list_from_api() -> Result<Vec<AvailableModel>> {
        let url = "https://boquila.org/api/models.json";
        let listmodels: Vec<AvailableModel> = reqwest::get(url).await?.json().await?;
        Ok(listmodels)
    }
}

fn analyze_folder(folder_path: &str) -> Result<Vec<AIMetadata>> {
    let path = Path::new(folder_path);
    if !path.is_dir() || !path.exists() {
        fs::create_dir(folder_path).context("Failed to create models directory")?;
    }

    let mut ai_models = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if let Some(extension) = file_path.extension() {
            if extension == "bq" {
                match BQModel::from_file_to_metadata(&file_path) {
                    Ok(model) => ai_models.push(model),
                    Err(e) => eprintln!("Error processing file {:?}: {}", file_path, e),
                }
            }
        }
    }

    Ok(ai_models)
}

pub fn init_geofence_data() -> Result<()> {
    if GEOFENCE_DATA.get().is_some() {
        return Ok(());
    }

    let json_content = fs::read_to_string("assets/geofence.json")
        .context("Failed to read geofence data")?;
    let geofence_map: HashMap<String, Vec<String>> = serde_json::from_str(&json_content)
        .context("Failed to parse geofence data")?;
    GEOFENCE_DATA
        .set(geofence_map)
        .map_err(|_| anyhow::anyhow!("Failed to initialize geofence data"))?;
    Ok(())
}

pub static CURRENT_AI: RwLock<Option<Model>> = RwLock::new(None);
pub static CURRENT_AI2: RwLock<Option<Model>> = RwLock::new(None);
pub static GEOFENCE_DATA: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();

#[inline(always)]
pub fn process_imgbuf(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
    let mut outputs = GlobalBQ::First.run(&AIInput::Image(img));
    process_with_ai2(&mut outputs, img);
    outputs
}

#[inline(always)]
pub fn process_audio(audio: &AudioData) -> AIOutputs {
    GlobalBQ::First.run(&AIInput::Audio(audio))
}

fn process_with_ai2(outputs: &mut AIOutputs, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Option<()> {
    let ai2 = CURRENT_AI2.read().ok()?;
    let ai2_ref = ai2.as_ref()?;

    match outputs {
        AIOutputs::ObjectDetection(detections) => {
            for xyxyc in detections.iter_mut() {
                let sliced_img = slice_image(img, &xyxyc.xyxy);
                let cls_output = ai2_ref.run(&AIInput::Image(&sliced_img));
                if let AIOutputs::Classification(probs) = cls_output {
                    xyxyc.extra_cls = Some(probs);
                }
            }
        }
        AIOutputs::Segmentation(segmentations) => {
            for segc in segmentations {
                let xyxyc = &mut segc.bbox;
                let sliced_img = slice_image(img, &xyxyc.xyxy);
                let cls_output = ai2_ref.run(&AIInput::Image(&sliced_img));
                if let AIOutputs::Classification(probs) = cls_output {
                    xyxyc.extra_cls = Some(probs);
                }
            }
        }
        _ => {}
    }

    Some(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ep {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "webgpu")]
    WebGPU,
    BoquilaHubRemote,
}

impl Ep {
    pub fn gpu() -> Ep {
        #[cfg(feature = "cuda")]
        { return Ep::Cuda; }
        #[cfg(feature = "webgpu")]
        { return Ep::WebGPU; }
    }

    pub const fn variants() -> &'static [Ep] {
        &[
            Ep::Cpu,
            #[cfg(feature = "cuda")]
            Ep::Cuda,
            #[cfg(feature = "webgpu")]
            Ep::WebGPU,
            Ep::BoquilaHubRemote,
        ]
    }

    pub fn locals() -> Vec<Ep> {
        Self::variants().iter().copied().filter(|e| e.is_local()).collect()
    }
    
    pub const fn name(&self) -> &'static str {
        match self {
            Ep::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Ep::Cuda => "CUDA",
            #[cfg(feature = "webgpu")]
            Ep::WebGPU => "GPU",
            Ep::BoquilaHubRemote => "BoquilaHUB Remote",
        }
    }
    
    pub const fn is_local(&self) -> bool {
        !matches!(self, Ep::BoquilaHubRemote)
    }
}

impl AsRef<str> for Ep {
    fn as_ref(&self) -> &str {
        self.name()
    }
}

impl AIMetadataRaw {
    pub fn cook(self, name: &str) -> AIMetadata {
        let post_processing = self
            .post_processing
            .iter()
            .map(|s| PostProcessing::from(s.as_str()))
            .filter(|t| !matches!(t, PostProcessing::None))
            .collect();

        let modality = match self.modality.as_deref() {
            Some("audio") => Modality::Audio,
            _ => Modality::Image,
        };

        let task = Task::from(self.task.as_str());

        let architecture = self.architecture.unwrap_or_else(|| "yolo".to_string());

        AIMetadata {
            task: task,
            architecture: architecture,
            post_processing,
            classes: self.classes,
            name: name.to_owned(),
            modality,
            audio_config: self.audio_config,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AIMetadata {
    pub task: Task,
    pub architecture: String, 
    pub post_processing: Vec<PostProcessing>,
    pub classes: Vec<String>,
    pub name: String,
    pub modality: Modality,
    pub audio_config: Option<AudioConfig>,
}

impl AIMetadata {
    pub fn get_path(&self) -> String {
        format!("models/{}.bq", self.name)
    }
}

impl AsRef<str> for AIMetadata {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Audio,
    Image,
}