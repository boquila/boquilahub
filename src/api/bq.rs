use super::abstractions::AI;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

pub fn import_bq(file_path: &str) -> io::Result<(AI, Vec<u8>)> {
    // Open the .bq file
    let mut file = File::open(file_path)?;
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
    let path = std::path::Path::new(file_path);
    let name = path
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

pub fn get_ai_model(file_path: &str) -> io::Result<AI> {
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
    
    return Ok(ai_model)
}

fn analyze_folder(folder_path: &str) -> io::Result<Vec<AI>> {
    // Validate the folder path
    let path = Path::new(folder_path);
    if !path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Specified path is not a directory",
        ));
    }

    // Collect BQ files and process them
    let mut ai_models = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        // Check if the file has a .bq extension
        if let Some(extension) = file_path.extension() {
            if extension == "bq" {
                match get_ai_model(file_path.to_str().unwrap()) {
                    Ok(model) => ai_models.push(model),
                    Err(e) => eprintln!("Error processing file {:?}: {}", file_path, e),
                }
            }
        }
    }

    Ok(ai_models)
}

pub fn get_bqs() -> Vec<AI> {
    analyze_folder("models/").unwrap()
}
