use std::fs::File;
use std::io::{self, Read};
use super::abstractions::AImodel;

pub fn import_bq(file_path: &str) -> io::Result<(AImodel, Vec<u8>)> {
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
    let ai_model: AImodel = serde_json::from_str(&json_str)
        .unwrap_or_else(|_| panic!("Failed to deserialize JSON into AImodel"));

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