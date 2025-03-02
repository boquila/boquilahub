use regex::Regex;
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::str;

#[derive(Clone)]
pub struct EP {
    pub name: String,
    pub description: String,
    pub img_path: String,
    pub version: f32, // the version that is accepted by BoquilaHUB
    pub local: bool,
    pub dependencies: String,
}

pub fn get_ep_version(provider: &EP) -> f64 {
    match provider.name.as_str() {
        "CUDA" => {
            let output = Command::new("nvcc").args(["--version"]).creation_flags(0).output().unwrap();

            let output_text = match str::from_utf8(&output.stdout) {
                Ok(v) => v,
                Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
            };

            let version_regex = Regex::new(r"release (\d+\.\d+),").unwrap();

            if let Some(captures) = version_regex.captures(output_text) {
                if let Some(version_str) = captures.get(1) {
                    // Convert the version string to a float
                    return version_str.as_str().parse::<f64>().unwrap_or(0.0);
                }
            }
            0.0 // Return 0.0 if no match is found
        }
        "CPU" => {
            todo!();
        }
        "ROCm" => {
            todo!();
        }
        "VitisAI" => {
            todo!();
        }
        "TensorRT" => {
            todo!();
        }
        "BoquilaHUBRemoto" => {
            todo!();
        } 
        _ => 0.0, // Default case
    }
}

#[flutter_rust_bridge::frb(sync)]
pub fn get_ep_by_name(list_eps: &[EP], name: &str) -> EP {
    list_eps.iter().find(|ep| ep.name == name).unwrap().clone()
}
