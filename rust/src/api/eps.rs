use regex::Regex;
use std::process::Command;
use std::str;


#[derive(Debug, Clone)]
pub struct EP {
    pub name: String,
    pub description: String,
    pub img_path: String,
    pub version: f32,      // the version that is accepted by BoquilaHUB
    pub dependencies: String, // the dependencies that are required
}

#[derive(Debug)]
pub enum ExecutionProviders {
    CUDA(EP),
    ROCm(EP),
}

impl ExecutionProviders {
    pub fn get_version(&self) -> f64 {
        match self {
            ExecutionProviders::CUDA(_) => {
                let output = Command::new("nvcc").args(["--version"]).output().unwrap();

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
            ExecutionProviders::ROCm(_) => {
                todo!();
            }
        }
    }
}

#[flutter_rust_bridge::frb(sync)]
pub fn get_ep_by_name(list_eps: &[EP], name: &str) -> EP {
    list_eps.iter().find(|ep| ep.name == name).unwrap().clone()
}