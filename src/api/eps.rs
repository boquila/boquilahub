use regex::Regex;
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::str;

#[derive(Clone)]
pub struct EP {
    pub name: &'static str,
    pub img_path: &'static str,
    pub version: f32,
    pub local: bool,
    pub dependencies: &'static str,
    pub ep_type: EPType,
}

pub const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn get_ep_version(provider: &EP) -> f64 {
    match provider.name {
        "CUDA" => {
            let output = Command::new("nvcc")
                .args(["--version"])
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .unwrap();

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

pub fn get_ep_by_name(list_eps: &[EP], name: &str) -> EP {
    list_eps.iter().find(|ep| ep.name == name).unwrap().clone()
}

pub static LIST_EPS: &[EP] = &[
    EP {
        name: "CPU",
        img_path: "tiny_cpu.png",
        version: 0.0,
        local: true,
        dependencies: "none",
        ep_type: EPType::CPU,
    },
    EP {
        name: "CUDA",
        img_path: "tiny_nvidia.png",
        version: 12.4,
        local: true,
        dependencies: "cuDNN",
        ep_type: EPType::CUDA,
    },
    EP {
        name: "BoquilaHUB Remote",
        img_path: "tiny_boquila.png",
        version: 0.0,
        local: false,
        dependencies: "none",
        ep_type: EPType::BoquilaHUBRemote,
    },
];

#[derive(Clone)]
pub enum EPType {
    CPU,
    CUDA,
    BoquilaHUBRemote,   
}