use anyhow::Error;
use std::process::Command;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ep {
    #[default]
    Cpu,
    Cuda,
    BoquilaHubRemote,
}

impl Ep {
    pub const fn name(&self) -> &'static str {
        match self {
            Ep::Cpu => "CPU",
            Ep::Cuda => "CUDA",
            Ep::BoquilaHubRemote => "BoquilaHUB Remote",
        }
    }

    pub const fn is_local(&self) -> bool {
        !matches!(self, Ep::BoquilaHubRemote)
    }

    pub const fn dependencies(&self) -> &'static str {
        match self {
            Ep::Cuda => "cuDNN",
            _ => "none",
        }
    }

    pub fn version(&self) -> Result<f32, Error> {
        match self {
            Ep::Cuda => get_cuda_version(),
            _ => Ok(0.0),
        }
    }
}

fn get_cuda_version() -> Result<f32, Error> {
    let mut cmd = Command::new("nvcc");
    cmd.args(["--version"]);

    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        use std::os::windows::process::CommandExt;        
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output()?;

    let output_text = match std::str::from_utf8(&output.stdout) {
        Ok(v) => v,
        Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
    };

    let version_regex = regex::Regex::new(r"release (\d+\.\d+),")?;

    if let Some(captures) = version_regex.captures(output_text) {
        if let Some(version_str) = captures.get(1) {
            return Ok(version_str.as_str().parse::<f32>().unwrap_or(0.0));
        }
    }
    Ok(0.0)
}
