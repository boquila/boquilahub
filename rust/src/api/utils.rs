use std::process::Command;
use std::str;
use regex::Regex;

pub fn get_cuda_version() -> f64 {
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
