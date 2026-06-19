use std::path::PathBuf;

#[cfg(windows)]
const FFMPEG_DIR: &str = "deps/ffmpeg-8.1.1-full_build-shared";

#[cfg(windows)]
const ORT_DIR: &str = "deps/onnxruntime-win-x64-gpu-1.26.0";
#[cfg(target_os = "linux")]
const ORT_DIR: &str = "deps/onnxruntime-linux-x64-gpu-1.26.0";

pub fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../.."); // target/<profile>/, next to the binary

    #[cfg(windows)]
    {
        ensure_ffmpeg();
        copy_ffmpeg_libs(&target_dir);
        add_icon()
    }

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    ensure_onnxruntime();
    copy_onnxruntime_libs(&target_dir);

    copy_geofence(&target_dir)
}

fn copy_geofence(target_dir: &std::path::Path) {
    std::fs::create_dir_all(target_dir.join("assets")).unwrap();
    std::fs::copy(
        "assets/geofence.json",
        target_dir.join("assets/geofence.json"),
    )
    .unwrap();
}

#[cfg(windows)]
fn add_icon() {
    embed_resource::compile("assets/app.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}

// Download FFmpeg 8.1.1 into deps/ if it's missing.
#[cfg(windows)]
fn ensure_ffmpeg() {
    if std::path::Path::new(FFMPEG_DIR).exists() {
        return;
    }
    println!("cargo:warning=FFmpeg not found in deps/, downloading 8.1.1 ...");
    std::fs::create_dir_all("deps").unwrap();
    // curl.exe + a standalone 7zr.exe, so there are no extra build-dependencies.
    run("curl.exe", &["-L", "-o", "deps/ffmpeg.7z",
        "https://github.com/GyanD/codexffmpeg/releases/download/8.1.1/ffmpeg-8.1.1-full_build-shared.7z"]);
    run("curl.exe", &["-sL", "-o", "deps/7zr.exe", "https://www.7-zip.org/a/7zr.exe"]);
    run("deps/7zr.exe", &["x", "deps/ffmpeg.7z", "-odeps", "-y"]);
    let _ = std::fs::remove_file("deps/ffmpeg.7z");
    let _ = std::fs::remove_file("deps/7zr.exe");
    assert!(
        std::path::Path::new(FFMPEG_DIR).exists(),
        "FFmpeg setup failed: {FFMPEG_DIR} missing after download"
    );
}

// Copy the FFmpeg shared libraries next to the binary.
#[cfg(windows)]
fn copy_ffmpeg_libs(target_dir: &std::path::Path) {
    for entry in std::fs::read_dir(format!("{FFMPEG_DIR}/bin")).expect("read ffmpeg bin/") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.ends_with(".dll") {
            let dest = target_dir.join(&name);
            if !dest.exists() {
                std::fs::copy(&path, &dest).unwrap();
            }
        }
    }
}

// Download ONNX Runtime 1.26.0 (GPU) into deps/ if it's missing. CUDA is kept;
// the TensorRT provider is skipped at copy time.
fn ensure_onnxruntime() {
    if std::path::Path::new(ORT_DIR).exists() {
        return;
    }
    println!("cargo:warning=ONNX Runtime not found in deps/, downloading 1.26.0 ...");
    std::fs::create_dir_all("deps").unwrap();
    #[cfg(windows)]
    {
        // curl.exe + tar.exe are built into Windows 10+, so no extra build-dependencies.
        run("curl.exe", &["-L", "-o", "deps/onnxruntime.zip",
            "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-win-x64-gpu-1.26.0.zip"]);
        run("tar.exe", &["-xf", "deps/onnxruntime.zip", "-C", "deps"]);
        let _ = std::fs::remove_file("deps/onnxruntime.zip");
    }
    #[cfg(target_os = "linux")]
    {
        run("curl", &["-L", "-o", "deps/onnxruntime.tgz",
            "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz"]);
        run("tar", &["xzf", "deps/onnxruntime.tgz", "-C", "deps"]);
        let _ = std::fs::remove_file("deps/onnxruntime.tgz");
    }
    assert!(
        std::path::Path::new(ORT_DIR).exists(),
        "ONNX Runtime setup failed: {ORT_DIR} missing after download"
    );
}

// Copy the ONNX Runtime shared libraries (skipping TensorRT) next to the binary.
fn copy_onnxruntime_libs(target_dir: &std::path::Path) {
    #[cfg(windows)]
    let ext = "dll";
    #[cfg(target_os = "linux")]
    let ext = "so";

    for entry in std::fs::read_dir(format!("{ORT_DIR}/lib")).expect("read onnxruntime lib/") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // On Linux files look like libonnxruntime.so.1.26.0, so match on the name.
        if name.contains(ext) && !name.contains("tensorrt") {
            let dest = target_dir.join(&name);
            if !dest.exists() {
                std::fs::copy(&path, &dest).unwrap();
            }
        }
    }
}

fn run(cmd: &str, args: &[&str]) {
    let status = std::process::Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch {cmd}: {e}"));
    assert!(status.success(), "{cmd} exited with {status}");
}
