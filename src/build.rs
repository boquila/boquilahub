use std::path::{Path, PathBuf};

// Stable, normalized dep dirs produced by `cargo xtask fetch`.
const FFMPEG_DIR: &str = "deps/ffmpeg";
const ORT_DIR: &str = "deps/onnxruntime";

pub fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../.."); // target/<profile>/, next to the binary

    require_deps();

    #[cfg(windows)]
    {
        copy_ffmpeg_libs(&target_dir);
        add_icon()
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        copy_ffmpeg_libs(&target_dir);
    }

    copy_onnxruntime_libs(&target_dir);

    copy_geofence(&target_dir)
}

// The native deps must exist before this crate (and ffmpeg-sys-next) builds.
fn require_deps() {
    for dir in [FFMPEG_DIR, ORT_DIR] {
        assert!(
            Path::new(dir).exists(),
            "missing native deps: `{dir}` not found.\n\
             Run `cargo xtask fetch` first to download ffmpeg + ONNX Runtime."
        );
    }
}

fn copy_geofence(target_dir: &Path) {
    std::fs::create_dir_all(target_dir.join("assets")).unwrap();
    std::fs::copy("assets/geofence.json", target_dir.join("assets/geofence.json")).unwrap();
}

#[cfg(windows)]
fn add_icon() {
    embed_resource::compile("assets/app.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}

// Copy the ffmpeg shared libraries next to the binary.
// Windows ships its DLLs in bin/, Linux its .so files in lib/.
#[cfg(any(windows, target_os = "linux"))]
fn copy_ffmpeg_libs(target_dir: &Path) {
    #[cfg(windows)]
    let (subdir, ext) = ("bin", ".dll");
    #[cfg(target_os = "linux")]
    let (subdir, ext) = ("lib", ".so");

    for entry in std::fs::read_dir(format!("{FFMPEG_DIR}/{subdir}")).expect("read ffmpeg dir") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // Linux files look like libavcodec.so.61, so match on the name.
        if name.contains(ext) {
            let dest = target_dir.join(&name);
            if !dest.exists() {
                std::fs::copy(&path, &dest).unwrap();
            }
        }
    }
}

// Copy the ONNX Runtime shared libraries (skipping TensorRT) next to the binary.
fn copy_onnxruntime_libs(target_dir: &Path) {
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
