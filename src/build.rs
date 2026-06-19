use std::path::PathBuf;

#[cfg(windows)]
const FFMPEG_DIR: &str = "deps/ffmpeg-7.1.1-full_build-shared";

pub fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../.."); // target/<profile>/, next to the binary

    #[cfg(windows)]
    {
        ensure_ffmpeg(&target_dir);
        add_icon()
    }

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    copy_geofence(&target_dir)
}

fn copy_geofence(target_dir: &std::path::Path) {
    std::fs::create_dir_all(target_dir.join("assets")).unwrap();
    std::fs::copy(
        "assets/geofence.json",
        target_dir.join("assets/geofence.json"),
    )
    .unwrap();
    println!("cargo:rerun-if-changed=assets/geofence.json");
}

fn add_icon(){
    embed_resource::compile("assets/app.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}

// Download FFmpeg 7.1.1 into deps/ if it's missing.
#[cfg(windows)]
fn ensure_ffmpeg(target_dir: &std::path::Path) {
    println!("cargo:rerun-if-changed={FFMPEG_DIR}");
    if std::path::Path::new(FFMPEG_DIR).exists() {
        return;
    }
    println!("cargo:warning=FFmpeg not found in deps/, downloading 7.1.1 ...");
    std::fs::create_dir_all("deps").unwrap();
    // curl.exe + a standalone 7zr.exe, so there are no extra build-dependencies.
    run("curl.exe", &["-L", "-o", "deps/ffmpeg.7z",
        "https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.7z"]);
    run(
        "curl.exe",
        &[
            "-sL",
            "-o",
            "deps/7zr.exe",
            "https://www.7-zip.org/a/7zr.exe",
        ],
    );
    run("deps/7zr.exe", &["x", "deps/ffmpeg.7z", "-odeps", "-y"]);
    let _ = std::fs::remove_file("deps/ffmpeg.7z");
    let _ = std::fs::remove_file("deps/7zr.exe");

    assert!(
        std::path::Path::new(FFMPEG_DIR).exists(),
        "FFmpeg setup failed: {FFMPEG_DIR} missing after download"
    );
    copy_ffmpeg_dlls(&target_dir);
}

#[cfg(windows)]
fn copy_ffmpeg_dlls(target_dir: &std::path::Path) {
    for entry in std::fs::read_dir(format!("{FFMPEG_DIR}/bin")).expect("read ffmpeg bin/") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("dll") {
            std::fs::copy(&path, target_dir.join(path.file_name().unwrap())).unwrap();
        }
    }
}

#[cfg(windows)]
fn run(cmd: &str, args: &[&str]) {
    let status = std::process::Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch {cmd}: {e}"));
    assert!(status.success(), "{cmd} exited with {status}");
}
