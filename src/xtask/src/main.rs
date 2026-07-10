// Workspace tasks. Run via `cargo xtask <cmd>` (alias in .cargo/config.toml).
//
//   cargo xtask fetch   Download native deps (ffmpeg + ONNX Runtime) into deps/.
//
// This runs as its own cargo invocation, so the deps are on disk *before*
// `cargo build` compiles ffmpeg-sys-next (which links ffmpeg at build time).
//
// The ffmpeg fetched here is a prebuilt shared lib, used for ordinary/debug
// builds. `cargo build --features ffmpeg-static` (release builds; see
// Cargo.toml) instead compiles ffmpeg from source and links it statically on
// Linux, ignoring this fetch for ffmpeg — it still runs, just unused.

// On macOS only ffmpeg is fetched (ORT comes from the `ort` crate), so the
// download/extract helpers below go unused there.
#![cfg_attr(target_os = "macos", allow(dead_code))]

use std::path::{Path, PathBuf};
use std::process::Command;

// Normalized destination dirs, used by boquilahub/build.rs and .cargo/config.toml.
const FFMPEG_DIR: &str = "ffmpeg";
const ORT_DIR: &str = "onnxruntime";

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| "fetch".into());
    match cmd.as_str() {
        "fetch" => fetch(),
        other => {
            eprintln!("unknown task: {other}\n\nusage: cargo xtask fetch");
            std::process::exit(1);
        }
    }
}

// deps/ lives at the workspace root, regardless of the caller's working dir.
fn deps_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // <root>/src/xtask
    manifest
        .parent()
        .and_then(Path::parent)
        .expect("xtask manifest should be at <root>/src/xtask")
        .join("deps")
}

fn fetch() {
    let deps = deps_dir();
    std::fs::create_dir_all(&deps).unwrap();
    ensure_ffmpeg(&deps);
    ensure_onnxruntime(&deps);
    println!("xtask: deps ready in {}", deps.display());
}

// ---- ffmpeg --------------------------------------------------------------

#[cfg(windows)]
fn ensure_ffmpeg(deps: &Path) {
    let dest = deps.join(FFMPEG_DIR);
    if dest.exists() {
        return;
    }
    println!("xtask: downloading ffmpeg 8.1.1 (windows) ...");
    let archive = deps.join("ffmpeg.7z");
    let z7 = deps.join("7zr.exe");
    // curl.exe + a standalone 7zr.exe, both available without extra tooling.
    run("curl.exe", &["-L", "-o", str(&archive),
        "https://github.com/GyanD/codexffmpeg/releases/download/8.1.1/ffmpeg-8.1.1-full_build-shared.7z"]);
    run("curl.exe", &["-sL", "-o", str(&z7), "https://www.7-zip.org/a/7zr.exe"]);
    run(str(&z7), &["x", str(&archive), &format!("-o{}", deps.display()), "-y"]);
    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_file(&z7);
    rename_into(deps, "ffmpeg-8.1.1-full_build-shared", FFMPEG_DIR);
    assert!(dest.exists(), "ffmpeg setup failed: {} missing", dest.display());
}

#[cfg(target_os = "linux")]
fn ensure_ffmpeg(deps: &Path) {
    let dest = deps.join(FFMPEG_DIR);
    if dest.exists() {
        return;
    }
    println!("xtask: downloading ffmpeg 8.1 (linux, BtbN gpl-shared) ...");
    let archive = deps.join("ffmpeg.tar.xz");
    run("curl", &["-L", "-o", str(&archive),
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz"]);
    run("tar", &["xf", str(&archive), "-C", str(deps)]);
    let _ = std::fs::remove_file(&archive);
    rename_into(deps, "ffmpeg-n8.1-latest-linux64-gpl-shared-8.1", FFMPEG_DIR);
    assert!(dest.exists(), "ffmpeg setup failed: {} missing", dest.display());
}

#[cfg(target_os = "macos")]
fn ensure_ffmpeg(deps: &Path) {
    let dest = deps.join(FFMPEG_DIR);
    if dest.exists() {
        return;
    }

    println!("xtask: linking Homebrew ffmpeg into deps/ffmpeg ...");
    let out = Command::new("brew")
        .args(["--prefix", "ffmpeg"])
        .output()
        .expect("failed to run `brew`; install Homebrew, then `brew install ffmpeg`");
    assert!(
        out.status.success(),
        "`brew --prefix ffmpeg` failed; run `brew install ffmpeg` first"
    );
    let prefix = String::from_utf8(out.stdout).expect("brew output not UTF-8");
    let prefix = Path::new(prefix.trim());
    assert!(
        prefix.join("include").exists() && prefix.join("lib").exists(),
        "Homebrew ffmpeg at {} lacks include/ or lib/; try `brew reinstall ffmpeg`",
        prefix.display()
    );
    std::os::unix::fs::symlink(prefix, &dest)
        .unwrap_or_else(|e| panic!("symlink {} -> {}: {e}", prefix.display(), dest.display()));
    assert!(dest.exists(), "ffmpeg setup failed: {} missing", dest.display());
}

// ---- ONNX Runtime --------------------------------------------------------
#[cfg(target_os = "macos")]
fn ensure_onnxruntime(_deps: &Path) {}

#[cfg(not(target_os = "macos"))]
fn ensure_onnxruntime(deps: &Path) {
    let dest = deps.join(ORT_DIR);
    if dest.exists() {
        return;
    }
    println!("xtask: downloading ONNX Runtime 1.26.0 (GPU) ...");
    #[cfg(windows)]
    {
        let archive = deps.join("onnxruntime.zip");
        run("curl.exe", &["-L", "-o", str(&archive),
            "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-win-x64-gpu-1.26.0.zip"]);
        run("tar.exe", &["-xf", str(&archive), "-C", str(deps)]);
        let _ = std::fs::remove_file(&archive);
        rename_into(deps, "onnxruntime-win-x64-gpu-1.26.0", ORT_DIR);
    }
    #[cfg(target_os = "linux")]
    {
        let archive = deps.join("onnxruntime.tgz");
        run("curl", &["-L", "-o", str(&archive),
            "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz"]);
        run("tar", &["xzf", str(&archive), "-C", str(deps)]);
        let _ = std::fs::remove_file(&archive);
        rename_into(deps, "onnxruntime-linux-x64-gpu-1.26.0", ORT_DIR);
    }
    assert!(dest.exists(), "ONNX Runtime setup failed: {} missing", dest.display());
}

// ---- helpers -------------------------------------------------------------

// Rename the freshly extracted, version-named folder to a stable name so the
// rest of the build can reference a single path across platforms/versions.
fn rename_into(deps: &Path, extracted: &str, stable: &str) {
    let from = deps.join(extracted);
    let to = deps.join(stable);
    if from.exists() {
        std::fs::rename(&from, &to)
            .unwrap_or_else(|e| panic!("rename {} -> {}: {e}", from.display(), to.display()));
    }
}

fn str(p: &Path) -> &str {
    p.to_str().expect("path is not valid UTF-8")
}

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to launch {cmd}: {e}"));
    assert!(status.success(), "{cmd} exited with {status}");
}
