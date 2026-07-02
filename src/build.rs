use ab_glyph::{Font as _, FontRef};
use font_subset::FontReader;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const FFMPEG_DIR: &str = "deps/ffmpeg";
#[cfg(feature = "cuda")]
const ORT_DIR: &str = "deps/onnxruntime";

pub fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.ancestors().nth(3).unwrap().to_path_buf();

    require_deps();
    subset_font(&out_dir);

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

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");

    #[cfg(feature = "cuda")]
    copy_onnxruntime_libs(&target_dir);

    copy_geofence(&target_dir)
}

fn require_deps() {
    assert!(
        Path::new(FFMPEG_DIR).exists(),
        "missing native deps: `{FFMPEG_DIR}` not found.\n\
         Run `cargo xtask fetch` first to download ffmpeg."
    );

    #[cfg(feature = "cuda")]
    assert!(
        Path::new(ORT_DIR).exists(),
        "missing native deps: `{ORT_DIR}` not found.\n\
         Run `cargo xtask fetch` first to download ONNX Runtime."
    );
}

fn subset_font(out_dir: &Path) {
    const FONT_SRC: &str = "assets/NotoSansSC-Regular.ttf";
    const LOC_SRC: &str = "src/localization.rs";
    const HANZI_SRC: &str = "assets/common-hanzi.txt";

    println!("cargo:rerun-if-changed={FONT_SRC}");
    println!("cargo:rerun-if-changed={LOC_SRC}");
    println!("cargo:rerun-if-changed={HANZI_SRC}");

    let mut wanted: BTreeSet<char> = BTreeSet::new();

    wanted.extend(std::fs::read_to_string(LOC_SRC).unwrap().chars());

    for &(lo, hi) in &[
        (0x0020u32, 0x007E),
        (0x00A0, 0x024F),
        (0x0300, 0x036F),
        (0x1E00, 0x1EFF),
        (0x2000, 0x206F),
        (0x3000, 0x303F),
        (0x3040, 0x30FF),
        (0xFF00, 0xFFEF),
    ] {
        wanted.extend((lo..=hi).filter_map(char::from_u32));
    }

    match std::fs::read_to_string(HANZI_SRC) {
        Ok(list) => wanted.extend(list.chars().filter(|c| !c.is_whitespace())),
        Err(_) => println!(
            "cargo:warning={HANZI_SRC} missing: font subset covers UI text only — \
             Chinese filenames/model labels may render as boxes."
        ),
    }

    let bytes = std::fs::read(FONT_SRC).unwrap();
    let face = FontRef::try_from_slice(&bytes).expect("parse source font");
    wanted.retain(|&c| face.glyph_id(c).0 != 0);

    let subset = FontReader::new(bytes.as_slice())
        .expect("read source font")
        .read()
        .expect("parse source font")
        .subset(&wanted)
        .expect("subset font")
        .to_opentype();

    std::fs::write(out_dir.join("NotoSansSC-subset.ttf"), &subset).unwrap();
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

#[cfg(any(windows, target_os = "linux"))]
fn copy_ffmpeg_libs(target_dir: &Path) {
    #[cfg(windows)]
    let (subdir, ext) = ("bin", ".dll");
    #[cfg(target_os = "linux")]
    let (subdir, ext) = ("lib", ".so");

    for entry in std::fs::read_dir(format!("{FFMPEG_DIR}/{subdir}")).expect("read ffmpeg dir") {
        let entry = entry.unwrap();
        if !entry.file_type().map(|ft| ft.is_file() || ft.is_symlink()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.contains(ext) {
            let dest = target_dir.join(&name);
            if !dest.exists() {
                std::fs::copy(&path, &dest).unwrap();
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn copy_onnxruntime_libs(target_dir: &Path) {
    #[cfg(windows)]
    let ext = "dll";
    #[cfg(target_os = "linux")]
    let ext = "so";

    let lib_dir = format!("{ORT_DIR}/lib");
    for entry in std::fs::read_dir(&lib_dir).expect("read onnxruntime lib/") {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|ft| ft.is_file() || ft.is_symlink()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.contains(ext) && !name.contains("tensorrt") {
            let dest = target_dir.join(&name);
            if !dest.exists() {
                if let Err(e) = std::fs::copy(&path, &dest) {
                    println!("cargo:warning=failed to copy {}: {}", path.display(), e);
                }
            }
        }
    }
}
