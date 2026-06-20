use ab_glyph::{Font as _, FontRef};
use font_subset::FontReader;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// Stable, normalized dep dirs produced by `cargo xtask fetch`.
const FFMPEG_DIR: &str = "deps/ffmpeg";
const ORT_DIR: &str = "deps/onnxruntime";

pub fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../.."); // target/<profile>/, next to the binary

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

fn subset_font(out_dir: &Path) {
    const FONT_SRC: &str = "assets/NotoSansSC-Regular.ttf";
    const LOC_SRC: &str = "src/localization.rs";
    const HANZI_SRC: &str = "assets/common-hanzi.txt";

    println!("cargo:rerun-if-changed={FONT_SRC}");
    println!("cargo:rerun-if-changed={LOC_SRC}");
    println!("cargo:rerun-if-changed={HANZI_SRC}");

    let mut wanted: BTreeSet<char> = BTreeSet::new();

    // The source's chars are a superset of every string the UI can show.
    wanted.extend(std::fs::read_to_string(LOC_SRC).unwrap().chars());

    // Runtime text (filenames, model labels) isn't in the UI strings.
    for &(lo, hi) in &[
        (0x0020u32, 0x007E), // Basic Latin
        (0x00A0, 0x024F),    // Latin-1 Supplement + Latin Extended-A/-B
        (0x0300, 0x036F),    // Combining diacritics
        (0x1E00, 0x1EFF),    // Latin Extended Additional (Vietnamese)
        (0x2000, 0x206F),    // General punctuation
        (0x3000, 0x303F),    // CJK symbols & punctuation
        (0x3040, 0x30FF),    // Hiragana + Katakana
        (0xFF00, 0xFFEF),    // Halfwidth & fullwidth forms
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

    // font-subset returns an error for any char the font lacks, so drop them first.
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
