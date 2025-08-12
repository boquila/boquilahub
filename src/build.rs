extern crate embed_resource;
use std::{env, fs, path::PathBuf};

pub fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_path = out_dir.join("../../../assets/geofence.json");
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::copy("assets/geofence.json", target_path).unwrap();
    println!("cargo:rerun-if-changed=assets/geofence.json");
    embed_resource::compile("assets/app.rc", embed_resource::NONE).manifest_optional().unwrap();
}