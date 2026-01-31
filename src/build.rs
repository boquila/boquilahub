pub fn main() {
    // Move geofence.json
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_path = out_dir.join("../../../assets/geofence.json");
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::copy("assets/geofence.json", target_path).unwrap();
    println!("cargo:rerun-if-changed=assets/geofence.json");

    // Boquila Icon
    embed_resource::compile("assets/app.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}
