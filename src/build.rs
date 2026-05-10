fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_path = out_dir.join("../../../assets/geofence.json");
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::copy("assets/geofence.json", target_path).unwrap();
    println!("cargo:rerun-if-changed=assets/geofence.json");

    embed_resource::compile("assets/app.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();

    if std::env::var("CARGO_FEATURE_GRPC").is_ok() {
        tonic_build::configure()
            .compile_protos(&["assets/boquila.proto"], &["assets/"])
            .expect("Failed to compile boquila.proto");
        println!("cargo:rerun-if-changed=assets/boquila.proto");
    }
}