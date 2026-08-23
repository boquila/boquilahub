use boquilahub::api::bq::BQModel;

fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("boquilahub_bq_regression");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A file containing only the seven-byte `BQMODEL` magic used to pass the
/// `len >= 7` check and then panic reading `content[7]` for the version.
/// Both metadata loading paths must return an error instead.
#[test]
fn seven_byte_bqmodel_magic_is_rejected_not_panic() {
    let path = write_temp("seven.bq", b"BQMODEL");
    assert!(
        BQModel::import_data(&path).is_err(),
        "import_data must reject a seven-byte file"
    );
    assert!(
        BQModel::from_file_to_metadata(&path).is_err(),
        "from_file_to_metadata must reject a seven-byte file"
    );
}

/// Eight bytes (magic + version byte) still fail cleanly on the missing
/// JSON length field rather than panicking.
#[test]
fn eight_byte_bqmodel_file_is_rejected_not_panic() {
    let path = write_temp("eight.bq", b"BQMODEL\x01");
    assert!(
        BQModel::import_data(&path).is_err(),
        "import_data must reject a file with no JSON length"
    );
}
