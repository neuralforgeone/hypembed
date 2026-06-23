mod common {
    include!("../../tests/common/mod.rs");
}

#[test]
fn write_tiny_fixture_creates_required_model_files() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    common::write_tiny_model_dir(dir.path());

    for file in ["config.json", "vocab.txt", "model.safetensors"] {
        let path = dir.path().join(file);
        assert!(path.is_file(), "missing fixture file: {}", path.display());
        assert!(std::fs::metadata(&path).unwrap().len() > 0, "empty fixture: {file}");
    }

    let config = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
    assert!(config.contains("hidden_size"));
}

#[test]
fn write_tiny_fixture_to_custom_path() {
    let base = tempfile::TempDir::new().expect("temp dir");
    let model_dir = base.path().join("nested").join("model");
    std::fs::create_dir_all(&model_dir).unwrap();
    common::write_tiny_model_dir(&model_dir);
    assert!(model_dir.join("vocab.txt").is_file());
}