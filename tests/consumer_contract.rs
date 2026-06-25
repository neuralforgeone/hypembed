//! Verifies that an external consumer crate cannot import internal hypembed modules.

use std::process::Command;

#[test]
fn external_crate_cannot_import_internal_tensor_module() {
    let scratch = std::env::temp_dir().join(format!("hypembed-consumer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(scratch.join("src")).expect("create consumer src");

    std::fs::write(
        scratch.join("Cargo.toml"),
        r#"
[package]
name = "hypembed-consumer-check"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
hypembed = { path = "REPLACE_ROOT" }
"#,
    )
    .expect("write consumer Cargo.toml");

    std::fs::write(
        scratch.join("src/main.rs"),
        r#"
fn main() {
    let _ = hypembed::tensor::Tensor::zeros;
}
"#,
    )
    .expect("write consumer main.rs");

    let manifest = std::fs::read_to_string(scratch.join("Cargo.toml")).expect("read manifest");
    let root = env!("CARGO_MANIFEST_DIR").replace('\\', "/");
    std::fs::write(
        scratch.join("Cargo.toml"),
        manifest.replace("REPLACE_ROOT", &root),
    )
    .expect("patch manifest path");

    let output = Command::new("cargo")
        .args(["check", "--manifest-path"])
        .arg(scratch.join("Cargo.toml"))
        .output()
        .expect("cargo check consumer");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "consumer importing hypembed::tensor must fail to compile; stderr={stderr}"
    );
    assert!(
        stderr.contains("tensor") || stderr.contains("private"),
        "expected private-module error in stderr, got: {stderr}"
    );
}
