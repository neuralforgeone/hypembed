use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn repo_scope_mcps_not_tracked_by_git() {
    let output = Command::new("git")
        .args(["ls-files", "--", "mcps/"])
        .current_dir(repo_root())
        .output()
        .expect("git must be available");

    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tracked = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        tracked.trim(),
        "",
        "mcps/ must not be tracked by git; found: {tracked:?}"
    );
}

#[test]
fn repo_scope_gitignore_excludes_mcps() {
    let gitignore =
        std::fs::read_to_string(repo_root().join(".gitignore")).expect("read .gitignore");

    assert!(
        gitignore.lines().any(|line| line.trim() == "/mcps/"),
        ".gitignore must contain a /mcps/ entry"
    );
}

#[test]
fn public_api_exports_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<hypembed::Embedder>();
    assert_send_sync::<hypembed::EmbeddingOptions>();
    assert_send_sync::<hypembed::PoolingStrategy>();
    assert_send_sync::<hypembed::HypEmbedError>();
}

#[test]
fn lib_rs_keeps_internals_pub_crate_not_public() {
    let lib_rs = std::fs::read_to_string(repo_root().join("src/lib.rs")).expect("read lib.rs");

    for module in ["tensor", "tokenizer", "model", "pipeline"] {
        assert!(
            lib_rs.contains(&format!("pub(crate) mod {module};")),
            "src/lib.rs must declare `pub(crate) mod {module};`"
        );
        assert!(
            !lib_rs.contains(&format!("#[doc(hidden)]\npub mod {module};")),
            "src/lib.rs must not expose public mod {module}"
        );
    }

    assert!(
        !lib_rs.contains("pub fn config("),
        "public config() must not be re-exported from lib.rs"
    );
}

#[test]
fn external_consumer_cannot_reference_internal_modules() {
    // Integration tests compile as an external crate. This file intentionally uses only
    // the stable re-exports; if tensor/model/tokenizer/pipeline were public `mod` again,
    // downstream could depend on them — the lib_rs_keeps_internals test guards that.
    use hypembed::{Embedder, EmbeddingOptions, HypEmbedError, PoolingStrategy};

    fn _stable_only(_: Embedder, _: EmbeddingOptions, _: PoolingStrategy, _: HypEmbedError) {}
}
