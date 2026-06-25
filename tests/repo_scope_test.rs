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
fn public_api_surface_uses_only_documented_reexports() {
    use hypembed::{Embedder, EmbeddingOptions, HypEmbedError, PoolingStrategy};

    let _ = std::any::type_name::<Embedder>();
    let _ = std::any::type_name::<EmbeddingOptions>();
    let _ = std::any::type_name::<PoolingStrategy>();
    let _ = std::any::type_name::<HypEmbedError>();
}
