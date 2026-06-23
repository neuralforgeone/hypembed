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
    let gitignore = std::fs::read_to_string(repo_root().join(".gitignore"))
        .expect("read .gitignore");

    assert!(
        gitignore.lines().any(|line| line.trim() == "/mcps/"),
        ".gitignore must contain a /mcps/ entry"
    );
}