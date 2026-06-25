use std::path::{Path, PathBuf};

use hypembed::Embedder;
use walkdir::WalkDir;

use crate::chunk::{chunk_text, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS};
use crate::error::{HypeRagError, Result};
use crate::store::ChunkStore;

/// Index all `.md` and `.txt` files under `root`.
pub fn index_directory(
    embedder: &Embedder,
    store: &ChunkStore,
    root: &Path,
    recursive: bool,
) -> Result<usize> {
    store.clear_chunks()?;
    let mut files: Vec<PathBuf> = Vec::new();

    if recursive {
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() && is_text_doc(entry.path()) {
                files.push(entry.path().to_path_buf());
            }
        }
    } else if root.is_file() && is_text_doc(root) {
        files.push(root.to_path_buf());
    } else if root.is_dir() {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && is_text_doc(&path) {
                files.push(path);
            }
        }
    } else {
        return Err(HypeRagError::Other(format!(
            "path '{}' is not a text file or directory",
            root.display()
        )));
    }

    files.sort();
    let mut indexed = 0usize;
    let max_length = embedder
        .max_position_embeddings()
        .min(hypembed::EmbeddingOptions::default().max_length);
    let options = hypembed::EmbeddingOptions::default().with_max_length(max_length);

    for path in files {
        let content = std::fs::read_to_string(&path)?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string()
            .replace('\\', "/");

        let chunks = chunk_text(&rel, &content, DEFAULT_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS);

        if chunks.is_empty() {
            continue;
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder.embed(&texts, &options)?;

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            store.insert_chunk(&chunk.path, chunk.chunk_index, &chunk.text, embedding)?;
            indexed += 1;
        }
    }

    Ok(indexed)
}

fn is_text_doc(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("md") | Some("txt")
    )
}
