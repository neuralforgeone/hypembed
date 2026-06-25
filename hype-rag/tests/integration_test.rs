mod common {
    include!("../../tests/common/mod.rs");
}

use std::fs;

use hype_rag::{index_directory, search_chunks, ChunkStore};
use hypembed::Embedder;
use tempfile::TempDir;

#[test]
fn index_and_search_returns_ranked_hits() {
    let tmp = TempDir::new().unwrap();
    let model_dir = tmp.path().join("model");
    common::write_tiny_model_dir(&model_dir);

    let docs = tmp.path().join("docs");
    fs::create_dir_all(&docs).unwrap();
    fs::write(
        docs.join("rust.md"),
        "Rust is a systems programming language focused on safety and performance.",
    )
    .unwrap();
    fs::write(
        docs.join("ml.txt"),
        "Machine learning models generate embedding vectors for semantic search.",
    )
    .unwrap();
    fs::write(
        docs.join("cooking.txt"),
        "Bake the cake at 180 degrees for forty minutes.",
    )
    .unwrap();

    let data_dir = tmp.path().join(".hype-rag");
    let store = ChunkStore::open(&data_dir).unwrap();
    store.set_model_dir(&model_dir).unwrap();

    let embedder = Embedder::load(&model_dir).unwrap();
    let indexed = index_directory(&embedder, &store, &docs, true).unwrap();
    assert!(indexed >= 3);

    let chunks = store.all_chunks().unwrap();
    let hits = search_chunks(&embedder, "rust embedding semantic", &chunks, 3).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].score > 0.0,
        "expected positive similarity score, got {}",
        hits[0].score
    );
    assert!(!hits[0].path.is_empty());
}
