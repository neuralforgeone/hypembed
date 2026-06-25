mod common;

use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};

#[test]
fn embed_from_bytes_produces_expected_dimension() {
    let weights = common::tiny_safetensors_bytes();
    let vocab = common::tiny_vocab_txt();
    let embedder = Embedder::from_bytes(common::TINY_CONFIG_JSON, &vocab, &weights, true)
        .expect("load tiny model from bytes");

    assert_eq!(embedder.hidden_size(), 8);

    let options = EmbeddingOptions::default()
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true)
        .with_max_length(16);

    let embeddings = embedder
        .embed(&["hello rust embedding"], &options)
        .expect("embed text");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 8);

    let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2-normalized vector, got norm {norm}"
    );
}

#[test]
fn embed_from_bytes_is_deterministic() {
    let weights = common::tiny_safetensors_bytes();
    let vocab = common::tiny_vocab_txt();
    let embedder = Embedder::from_bytes(common::TINY_CONFIG_JSON, &vocab, &weights, true).unwrap();

    let options = EmbeddingOptions::default().with_max_length(16);
    let a = embedder.embed(&["semantic search"], &options).unwrap();
    let b = embedder.embed(&["semantic search"], &options).unwrap();
    assert_eq!(a, b);
}
