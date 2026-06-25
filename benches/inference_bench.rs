use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};

mod fixtures {
    include!("../tests/common/mod.rs");
}

fn tiny_embedder() -> Embedder {
    use fixtures::*;
    Embedder::from_bytes(
        TINY_CONFIG_JSON,
        &tiny_vocab_txt(),
        &tiny_safetensors_bytes(),
        true,
    )
    .expect("tiny embedder for benchmarks")
}

fn embed_single_bench(c: &mut Criterion) {
    let embedder = tiny_embedder();
    let options = EmbeddingOptions::default()
        .with_max_length(16)
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true);
    let text = "hello world rust embedding inference benchmark";

    c.bench_function("embed_single_short", |bench| {
        bench.iter(|| {
            black_box(embedder.embed(&[text], &options).expect("embed"));
        });
    });
}

fn embed_batch_bench(c: &mut Criterion) {
    let embedder = tiny_embedder();
    let options = EmbeddingOptions::default().with_max_length(16);
    let texts = [
        "semantic search with embeddings",
        "machine learning on device",
        "rust systems programming",
        "transformer encoder forward pass",
    ];

    c.bench_function("embed_batch_4", |bench| {
        bench.iter(|| {
            black_box(embedder.embed(&texts, &options).expect("embed batch"));
        });
    });
}

criterion_group!(benches, embed_single_bench, embed_batch_bench);
criterion_main!(benches);
