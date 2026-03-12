//! Basic example: load a model and generate embeddings.
//!
//! To run this example, download a BERT-like model (e.g., all-MiniLM-L6-v2)
//! and place config.json, vocab.txt, and model.safetensors in a directory.
//!
//! ```bash
//! cargo run --example basic_embed -- ./path/to/model
//! ```

use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: basic_embed <model_directory>");
        eprintln!("  The directory must contain: config.json, vocab.txt, model.safetensors");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    println!("Loading model from '{}'...", model_dir);

    let model = match Embedder::load(model_dir) {
        Ok(m) => {
            println!("Model loaded successfully (hidden_size={})", m.hidden_size());
            m
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let options = EmbeddingOptions::default()
        .with_max_length(128)
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true);

    let texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above a sleepy hound.",
        "Quantum computing will revolutionize cryptography.",
    ];

    println!("\nGenerating embeddings for {} texts...\n", texts.len());
    let embeddings = match model.embed(&texts, &options) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Embedding failed: {}", e);
            std::process::exit(1);
        }
    };

    for (i, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
        println!("Text {}: \"{}\"", i, text);
        println!("  Embedding dim: {}", emb.len());
        println!("  First 5 values: {:?}", &emb[..5.min(emb.len())]);

        // Verify L2 norm ≈ 1.0
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  L2 norm: {:.6}", norm);
        println!();
    }

    // Compute cosine similarity between first two texts
    if embeddings.len() >= 2 {
        let cos_sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        let cos_sim_diff = cosine_similarity(&embeddings[0], &embeddings[2]);
        println!("Cosine similarity (similar sentences): {:.6}", cos_sim);
        println!("Cosine similarity (different topics):  {:.6}", cos_sim_diff);
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
