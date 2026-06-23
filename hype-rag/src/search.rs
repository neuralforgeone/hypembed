use hypembed::{Embedder, EmbeddingOptions};

use crate::error::Result;
use crate::store::StoredChunk;

/// A ranked search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub path: String,
    pub chunk_index: usize,
    pub text: String,
    pub score: f32,
}

/// Cosine similarity between two equal-length vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}

/// Embed a query and rank stored chunks by cosine similarity.
pub fn search_chunks(
    embedder: &Embedder,
    query: &str,
    chunks: &[StoredChunk],
    top_k: usize,
) -> Result<Vec<SearchHit>> {
    let max_length = embedder
        .config()
        .max_position_embeddings
        .min(EmbeddingOptions::default().max_length);
    let options = EmbeddingOptions::default().with_max_length(max_length);
    let query_emb = embedder.embed(&[query], &options)?[0].clone();

    let mut hits: Vec<SearchHit> = chunks
        .iter()
        .filter(|c| c.embedding.len() == query_emb.len())
        .map(|c| SearchHit {
            path: c.path.clone(),
            chunk_index: c.chunk_index,
            text: c.text.clone(),
            score: cosine_similarity(&query_emb, &c.embedding),
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_score_one() {
        let v = vec![0.6, 0.8];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors_score_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }
}