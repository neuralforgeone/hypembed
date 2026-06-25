/// Token, position, and segment embeddings.
///
/// Implements the BERT embedding layer:
/// ```text
/// output = LayerNorm(word_embed + position_embed + token_type_embed)
/// ```
use crate::error::{HypEmbedError, Result};
use crate::tensor::layernorm;
use crate::tensor::{Shape, Tensor};

/// Compute embeddings for a batch of token sequences.
///
/// # Arguments
/// - `input_ids`: `[batch_size, seq_len]` integer token IDs
/// - `token_type_ids`: `[batch_size, seq_len]` segment IDs (0 or 1)
/// - `word_embeddings`: `[vocab_size, hidden_size]`
/// - `position_embeddings`: `[max_positions, hidden_size]`
/// - `token_type_embeddings`: `[type_vocab_size, hidden_size]` or `None` for models without them
/// - `ln_weight`, `ln_bias`: LayerNorm parameters `[hidden_size]`
/// - `eps`: LayerNorm epsilon
///
/// # Returns
/// Tensor of shape `[batch_size, seq_len, hidden_size]`
pub fn compute_embeddings(
    input_ids: &[Vec<u32>],
    token_type_ids: &[Vec<u32>],
    word_embeddings: &Tensor,
    position_embeddings: &Tensor,
    token_type_embeddings: Option<&Tensor>,
    ln_weight: &Tensor,
    ln_bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if input_ids.is_empty() {
        return Err(HypEmbedError::invalid_input("input_ids batch is empty"));
    }

    let batch_size = input_ids.len();
    let seq_len = input_ids[0].len();
    if seq_len == 0 {
        return Err(HypEmbedError::invalid_input(
            "input_ids sequence length is zero",
        ));
    }

    let max_positions = position_embeddings.shape().dim(0)?;
    if seq_len > max_positions {
        return Err(HypEmbedError::invalid_input(format!(
            "sequence length {seq_len} exceeds position embeddings ({max_positions})"
        )));
    }

    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            return Err(HypEmbedError::invalid_input(format!(
                "input_ids row {i} length {} != expected {seq_len}",
                row.len()
            )));
        }
    }
    if token_type_ids.len() != batch_size {
        return Err(HypEmbedError::invalid_input(format!(
            "token_type_ids batch size {} != input_ids batch size {batch_size}",
            token_type_ids.len()
        )));
    }
    for (i, row) in token_type_ids.iter().enumerate() {
        if row.len() != seq_len {
            return Err(HypEmbedError::invalid_input(format!(
                "token_type_ids row {i} length {} != expected {seq_len}",
                row.len()
            )));
        }
    }

    let hidden_size = word_embeddings.shape().dim(1)?;

    let mut output = vec![0.0f32; batch_size * seq_len * hidden_size];
    let word_data = word_embeddings.data();
    let pos_data = position_embeddings.data();
    let tt_data = token_type_embeddings.map(|t| t.data());

    for b in 0..batch_size {
        for s in 0..seq_len {
            let token_id = input_ids[b][s] as usize;
            let out_offset = (b * seq_len + s) * hidden_size;

            // word_embed[token_id] + position_embed[s]
            let word_offset = token_id * hidden_size;
            let pos_offset = s * hidden_size;

            for h in 0..hidden_size {
                output[out_offset + h] = word_data[word_offset + h] + pos_data[pos_offset + h];
            }

            // + token_type_embed[type_id] (if available)
            if let Some(tt) = tt_data {
                let type_id = token_type_ids[b][s] as usize;
                let tt_offset = type_id * hidden_size;
                for h in 0..hidden_size {
                    output[out_offset + h] += tt[tt_offset + h];
                }
            }
        }
    }

    let embed_tensor =
        Tensor::from_vec(output, Shape::new(vec![batch_size, seq_len, hidden_size]))?;

    // Apply LayerNorm
    layernorm::layer_norm(&embed_tensor, ln_weight, ln_bias, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let vocab_size = 10;
        let hidden_size = 4;
        let max_pos = 8;
        let type_vocab = 2;

        let word_emb = Tensor::ones(Shape::new(vec![vocab_size, hidden_size]));
        let pos_emb = Tensor::ones(Shape::new(vec![max_pos, hidden_size]));
        let tt_emb = Tensor::zeros(Shape::new(vec![type_vocab, hidden_size]));
        let ln_w = Tensor::ones(Shape::new(vec![hidden_size]));
        let ln_b = Tensor::zeros(Shape::new(vec![hidden_size]));

        let input_ids = vec![vec![0u32, 1, 2], vec![3, 4, 5]];
        let token_type_ids = vec![vec![0u32, 0, 0], vec![0, 0, 0]];

        let result = compute_embeddings(
            &input_ids,
            &token_type_ids,
            &word_emb,
            &pos_emb,
            Some(&tt_emb),
            &ln_w,
            &ln_b,
            1e-12,
        )
        .unwrap();

        assert_eq!(result.shape().dims(), &[2, 3, 4]); // [batch, seq, hidden]
    }

    #[test]
    fn compute_embeddings_rejects_empty_batch() {
        let word_emb = Tensor::ones(Shape::new(vec![10, 4]));
        let pos_emb = Tensor::ones(Shape::new(vec![8, 4]));
        let ln_w = Tensor::ones(Shape::new(vec![4]));
        let ln_b = Tensor::zeros(Shape::new(vec![4]));

        let err = compute_embeddings(&[], &[], &word_emb, &pos_emb, None, &ln_w, &ln_b, 1e-12)
            .unwrap_err();
        assert!(matches!(err, HypEmbedError::InvalidInput { .. }));
    }

    #[test]
    fn compute_embeddings_rejects_seq_len_beyond_positions() {
        let word_emb = Tensor::ones(Shape::new(vec![10, 4]));
        let pos_emb = Tensor::ones(Shape::new(vec![2, 4]));
        let ln_w = Tensor::ones(Shape::new(vec![4]));
        let ln_b = Tensor::zeros(Shape::new(vec![4]));
        let input_ids = vec![vec![0u32, 1, 2]];
        let token_type_ids = vec![vec![0u32, 0, 0]];

        let err = compute_embeddings(
            &input_ids,
            &token_type_ids,
            &word_emb,
            &pos_emb,
            None,
            &ln_w,
            &ln_b,
            1e-12,
        )
        .unwrap_err();
        assert!(matches!(err, HypEmbedError::InvalidInput { .. }));
    }
}
