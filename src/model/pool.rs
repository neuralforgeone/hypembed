/// Pooling strategies for converting token-level hidden states to a single vector.
///
/// ## Mean Pooling (with mask)
///
/// ```text
/// pooled = sum(hidden_states * mask_expanded, dim=seq) / max(sum(mask), epsilon)
/// ```
///
/// This correctly ignores padding tokens by zeroing them out before summing,
/// and divides by the actual number of real tokens per sequence.
///
/// ## CLS Pooling
///
/// Simply takes the hidden state at position 0 (the `[CLS]` token).
use crate::error::Result;
use crate::tensor::{Shape, Tensor};

/// Pooling strategy for combining token embeddings into a sentence embedding.
///
/// # Examples
///
/// ```
/// use hypembed::PoolingStrategy;
///
/// let strategy = PoolingStrategy::Mean;
/// assert_eq!(strategy, PoolingStrategy::Mean);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pooling over non-padding tokens (recommended for sentence embeddings).
    Mean,
    /// Use the `[CLS]` token (position 0) embedding.
    Cls,
}

/// Apply pooling to hidden states.
///
/// # Arguments
/// - `hidden`: `[batch, seq, hidden_size]`
/// - `attention_mask`: `[batch, seq]` with `1 = real` and `0 = padding`
/// - `strategy`: Which pooling to use
///
/// # Returns
/// Pooled tensor `[batch, hidden_size]`
pub fn pool(
    hidden: &Tensor,
    attention_mask: &[Vec<u32>],
    strategy: PoolingStrategy,
) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Mean => mean_pool(hidden, attention_mask),
        PoolingStrategy::Cls => cls_pool(hidden),
    }
}

/// Mean pooling with attention mask.
///
/// Formula: sum(hidden * mask) / max(count_real_tokens, eps)
fn mean_pool(hidden: &Tensor, attention_mask: &[Vec<u32>]) -> Result<Tensor> {
    let batch_size = hidden.shape().dim(0)?;
    let seq_len = hidden.shape().dim(1)?;
    let hidden_size = hidden.shape().dim(2)?;
    let data = hidden.data();

    let eps = 1e-9f32;
    let mut pooled = vec![0.0f32; batch_size * hidden_size];

    for b in 0..batch_size {
        let mask = &attention_mask[b];

        // Count real tokens
        let token_count: f32 = mask.iter().map(|&m| m as f32).sum();
        let divisor = token_count.max(eps);

        // Sum hidden states for real tokens
        for s in 0..seq_len {
            if mask[s] == 1 {
                let src_offset = (b * seq_len + s) * hidden_size;
                let dst_offset = b * hidden_size;
                for h in 0..hidden_size {
                    pooled[dst_offset + h] += data[src_offset + h];
                }
            }
        }

        // Divide by token count
        let dst_offset = b * hidden_size;
        for h in 0..hidden_size {
            pooled[dst_offset + h] /= divisor;
        }
    }

    Tensor::from_vec(pooled, Shape::new(vec![batch_size, hidden_size]))
}

/// CLS pooling: take hidden state at position 0.
fn cls_pool(hidden: &Tensor) -> Result<Tensor> {
    let batch_size = hidden.shape().dim(0)?;
    let seq_len = hidden.shape().dim(1)?;
    let hidden_size = hidden.shape().dim(2)?;
    let data = hidden.data();

    let mut pooled = vec![0.0f32; batch_size * hidden_size];
    for b in 0..batch_size {
        let src_offset = b * seq_len * hidden_size; // position 0
        let dst_offset = b * hidden_size;
        pooled[dst_offset..dst_offset + hidden_size]
            .copy_from_slice(&data[src_offset..src_offset + hidden_size]);
    }

    Tensor::from_vec(pooled, Shape::new(vec![batch_size, hidden_size]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_no_padding() {
        // [1, 3, 2] hidden states, all real tokens
        let hidden = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![1, 3, 2]),
        )
        .unwrap();
        let mask = vec![vec![1u32, 1, 1]];
        let pooled = pool(&hidden, &mask, PoolingStrategy::Mean).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, 2]);
        // mean of [1,3,5] = 3.0, mean of [2,4,6] = 4.0
        assert!((pooled.data()[0] - 3.0).abs() < 1e-6);
        assert!((pooled.data()[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_with_padding() {
        // 3 tokens but last is padding
        let hidden = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0], // last token should be ignored
            Shape::new(vec![1, 3, 2]),
        )
        .unwrap();
        let mask = vec![vec![1u32, 1, 0]]; // only first 2 real
        let pooled = pool(&hidden, &mask, PoolingStrategy::Mean).unwrap();
        // mean of [1,3] = 2.0, mean of [2,4] = 3.0
        assert!((pooled.data()[0] - 2.0).abs() < 1e-6);
        assert!((pooled.data()[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cls_pool() {
        let hidden = Tensor::from_vec(
            vec![10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            Shape::new(vec![1, 3, 2]),
        )
        .unwrap();
        let mask = vec![vec![1u32, 1, 1]];
        let pooled = pool(&hidden, &mask, PoolingStrategy::Cls).unwrap();
        assert_eq!(pooled.data(), &[10.0, 20.0]); // position 0
    }

    #[test]
    fn test_mean_pool_batch() {
        let hidden = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // batch 0, seq 0, seq 1
                5.0, 6.0, 7.0, 8.0, // batch 1, seq 0, seq 1
            ],
            Shape::new(vec![2, 2, 2]),
        )
        .unwrap();
        let mask = vec![vec![1u32, 1], vec![1, 0]];
        let pooled = pool(&hidden, &mask, PoolingStrategy::Mean).unwrap();
        assert_eq!(pooled.shape().dims(), &[2, 2]);
        // batch 0: mean([1,3], [2,4]) = [2, 3]
        assert!((pooled.data()[0] - 2.0).abs() < 1e-6);
        assert!((pooled.data()[1] - 3.0).abs() < 1e-6);
        // batch 1: only first token [5, 6]
        assert!((pooled.data()[2] - 5.0).abs() < 1e-6);
        assert!((pooled.data()[3] - 6.0).abs() < 1e-6);
    }
}
