/// Multi-head self-attention.
///
/// Implements scaled dot-product attention with multiple heads:
///
/// ```text
/// Q = x @ Wq + bq
/// K = x @ Wk + bk
/// V = x @ Wv + bv
///
/// // Split into heads: [batch, seq, hidden] → [batch * heads, seq, head_dim]
/// scores = Q @ K^T / sqrt(d_k)
/// scores = scores + attention_mask   (mask: 0 for real, -10000 for padding)
/// weights = softmax(scores)
/// context = weights @ V
///
/// // Concatenate heads: [batch * heads, seq, head_dim] → [batch, seq, hidden]
/// output = context @ Wo + bo
/// ```
use crate::error::Result;
use crate::model::weights::AttentionWeights;
use crate::tensor::matmul;
use crate::tensor::ops;
use crate::tensor::softmax;
use crate::tensor::{Shape, Tensor};

/// Attention mask value for padding positions.
/// Added to scores before softmax to effectively zero out padding attention.
const MASK_VALUE: f32 = -10000.0;

/// Run multi-head self-attention.
///
/// # Arguments
/// - `hidden`: Input tensor [batch, seq, hidden_size]
/// - `attention_mask`: [batch, seq] with 1 for real tokens, 0 for padding
/// - `weights`: Attention layer weights
/// - `num_heads`: Number of attention heads
///
/// # Returns
/// Output tensor [batch, seq, hidden_size]
pub fn multi_head_attention(
    hidden: &Tensor,
    attention_mask: &[Vec<u32>],
    weights: &AttentionWeights,
    num_heads: usize,
) -> Result<Tensor> {
    let batch_size = hidden.shape().dim(0)?;
    let seq_len = hidden.shape().dim(1)?;
    let hidden_size = hidden.shape().dim(2)?;
    let head_dim = hidden_size / num_heads;

    // Reshape hidden to [batch * seq, hidden] for matmul with weight matrices
    let hidden_2d = hidden.reshape(Shape::new(vec![batch_size * seq_len, hidden_size]))?;

    // Q, K, V linear projections
    // Weight matrices are [hidden, hidden] stored as [out_features, in_features]
    // so we transpose them for x @ W^T
    let q_wt = weights.query_weight.transpose_2d()?;
    let k_wt = weights.key_weight.transpose_2d()?;
    let v_wt = weights.value_weight.transpose_2d()?;

    let q = ops::add_bias(&matmul::matmul(&hidden_2d, &q_wt)?, &weights.query_bias)?;
    let k = ops::add_bias(&matmul::matmul(&hidden_2d, &k_wt)?, &weights.key_bias)?;
    let v = ops::add_bias(&matmul::matmul(&hidden_2d, &v_wt)?, &weights.value_bias)?;

    // Reshape to [batch, seq, num_heads, head_dim]
    let q = q.reshape(Shape::new(vec![batch_size, seq_len, num_heads, head_dim]))?;
    let k = k.reshape(Shape::new(vec![batch_size, seq_len, num_heads, head_dim]))?;
    let v = v.reshape(Shape::new(vec![batch_size, seq_len, num_heads, head_dim]))?;

    // Transpose to [batch, num_heads, seq, head_dim] via manual rearrangement
    let q = transpose_1_2(&q, batch_size, seq_len, num_heads, head_dim)?;
    let k = transpose_1_2(&k, batch_size, seq_len, num_heads, head_dim)?;
    let v = transpose_1_2(&v, batch_size, seq_len, num_heads, head_dim)?;

    // Reshape to [batch * num_heads, seq, head_dim] for batched matmul
    let q = q.reshape(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]))?;
    let k = k.reshape(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]))?;
    let v = v.reshape(Shape::new(vec![batch_size * num_heads, seq_len, head_dim]))?;

    // K^T: [batch*heads, head_dim, seq]
    let k_t = batched_transpose_last_two(&k, batch_size * num_heads, seq_len, head_dim)?;

    // Scores = Q @ K^T / sqrt(d_k)
    let scores = matmul::batched_matmul(&q, &k_t)?; // [batch*heads, seq, seq]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = ops::scalar_mul(&scores, scale);

    // Apply attention mask
    let scores = apply_attention_mask(&scores, attention_mask, batch_size, num_heads, seq_len)?;

    // Softmax
    let weights_tensor = softmax::softmax(&scores)?;

    // Context = weights @ V
    let context = matmul::batched_matmul(&weights_tensor, &v)?; // [batch*heads, seq, head_dim]

    // Reshape back: [batch, num_heads, seq, head_dim] → [batch, seq, hidden]
    let context = context.reshape(Shape::new(vec![batch_size, num_heads, seq_len, head_dim]))?;
    let context = transpose_1_2(&context, batch_size, num_heads, seq_len, head_dim)?;
    let context = context.reshape(Shape::new(vec![batch_size * seq_len, hidden_size]))?;

    // Output projection
    let out_wt = weights.output_weight.transpose_2d()?;
    let output = ops::add_bias(&matmul::matmul(&context, &out_wt)?, &weights.output_bias)?;

    // Reshape back to [batch, seq, hidden]
    output.reshape(Shape::new(vec![batch_size, seq_len, hidden_size]))
}

/// Transpose dimensions 1 and 2 of a 4D tensor.
///
/// [batch, A, B, C] → [batch, B, A, C]
fn transpose_1_2(
    tensor: &Tensor,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
) -> Result<Tensor> {
    let data = tensor.data();
    let mut out = vec![0.0f32; data.len()];

    for b in 0..dim0 {
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    let src_idx = b * dim1 * dim2 * dim3 + i * dim2 * dim3 + j * dim3 + k;
                    let dst_idx = b * dim2 * dim1 * dim3 + j * dim1 * dim3 + i * dim3 + k;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }

    Tensor::from_vec(out, Shape::new(vec![dim0, dim2, dim1, dim3]))
}

/// Transpose last two dimensions of a 3D tensor.
///
/// [batch, M, N] → [batch, N, M]
fn batched_transpose_last_two(tensor: &Tensor, batch: usize, m: usize, n: usize) -> Result<Tensor> {
    let data = tensor.data();
    let mut out = vec![0.0f32; data.len()];

    for b in 0..batch {
        let base = b * m * n;
        for i in 0..m {
            for j in 0..n {
                out[base + j * m + i] = data[base + i * n + j];
            }
        }
    }

    Tensor::from_vec(out, Shape::new(vec![batch, n, m]))
}

/// Apply attention mask to scores.
///
/// For each position where `attention_mask[b][s] == 0` (padding),
/// add MASK_VALUE (-10000) to the corresponding score column,
/// ensuring softmax drives those weights to ~0.
fn apply_attention_mask(
    scores: &Tensor,
    attention_mask: &[Vec<u32>],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let mut data = scores.data().to_vec();

    for b in 0..batch_size {
        for h in 0..num_heads {
            let head_idx = b * num_heads + h;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if attention_mask[b][j] == 0 {
                        let idx = head_idx * seq_len * seq_len + i * seq_len + j;
                        data[idx] += MASK_VALUE;
                    }
                }
            }
        }
    }

    Tensor::from_vec(data, scores.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_1_2() {
        // [1, 2, 3, 1] → [1, 3, 2, 1]
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![1, 2, 3, 1]),
        )
        .unwrap();
        let tt = transpose_1_2(&t, 1, 2, 3, 1).unwrap();
        assert_eq!(tt.shape().dims(), &[1, 3, 2, 1]);
        // Original: [[1,2,3],[4,5,6]] → transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_batched_transpose() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![1, 2, 3]),
        )
        .unwrap();
        let tt = batched_transpose_last_two(&t, 1, 2, 3).unwrap();
        assert_eq!(tt.shape().dims(), &[1, 3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
