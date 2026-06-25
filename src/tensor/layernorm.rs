/// Layer normalization.
///
/// # Formula
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// Where:
/// - mean and var are computed over the last dimension (per-token)
/// - eps is a small constant (typically 1e-12) for numerical stability
/// - `gamma` (scale) and `beta` (bias) are learnable parameters of shape `[hidden_size]`
///
/// This is the standard pre-/post-layer normalization used in transformers.
use crate::error::{HypEmbedError, Result};
use crate::tensor::Tensor;

/// Apply layer normalization along the last dimension.
///
/// # Arguments
/// - `tensor`: Input tensor of shape `[..., hidden_size]`
/// - `gamma`: Scale parameter of shape `[hidden_size]`
/// - `beta`: Bias parameter of shape `[hidden_size]`
/// - `eps`: Small constant for numerical stability (typically 1e-12)
pub fn layer_norm(tensor: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.is_empty() {
        return Err(HypEmbedError::Tensor(
            "Cannot apply layer_norm to scalar".into(),
        ));
    }
    let hidden_size = *dims.last().unwrap();

    if gamma.rank() != 1 || gamma.shape().dim(0)? != hidden_size {
        return Err(HypEmbedError::Tensor(format!(
            "gamma shape mismatch: expected [{}], got {}",
            hidden_size,
            gamma.shape()
        )));
    }
    if beta.rank() != 1 || beta.shape().dim(0)? != hidden_size {
        return Err(HypEmbedError::Tensor(format!(
            "beta shape mismatch: expected [{}], got {}",
            hidden_size,
            beta.shape()
        )));
    }

    let data = tensor.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();
    let num_rows = data.len() / hidden_size;
    let mut result = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let row_data = &data[start..end];

        // Compute mean
        let mean: f32 = row_data.iter().sum::<f32>() / hidden_size as f32;

        // Compute variance: E[(x - mean)²]
        // Use two-pass for better numerical stability
        let var: f32 = row_data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / hidden_size as f32;

        // Normalization factor: 1 / sqrt(var + eps)
        let inv_std = 1.0 / (var + eps).sqrt();

        // Apply: gamma * (x - mean) * inv_std + beta
        for i in 0..hidden_size {
            result[start + i] = gamma_data[i] * (row_data[i] - mean) * inv_std + beta_data[i];
        }
    }

    Tensor::from_vec(result, tensor.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_layer_norm_basic() {
        // Simple case: normalize [1, 2, 3] with gamma=1, beta=0
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let gamma = Tensor::ones(Shape::new(vec![3]));
        let beta = Tensor::zeros(Shape::new(vec![3]));
        let result = layer_norm(&t, &gamma, &beta, 1e-12).unwrap();
        let data = result.data();

        // Mean should be ~0 after normalization
        let mean: f32 = data.iter().sum::<f32>() / 3.0;
        assert!(
            mean.abs() < 1e-5,
            "Mean after LN should be ~0, got {}",
            mean
        );

        // Variance should be ~1
        let var: f32 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 3.0;
        assert!(
            (var - 1.0).abs() < 1e-4,
            "Var after LN should be ~1, got {}",
            var
        );
    }

    #[test]
    fn test_layer_norm_with_params() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let gamma = Tensor::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3])).unwrap();
        let beta = Tensor::from_vec(vec![1.0, 1.0, 1.0], Shape::new(vec![3])).unwrap();
        let result = layer_norm(&t, &gamma, &beta, 1e-12).unwrap();
        let data = result.data();

        // With gamma=2, beta=1: result = 2 * normalized + 1
        // Mean of result should be 1.0 (since mean of normalized is 0)
        let mean: f32 = data.iter().sum::<f32>() / 3.0;
        assert!((mean - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_2d() {
        // 2D: each row normalized independently
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            Shape::new(vec![2, 3]),
        )
        .unwrap();
        let gamma = Tensor::ones(Shape::new(vec![3]));
        let beta = Tensor::zeros(Shape::new(vec![3]));
        let result = layer_norm(&t, &gamma, &beta, 1e-12).unwrap();
        let data = result.data();

        // Row 0 and row 1 should have same normalized pattern
        // (since they're scalar multiples of each other, they normalize the same)
        for i in 0..3 {
            assert!(
                (data[i] - data[3 + i]).abs() < 1e-4,
                "Row 0 and row 1 should normalize identically"
            );
        }
    }

    #[test]
    fn test_layer_norm_constant_input() {
        // All same values → variance is 0 → eps prevents division by zero
        let t = Tensor::from_vec(vec![5.0, 5.0, 5.0], Shape::new(vec![3])).unwrap();
        let gamma = Tensor::ones(Shape::new(vec![3]));
        let beta = Tensor::zeros(Shape::new(vec![3]));
        let result = layer_norm(&t, &gamma, &beta, 1e-12).unwrap();
        let data = result.data();
        // (x - mean) = 0 for all, so result should be 0 + beta = 0
        for &v in data {
            assert!(v.abs() < 1e-5);
        }
    }
}
