/// L2 normalization.
///
/// # Formula
///
/// ```text
/// x_normalized = x / max(||x||₂, epsilon)
/// ```
///
/// Where `||x||₂ = sqrt(sum(x_i²))`.
///
/// Using `max(norm, epsilon)` prevents division by zero for zero vectors.
/// The output vectors have unit L2 norm (or near-unit for near-zero inputs).
use crate::error::{HypEmbedError, Result};
use crate::tensor::Tensor;

/// L2 normalize along the last dimension.
///
/// For a tensor of shape [B, N], each row of length N is independently
/// normalized to unit length.
///
/// Epsilon default: 1e-12
pub fn l2_normalize(tensor: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.is_empty() {
        return Err(HypEmbedError::Tensor("Cannot L2-normalize a scalar".into()));
    }
    let last_dim = *dims.last().unwrap();
    if last_dim == 0 {
        return Err(HypEmbedError::Tensor(
            "Cannot L2-normalize a zero-length dimension".into(),
        ));
    }

    let data = tensor.data();
    let mut result = vec![0.0f32; data.len()];
    let num_rows = data.len() / last_dim;

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_data = &data[start..end];

        // Compute L2 norm
        let sq_sum: f32 = row_data.iter().map(|&x| x * x).sum();
        let norm = sq_sum.sqrt().max(eps);
        let inv_norm = 1.0 / norm;

        for i in 0..last_dim {
            result[start + i] = row_data[i] * inv_norm;
        }
    }

    Tensor::from_vec(result, tensor.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_l2_normalize_basic() {
        let t = Tensor::from_vec(vec![3.0, 4.0], Shape::new(vec![2])).unwrap();
        let n = l2_normalize(&t, 1e-12).unwrap();
        let data = n.data();
        // 3/5, 4/5
        assert!((data[0] - 0.6).abs() < 1e-6);
        assert!((data[1] - 0.8).abs() < 1e-6);
        // L2 norm should be 1.0
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let t = Tensor::zeros(Shape::new(vec![3]));
        let n = l2_normalize(&t, 1e-12).unwrap();
        // Should not panic or produce NaN
        for &v in n.data() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_l2_normalize_2d() {
        let t = Tensor::from_vec(vec![3.0, 4.0, 0.0, 5.0], Shape::new(vec![2, 2])).unwrap();
        let n = l2_normalize(&t, 1e-12).unwrap();
        let data = n.data();
        // Row 0: [3/5, 4/5]
        assert!((data[0] - 0.6).abs() < 1e-6);
        assert!((data[1] - 0.8).abs() < 1e-6);
        // Row 1: [0/5, 5/5] = [0, 1]
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_already_unit() {
        let t = Tensor::from_vec(vec![0.6, 0.8], Shape::new(vec![2])).unwrap();
        let n = l2_normalize(&t, 1e-12).unwrap();
        let data = n.data();
        assert!((data[0] - 0.6).abs() < 1e-5);
        assert!((data[1] - 0.8).abs() < 1e-5);
    }
}
