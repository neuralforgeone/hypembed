/// Numerically stable softmax.
///
/// # Formula
///
/// ```text
/// softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
/// ```
///
/// Subtracting the maximum prevents overflow in `exp()` and avoids
/// underflow of the denominator. This is the standard stable softmax
/// approach used in all production ML systems.
use crate::error::{HypEmbedError, Result};
use crate::tensor::Tensor;

/// Apply softmax along the last dimension.
///
/// For a tensor of shape [..., N], applies softmax independently
/// to each row of length N.
pub fn softmax(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.is_empty() {
        return Err(HypEmbedError::Tensor(
            "Cannot apply softmax to scalar".into(),
        ));
    }
    let last_dim = *dims.last().unwrap();
    if last_dim == 0 {
        return Err(HypEmbedError::Tensor(
            "Cannot apply softmax to zero-length dimension".into(),
        ));
    }

    let data = tensor.data();
    let mut result = vec![0.0f32; data.len()];
    let num_rows = data.len() / last_dim;

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_data = &data[start..end];

        // Step 1: Find max for numerical stability
        let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Step 2: Compute exp(x_i - max) and sum
        let mut sum = 0.0f32;
        for i in 0..last_dim {
            let exp_val = (row_data[i] - max_val).exp();
            result[start + i] = exp_val;
            sum += exp_val;
        }

        // Step 3: Normalize
        // sum should be > 0 since at least one exp() value is exp(0) = 1
        let inv_sum = 1.0 / sum;
        for i in 0..last_dim {
            result[start + i] *= inv_sum;
        }
    }

    Tensor::from_vec(result, tensor.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_softmax_basic() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let s = softmax(&t).unwrap();
        let data = s.data();
        // Sum should be 1.0
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be monotonically increasing
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
        // Check known values
        let e1 = 1.0f32.exp();
        let e2 = 2.0f32.exp();
        let e3 = 3.0f32.exp();
        let total = e1 + e2 + e3;
        assert!((data[0] - e1 / total).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_stability_large() {
        // Large values that would overflow without max subtraction
        let t = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], Shape::new(vec![3])).unwrap();
        let s = softmax(&t).unwrap();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Should still be monotonically increasing
        assert!(s.data()[0] < s.data()[1]);
        assert!(s.data()[1] < s.data()[2]);
    }

    #[test]
    fn test_softmax_stability_negative() {
        let t = Tensor::from_vec(vec![-1000.0, -999.0, -998.0], Shape::new(vec![3])).unwrap();
        let s = softmax(&t).unwrap();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_uniform() {
        // Equal values should give uniform distribution
        let t = Tensor::from_vec(vec![5.0, 5.0, 5.0, 5.0], Shape::new(vec![4])).unwrap();
        let s = softmax(&t).unwrap();
        for &v in s.data() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_2d() {
        // Softmax along last dim for 2D tensor
        let t =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], Shape::new(vec![2, 3])).unwrap();
        let s = softmax(&t).unwrap();
        // Each row should sum to 1
        let sum_row0: f32 = s.data()[0..3].iter().sum();
        let sum_row1: f32 = s.data()[3..6].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6);
        assert!((sum_row1 - 1.0).abs() < 1e-6);
        // Both rows have same input so same output
        assert!((s.data()[0] - s.data()[3]).abs() < 1e-6);
    }
}
