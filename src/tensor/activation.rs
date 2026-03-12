/// GELU activation function.
///
/// # Formula
///
/// Uses the tanh approximation:
/// ```text
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// This matches the implementation used in BERT, GPT-2, and most HuggingFace models.
/// The approximation is accurate to ~1e-4 relative error.

use crate::tensor::Tensor;

/// Constant: sqrt(2 / π) ≈ 0.7978845608
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Apply GELU element-wise.
pub fn gelu(tensor: &Tensor) -> Tensor {
    let data: Vec<f32> = tensor.data().iter().map(|&x| {
        let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();
    // Shape unchanged, length unchanged; unwrap is safe
    Tensor::from_vec(data, tensor.shape().clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_gelu_zero() {
        let t = Tensor::from_vec(vec![0.0], Shape::new(vec![1])).unwrap();
        let g = gelu(&t);
        assert!((g.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let t = Tensor::from_vec(vec![1.0], Shape::new(vec![1])).unwrap();
        let g = gelu(&t);
        // GELU(1.0) ≈ 0.8412
        assert!((g.data()[0] - 0.8412).abs() < 0.001);
    }

    #[test]
    fn test_gelu_negative() {
        let t = Tensor::from_vec(vec![-1.0], Shape::new(vec![1])).unwrap();
        let g = gelu(&t);
        // GELU(-1.0) ≈ -0.1588
        assert!((g.data()[0] - (-0.1588)).abs() < 0.001);
    }

    #[test]
    fn test_gelu_large() {
        let t = Tensor::from_vec(vec![3.0], Shape::new(vec![1])).unwrap();
        let g = gelu(&t);
        // GELU(3.0) ≈ 2.9960
        assert!((g.data()[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_batch() {
        let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new(vec![5])).unwrap();
        let g = gelu(&t);
        let data = g.data();
        // GELU(0) ≈ 0
        assert!(data[2].abs() < 1e-6);
        // GELU is positive for positive inputs
        assert!(data[3] > 0.0);
        assert!(data[4] > 0.0);
        // GELU(x) < x for positive x
        assert!(data[3] < 1.0);
        assert!(data[4] < 2.0);
        // GELU(-2) > GELU(-1) is NOT guaranteed (GELU has a trough)
        // Instead verify GELU is negative for negative inputs
        assert!(data[0] < 0.0);
        assert!(data[1] < 0.0);
    }
}
