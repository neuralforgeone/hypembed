/// Matrix multiplication operations.
///
/// Implements 2D matmul and batched matmul needed for transformer inference.

use crate::error::{HypEmbedError, Result};
use crate::tensor::{Tensor, Shape};
use crate::tensor::simd;

/// 2D matrix multiplication: C = A @ B
///
/// A: [M, K], B: [K, N] → C: [M, N]
///
/// Uses cache-friendly ikj loop ordering to improve data locality:
/// for each row i of A, for each element k in the inner dimension,
/// we scatter A[i,k] across the output row by iterating j.
/// This accesses B row-by-row (contiguous in memory).
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 2 || b.rank() != 2 {
        return Err(HypEmbedError::Tensor(format!(
            "matmul requires 2D tensors, got shapes {} and {}",
            a.shape(), b.shape()
        )));
    }
    let m = a.shape().dim(0)?;
    let k_a = a.shape().dim(1)?;
    let k_b = b.shape().dim(0)?;
    let n = b.shape().dim(1)?;

    if k_a != k_b {
        return Err(HypEmbedError::Tensor(format!(
            "matmul inner dimension mismatch: A is [{}, {}], B is [{}, {}]",
            m, k_a, k_b, n
        )));
    }

    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0f32; m * n];

    // ikj loop order for cache-friendly access, SIMD-accelerated inner loop
    for i in 0..m {
        let a_row = i * k_a;
        let c_row = i * n;
        for k in 0..k_a {
            let a_ik = a_data[a_row + k];
            let b_row = k * n;
            simd::add_assign_scaled(
                &mut c_data[c_row..c_row + n],
                &b_data[b_row..b_row + n],
                a_ik,
            );
        }
    }

    Tensor::from_vec(c_data, Shape::new(vec![m, n]))
}

/// Batched matrix multiplication for attention computation.
///
/// A: [B, M, K], B: [B, K, N] → C: [B, M, N]
///
/// Applies independent matmul for each batch element.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 3 || b.rank() != 3 {
        return Err(HypEmbedError::Tensor(format!(
            "batched_matmul requires 3D tensors, got shapes {} and {}",
            a.shape(), b.shape()
        )));
    }
    let batch_a = a.shape().dim(0)?;
    let batch_b = b.shape().dim(0)?;
    if batch_a != batch_b {
        return Err(HypEmbedError::Tensor(format!(
            "Batch dimension mismatch: {} vs {}",
            batch_a, batch_b
        )));
    }
    let batch = batch_a;
    let m = a.shape().dim(1)?;
    let k_a = a.shape().dim(2)?;
    let k_b = b.shape().dim(1)?;
    let n = b.shape().dim(2)?;

    if k_a != k_b {
        return Err(HypEmbedError::Tensor(format!(
            "batched_matmul inner dimension mismatch: [{}, {}, {}] vs [{}, {}, {}]",
            batch, m, k_a, batch, k_b, n
        )));
    }

    let a_data = a.data();
    let b_data = b.data();
    let a_batch_stride = m * k_a;
    let b_batch_stride = k_b * n;
    let c_batch_stride = m * n;
    let mut c_data = vec![0.0f32; batch * m * n];

    for bi in 0..batch {
        let a_base = bi * a_batch_stride;
        let b_base = bi * b_batch_stride;
        let c_base = bi * c_batch_stride;

        for i in 0..m {
            let a_row = a_base + i * k_a;
            let c_row = c_base + i * n;
            for k in 0..k_a {
                let a_ik = a_data[a_row + k];
                let b_row = b_base + k * n;
                simd::add_assign_scaled(
                    &mut c_data[c_row..c_row + n],
                    &b_data[b_row..b_row + n],
                    a_ik,
                );
            }
        }
    }

    Tensor::from_vec(c_data, Shape::new(vec![batch, m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // Multiply by identity matrix
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2])).unwrap();
        let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2])).unwrap();
        let c = matmul(&a, &eye).unwrap();
        assert_eq!(c.data(), a.data());
    }

    #[test]
    fn test_matmul_basic() {
        // [1, 2] @ [5, 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3, 4]   [7, 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2])).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2])).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_rect() {
        // [2, 3] @ [3, 1]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3])).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3, 1])).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 1]);
        assert_eq!(c.data(), &[4.0, 10.0]);
    }

    #[test]
    fn test_matmul_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3])).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], Shape::new(vec![2, 1])).unwrap();
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_batched_matmul() {
        // batch=2, each 2x2 matmul
        let a = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 1.0, // batch 0: identity
                2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
            ],
            Shape::new(vec![2, 2, 2]),
        ).unwrap();
        let b = Tensor::from_vec(
            vec![
                3.0, 4.0, 5.0, 6.0, // batch 0
                3.0, 4.0, 5.0, 6.0, // batch 1
            ],
            Shape::new(vec![2, 2, 2]),
        ).unwrap();
        let c = batched_matmul(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2, 2]);
        // batch 0: identity * B = B
        assert_eq!(c.data()[0..4], [3.0, 4.0, 5.0, 6.0]);
        // batch 1: 2I * B = 2B
        assert_eq!(c.data()[4..8], [6.0, 8.0, 10.0, 12.0]);
    }
}
