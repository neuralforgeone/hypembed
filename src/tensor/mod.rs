/// Core Tensor type with owned f32 storage.
///
/// The tensor uses row-major (C-order) contiguous storage.
/// All data is stored as a flat `Vec<f32>` with a `Shape` describing the layout.

pub mod shape;
pub mod ops;
pub mod matmul;
pub mod softmax;
pub mod activation;
pub mod layernorm;
pub mod normalize;
pub mod simd;

use crate::error::{HypEmbedError, Result};
pub use shape::Shape;

/// A dense tensor with owned f32 storage, row-major layout.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    // ── Constructors ──

    /// Create a tensor from a flat vector and shape.
    ///
    /// # Errors
    /// Returns an error if `data.len() != shape.numel()`.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(HypEmbedError::Tensor(format!(
                "Data length {} does not match shape {} (numel={})",
                data.len(),
                shape,
                shape.numel()
            )));
        }
        Ok(Self { data, shape })
    }

    /// Create a tensor of zeros.
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.numel();
        Self {
            data: vec![0.0; n],
            shape,
        }
    }

    /// Create a tensor of ones.
    pub fn ones(shape: Shape) -> Self {
        let n = shape.numel();
        Self {
            data: vec![1.0; n],
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Shape, value: f32) -> Self {
        let n = shape.numel();
        Self {
            data: vec![value; n],
            shape,
        }
    }

    // ── Accessors ──

    /// Get the shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the raw data as a slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get the raw data as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get a single element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let flat = self.shape.flat_index(indices)?;
        Ok(self.data[flat])
    }

    /// Set a single element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let flat = self.shape.flat_index(indices)?;
        self.data[flat] = value;
        Ok(())
    }

    // ── Shape manipulation ──

    /// Reshape the tensor to a new shape.
    ///
    /// The new shape must have the same total number of elements.
    pub fn reshape(&self, new_shape: Shape) -> Result<Tensor> {
        if self.shape.numel() != new_shape.numel() {
            return Err(HypEmbedError::Tensor(format!(
                "Cannot reshape from {} (numel={}) to {} (numel={})",
                self.shape,
                self.shape.numel(),
                new_shape,
                new_shape.numel()
            )));
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Transpose a 2D tensor.
    ///
    /// For [M, N] returns [N, M].
    pub fn transpose_2d(&self) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(HypEmbedError::Tensor(format!(
                "transpose_2d requires a 2D tensor, got shape {}",
                self.shape
            )));
        }
        let m = self.shape.dim(0)?;
        let n = self.shape.dim(1)?;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::from_vec(out, Shape::new(vec![n, m]))
    }

    /// Extract a slice along the first dimension.
    ///
    /// For a tensor of shape [A, B, C, ...], `slice_first(i)` returns
    /// a tensor of shape [B, C, ...] containing the i-th sub-tensor.
    pub fn slice_first(&self, index: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.is_empty() {
            return Err(HypEmbedError::Tensor("Cannot slice a scalar".into()));
        }
        let first_dim = dims[0];
        if index >= first_dim {
            return Err(HypEmbedError::Tensor(format!(
                "Slice index {} out of range for dim 0 (size {})",
                index, first_dim
            )));
        }
        let inner_size: usize = dims[1..].iter().product();
        let start = index * inner_size;
        let end = start + inner_size;
        let new_shape = Shape::new(dims[1..].to_vec());
        Tensor::from_vec(self.data[start..end].to_vec(), new_shape)
    }

    /// Stack multiple tensors along a new first dimension.
    ///
    /// All tensors must have the same shape. If each has shape [A, B, ...],
    /// the result has shape [N, A, B, ...] where N = tensors.len().
    pub fn stack(tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(HypEmbedError::Tensor("Cannot stack zero tensors".into()));
        }
        let expected_shape = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != expected_shape {
                return Err(HypEmbedError::Tensor(format!(
                    "Shape mismatch in stack: tensor 0 has shape {}, tensor {} has shape {}",
                    expected_shape,
                    i,
                    t.shape()
                )));
            }
        }
        let n = tensors.len();
        let inner = expected_shape.numel();
        let mut data = Vec::with_capacity(n * inner);
        for t in tensors {
            data.extend_from_slice(t.data());
        }
        let mut new_dims = vec![n];
        new_dims.extend_from_slice(expected_shape.dims());
        Tensor::from_vec(data, Shape::new(new_dims))
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3])).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_from_vec_mismatch() {
        let result = Tensor::from_vec(vec![1.0, 2.0], Shape::new(vec![2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::zeros(Shape::new(vec![3, 4]));
        assert_eq!(z.data().iter().sum::<f32>(), 0.0);
        let o = Tensor::ones(Shape::new(vec![2, 2]));
        assert_eq!(o.data().iter().sum::<f32>(), 4.0);
    }

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3])).unwrap();
        let tt = t.transpose_2d().unwrap();
        assert_eq!(tt.shape().dims(), &[3, 2]);
        assert_eq!(tt.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tt.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(tt.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(tt.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3])).unwrap();
        let r = t.reshape(Shape::new(vec![3, 2])).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(r.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_slice_first() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        ).unwrap();
        let s0 = t.slice_first(0).unwrap();
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(s0.data(), &[1.0, 2.0, 3.0]);
        let s1 = t.slice_first(1).unwrap();
        assert_eq!(s1.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3])).unwrap();
        let stacked = Tensor::stack(&[&a, &b]).unwrap();
        assert_eq!(stacked.shape().dims(), &[2, 3]);
        assert_eq!(stacked.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
