/// Tensor shape representation.
///
/// A `Shape` describes the dimensions of a tensor in row-major order.
/// For example, a shape `[2, 3, 4]` represents a 3D tensor where:
/// - dim 0 has size 2
/// - dim 1 has size 3
/// - dim 2 has size 4
///
/// Total element count = 2 * 3 * 4 = 24.
use crate::error::{HypEmbedError, Result};

/// A tensor shape with dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimension sizes.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Create a scalar shape (0 dimensions).
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Get the size of a specific dimension.
    pub fn dim(&self, index: usize) -> Result<usize> {
        self.dims.get(index).copied().ok_or_else(|| {
            HypEmbedError::Tensor(format!(
                "Dimension index {} out of range for shape {:?}",
                index, self.dims
            ))
        })
    }

    /// Get the dimension sizes as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Compute row-major strides.
    ///
    /// For shape `[2, 3, 4]`, strides are `[12, 4, 1]`.
    /// `stride[i] = product of dims[i+1..]`.
    pub fn strides(&self) -> Vec<usize> {
        let rank = self.dims.len();
        if rank == 0 {
            return vec![];
        }
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Validate that a flat index is within bounds.
    pub fn validate_flat_index(&self, index: usize) -> Result<()> {
        let n = self.numel();
        if index >= n {
            return Err(HypEmbedError::Tensor(format!(
                "Flat index {} out of range for shape {:?} (numel={})",
                index, self.dims, n
            )));
        }
        Ok(())
    }

    /// Convert multi-dimensional indices to a flat index.
    pub fn flat_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.dims.len() {
            return Err(HypEmbedError::Tensor(format!(
                "Expected {} indices, got {} for shape {:?}",
                self.dims.len(),
                indices.len(),
                self.dims
            )));
        }
        let strides = self.strides();
        let mut flat = 0usize;
        for (i, (&idx, &dim)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if idx >= dim {
                return Err(HypEmbedError::Tensor(format!(
                    "Index {} out of range for dimension {} (size {})",
                    idx, i, dim
                )));
            }
            flat += idx * strides[i];
        }
        Ok(flat)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(0).unwrap(), 2);
        assert_eq!(s.dim(1).unwrap(), 3);
        assert_eq!(s.dim(2).unwrap(), 4);
        assert!(s.dim(3).is_err());
    }

    #[test]
    fn test_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_flat_index() {
        let s = Shape::new(vec![2, 3]);
        assert_eq!(s.flat_index(&[0, 0]).unwrap(), 0);
        assert_eq!(s.flat_index(&[0, 2]).unwrap(), 2);
        assert_eq!(s.flat_index(&[1, 0]).unwrap(), 3);
        assert_eq!(s.flat_index(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.strides(), Vec::<usize>::new());
    }
}
