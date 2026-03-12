/// Tensor arithmetic operations.
///
/// All operations are dimension-checked and return new tensors (functional style).

use crate::error::{HypEmbedError, Result};
use crate::tensor::{Tensor, Shape};
use crate::tensor::simd;

/// Element-wise addition of two tensors with the same shape.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(HypEmbedError::Tensor(format!(
            "Shape mismatch in add: {} vs {}",
            a.shape(), b.shape()
        )));
    }
    let mut data = vec![0.0f32; a.numel()];
    simd::elementwise_add(a.data(), b.data(), &mut data);
    Tensor::from_vec(data, a.shape().clone())
}

/// Add a 1D bias to the last dimension of a tensor.
///
/// For tensor of shape [A, B, ..., N] and bias of shape [N],
/// adds bias to each row of the last dimension.
pub fn add_bias(tensor: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if bias.rank() != 1 {
        return Err(HypEmbedError::Tensor(format!(
            "Bias must be 1D, got shape {}",
            bias.shape()
        )));
    }
    let t_dims = tensor.shape().dims();
    let bias_len = bias.shape().dim(0)?;
    let last_dim = *t_dims.last().ok_or_else(|| {
        HypEmbedError::Tensor("Cannot add bias to scalar tensor".into())
    })?;
    if last_dim != bias_len {
        return Err(HypEmbedError::Tensor(format!(
            "Bias length {} does not match last dim {} of tensor shape {}",
            bias_len, last_dim, tensor.shape()
        )));
    }
    let bias_data = bias.data();
    let mut data = tensor.data().to_vec();
    for chunk in data.chunks_mut(last_dim) {
        for (val, &b) in chunk.iter_mut().zip(bias_data) {
            *val += b;
        }
    }
    Tensor::from_vec(data, tensor.shape().clone())
}

/// Element-wise multiplication (Hadamard product).
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(HypEmbedError::Tensor(format!(
            "Shape mismatch in mul: {} vs {}",
            a.shape(), b.shape()
        )));
    }
    let mut data = vec![0.0f32; a.numel()];
    simd::elementwise_mul(a.data(), b.data(), &mut data);
    Tensor::from_vec(data, a.shape().clone())
}

/// Multiply every element by a scalar.
pub fn scalar_mul(tensor: &Tensor, scalar: f32) -> Tensor {
    let mut data = vec![0.0f32; tensor.numel()];
    simd::scalar_mul_slice(tensor.data(), scalar, &mut data);
    // Safe: same shape, same length
    Tensor::from_vec(data, tensor.shape().clone()).unwrap()
}

/// Add a scalar to every element.
pub fn scalar_add(tensor: &Tensor, scalar: f32) -> Tensor {
    let data: Vec<f32> = tensor.data().iter().map(|&x| x + scalar).collect();
    Tensor::from_vec(data, tensor.shape().clone()).unwrap()
}

/// Element-wise addition with broadcasting on the last dimensions.
///
/// Supports broadcasting a tensor of shape [1, 1, N] or [N] onto [B, S, N].
/// The `mask` tensor is broadcast-expanded along dimensions of size 1.
pub fn add_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Simple broadcast: b has fewer dims or has 1s that get expanded
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    // Pad b_dims with leading 1s to match a's rank
    let rank = a_dims.len();
    let mut b_expanded = vec![1usize; rank];
    if b_dims.len() > rank {
        return Err(HypEmbedError::Tensor(format!(
            "Cannot broadcast {} onto {}",
            b.shape(), a.shape()
        )));
    }
    let offset = rank - b_dims.len();
    for (i, &d) in b_dims.iter().enumerate() {
        b_expanded[offset + i] = d;
    }

    // Validate broadcast compatibility
    for (i, (&ad, &bd)) in a_dims.iter().zip(b_expanded.iter()).enumerate() {
        if ad != bd && bd != 1 && ad != 1 {
            return Err(HypEmbedError::Tensor(format!(
                "Incompatible broadcast at dim {}: {} vs {}",
                i, ad, bd
            )));
        }
    }

    let a_strides = a.shape().strides();
    let b_shape_for_strides = Shape::new(b_expanded.clone());
    let b_strides = b_shape_for_strides.strides();

    let numel = a.shape().numel();
    let mut data = vec![0.0f32; numel];
    let a_data = a.data();
    let b_data = b.data();

    for flat_i in 0..numel {
        // Compute multi-dim index from flat index using a_dims
        let mut remaining = flat_i;
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for dim_idx in 0..rank {
            let idx = remaining / a_strides[dim_idx];
            remaining %= a_strides[dim_idx];
            a_offset += idx * a_strides[dim_idx];
            // For broadcast: if b_expanded[dim_idx] == 1, clamp index to 0
            let b_idx = if b_expanded[dim_idx] == 1 { 0 } else { idx };
            b_offset += b_idx * b_strides[dim_idx];
        }
        data[flat_i] = a_data[a_offset] + b_data[b_offset];
    }

    Tensor::from_vec(data, a.shape().clone())
}

/// Multiply a tensor [B, S, H] element-wise with tensor [B, S, 1],
/// broadcasting the last dimension.
///
/// This is used for applying attention masks in mean pooling.
pub fn mul_broadcast_last(tensor: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let t_dims = tensor.shape().dims();
    let m_dims = mask.shape().dims();

    if t_dims.len() != m_dims.len() {
        return Err(HypEmbedError::Tensor(format!(
            "Rank mismatch in mul_broadcast_last: {} vs {}",
            tensor.shape(), mask.shape()
        )));
    }

    // Must match on all dims except last, where mask can be 1
    for i in 0..t_dims.len() - 1 {
        if t_dims[i] != m_dims[i] {
            return Err(HypEmbedError::Tensor(format!(
                "Dimension mismatch at dim {}: {} vs {}",
                i, t_dims[i], m_dims[i]
            )));
        }
    }

    let last = *t_dims.last().unwrap();
    let m_last = *m_dims.last().unwrap();

    let mut data = tensor.data().to_vec();
    let mask_data = mask.data();

    if m_last == 1 {
        // Broadcast
        let rows = data.len() / last;
        for row in 0..rows {
            let m_val = mask_data[row];
            let start = row * last;
            for j in 0..last {
                data[start + j] *= m_val;
            }
        }
    } else if m_last == last {
        // Exact match
        for (val, &m) in data.iter_mut().zip(mask_data.iter()) {
            *val *= m;
        }
    } else {
        return Err(HypEmbedError::Tensor(format!(
            "Incompatible last dim: tensor has {}, mask has {}",
            last, m_last
        )));
    }

    Tensor::from_vec(data, tensor.shape().clone())
}

/// Sum along the specified axis, removing that dimension.
pub fn sum_along_axis(tensor: &Tensor, axis: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if axis >= dims.len() {
        return Err(HypEmbedError::Tensor(format!(
            "Axis {} out of range for shape {}",
            axis, tensor.shape()
        )));
    }

    let mut new_dims: Vec<usize> = dims.to_vec();
    new_dims.remove(axis);
    if new_dims.is_empty() {
        new_dims.push(1);
    }

    let new_shape = Shape::new(new_dims);
    let mut result = Tensor::zeros(new_shape.clone());
    let result_data = result.data_mut();

    let axis_size = dims[axis];
    let outer_size: usize = dims[..axis].iter().product();
    let inner_size: usize = dims[axis + 1..].iter().product();
    let data = tensor.data();

    for outer in 0..outer_size {
        for k in 0..axis_size {
            let src_offset = (outer * axis_size + k) * inner_size;
            let dst_offset = outer * inner_size;
            for inner in 0..inner_size {
                result_data[dst_offset + inner] += data[src_offset + inner];
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3])).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_bias() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        ).unwrap();
        let bias = Tensor::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3])).unwrap();
        let result = add_bias(&t, &bias).unwrap();
        let expected = [1.1, 2.2, 3.3, 4.1, 5.2, 6.3];
        for (a, b) in result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_vec(vec![2.0, 3.0], Shape::new(vec![2])).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0], Shape::new(vec![2])).unwrap();
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.data(), &[8.0, 15.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let b = scalar_mul(&a, 2.0);
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sum_along_axis() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        ).unwrap();
        let s = sum_along_axis(&t, 0).unwrap();
        assert_eq!(s.shape().dims(), &[3]);
        assert_eq!(s.data(), &[5.0, 7.0, 9.0]);

        let s1 = sum_along_axis(&t, 1).unwrap();
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(s1.data(), &[6.0, 15.0]);
    }
}
