/// SIMD-accelerated primitives for tensor operations.
///
/// Provides optimized inner loop functions using:
/// - **AVX2** (x86_64): 256-bit SIMD, 8 f32 lanes
/// - **SSE2** (x86_64): 128-bit SIMD, 4 f32 lanes
/// - **Scalar** fallback for all other platforms
///
/// Runtime feature detection ensures the fastest available path is used.

/// Add scaled source to destination: `dst[i] += src[i] * scale`
///
/// This is the matmul inner loop: scatter A[i,k] across the output row.
#[inline]
pub fn add_assign_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We've verified AVX2 is available and slice lengths match.
            unsafe { add_assign_scaled_avx2(dst, src, scale) };
            return;
        }
    }

    // Scalar fallback
    for i in 0..n {
        dst[i] += src[i] * scale;
    }
}

/// Dot product of two slices: `sum(a[i] * b[i])`.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We've verified AVX2 is available and slice lengths match.
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Element-wise addition: `out[i] = a[i] + b[i]`.
#[inline]
pub fn elementwise_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { elementwise_add_avx2(a, b, out) };
            return;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
#[inline]
pub fn elementwise_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { elementwise_mul_avx2(a, b, out) };
            return;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Scalar multiply: `out[i] = src[i] * scale`.
#[inline]
pub fn scalar_mul_slice(src: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(src.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { scalar_mul_avx2(src, scale, out) };
            return;
        }
    }

    for i in 0..src.len() {
        out[i] = src[i] * scale;
    }
}

/// Sum all elements in a slice.
#[inline]
pub fn sum_slice(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sum_avx2(data) };
        }
    }

    data.iter().sum()
}

// ── AVX2 implementations ──

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_scaled_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let scale_vec = _mm256_set1_ps(scale);
    let chunks = n / 8;
    let remainder = n % 8;

    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let d = _mm256_loadu_ps(dp.add(offset));
        let s = _mm256_loadu_ps(sp.add(offset));
        let result = _mm256_add_ps(d, _mm256_mul_ps(s, scale_vec));
        _mm256_storeu_ps(dp.add(offset), result);
    }

    // Handle remaining elements
    let tail_start = chunks * 8;
    for i in 0..remainder {
        *dp.add(tail_start + i) += *sp.add(tail_start + i) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(ap.add(offset));
        let vb = _mm256_loadu_ps(bp.add(offset));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    // Horizontal sum of AVX register
    // [a0+a4, a1+a5, a2+a6, a3+a7] via 128-bit halves
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    // [s0+s2, s1+s3, ...]
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Tail
    let tail_start = chunks * 8;
    for i in 0..remainder {
        total += *ap.add(tail_start + i) * *bp.add(tail_start + i);
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(ap.add(offset));
        let vb = _mm256_loadu_ps(bp.add(offset));
        _mm256_storeu_ps(op.add(offset), _mm256_add_ps(va, vb));
    }

    let tail = chunks * 8;
    for i in tail..n {
        *op.add(i) = *ap.add(i) + *bp.add(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(ap.add(offset));
        let vb = _mm256_loadu_ps(bp.add(offset));
        _mm256_storeu_ps(op.add(offset), _mm256_mul_ps(va, vb));
    }

    let tail = chunks * 8;
    for i in tail..n {
        *op.add(i) = *ap.add(i) * *bp.add(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_avx2(src: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let chunks = n / 8;
    let scale_vec = _mm256_set1_ps(scale);
    let sp = src.as_ptr();
    let op = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(sp.add(offset));
        _mm256_storeu_ps(op.add(offset), _mm256_mul_ps(v, scale_vec));
    }

    let tail = chunks * 8;
    for i in tail..n {
        *op.add(i) = *sp.add(i) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = data.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    let dp = data.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(dp.add(offset));
        acc = _mm256_add_ps(acc, v);
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    let tail = chunks * 8;
    for i in tail..n {
        total += *dp.add(i);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_assign_scaled() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let expected: Vec<f32> = dst.iter().zip(src.iter()).map(|(&d, &s)| d + s * 0.5).collect();
        add_assign_scaled(&mut dst, &src, 0.5);
        for (a, b) in dst.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_product(&a, &b);
        assert!((result - expected).abs() < 1e-4, "{} != {}", result, expected);
    }

    #[test]
    fn test_elementwise_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0];
        let mut out = vec![0.0; a.len()];
        elementwise_add(&a, &b, &mut out);
        for i in 0..a.len() {
            assert!((out[i] - (a[i] + b[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn test_elementwise_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let mut out = vec![0.0; a.len()];
        elementwise_mul(&a, &b, &mut out);
        for i in 0..a.len() {
            assert!((out[i] - (a[i] * b[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn test_scalar_mul_slice() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0; src.len()];
        scalar_mul_slice(&src, 3.0, &mut out);
        for i in 0..src.len() {
            assert!((out[i] - src[i] * 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sum_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = sum_slice(&data);
        assert!((result - 55.0).abs() < 1e-4);
    }

    #[test]
    fn test_dot_product_empty() {
        let result = dot_product(&[], &[]);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_assign_scaled_small() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0, 30.0];
        add_assign_scaled(&mut dst, &src, 2.0);
        assert_eq!(dst, vec![21.0, 42.0, 63.0]);
    }
}
