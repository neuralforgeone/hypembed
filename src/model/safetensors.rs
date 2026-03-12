/// SafeTensors format parser.
///
/// SafeTensors is a simple, safe binary format for storing tensors:
///
/// ```text
/// [8 bytes: header_size as u64 LE]
/// [header_size bytes: JSON metadata]
/// [remaining bytes: raw tensor data]
/// ```
///
/// The JSON metadata maps tensor names to their dtype, shape, and byte offsets
/// within the data section. This parser extracts f32 tensors from the file.
///
/// No external SafeTensors crate is used — the format is simple enough
/// to parse directly.
///
/// ## v0.2: Memory-mapped loading
///
/// `load_mmap()` uses `memmap2` to memory-map the file, avoiding a full
/// copy into heap memory. The `DataStore` enum abstracts over owned vs mapped storage.

use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use serde::Deserialize;
use memmap2::Mmap;
use crate::error::{HypEmbedError, Result};
use crate::tensor::{Tensor, Shape};

/// Metadata for a single tensor in a SafeTensors file.
#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    /// Data type string (e.g., "F32", "F16", "BF16").
    pub dtype: String,
    /// Shape as a list of dimension sizes.
    pub shape: Vec<usize>,
    /// Byte offsets [start, end) within the data section.
    pub data_offsets: [usize; 2],
}

/// Abstraction over owned and memory-mapped byte storage.
#[derive(Debug)]
enum DataStore {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl AsRef<[u8]> for DataStore {
    fn as_ref(&self) -> &[u8] {
        match self {
            DataStore::Owned(v) => v,
            DataStore::Mapped(m) => m,
        }
    }
}

/// A parsed SafeTensors file.
#[derive(Debug)]
pub struct SafeTensorsFile {
    /// Tensor metadata by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Raw data section (after the header).
    data: DataStore,
    /// Offset where tensor data starts (header_end) in the original file.
    /// For owned data this is 0 (data is already sliced).
    /// For mmap this is the offset to add.
    data_offset: usize,
}

impl SafeTensorsFile {
    /// Load and parse a SafeTensors file (reads entire file into memory).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;

        if bytes.len() < 8 {
            return Err(HypEmbedError::Model("SafeTensors file too small".into()));
        }

        let header_size = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;

        let header_end = 8 + header_size;
        if header_end > bytes.len() {
            return Err(HypEmbedError::Model(format!(
                "SafeTensors header extends beyond file: header_end={}, file_size={}",
                header_end,
                bytes.len()
            )));
        }

        let tensors = Self::parse_header(&bytes[8..header_end])?;
        let data = bytes[header_end..].to_vec();

        Ok(SafeTensorsFile {
            tensors,
            data: DataStore::Owned(data),
            data_offset: 0,
        })
    }

    /// Load and parse a SafeTensors file using memory-mapped I/O.
    ///
    /// This avoids copying the entire file into heap memory, which significantly
    /// reduces startup time and memory usage for large models.
    pub fn load_mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: The file is opened read-only and we keep the Mmap alive
        // for the lifetime of the SafeTensorsFile. The file must not be
        // modified externally while mapped.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(HypEmbedError::Model("SafeTensors file too small".into()));
        }

        let header_size = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3],
            mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        let header_end = 8 + header_size;
        if header_end > mmap.len() {
            return Err(HypEmbedError::Model(format!(
                "SafeTensors header extends beyond file: header_end={}, file_size={}",
                header_end,
                mmap.len()
            )));
        }

        let tensors = Self::parse_header(&mmap[8..header_end])?;

        Ok(SafeTensorsFile {
            tensors,
            data: DataStore::Mapped(mmap),
            data_offset: header_end,
        })
    }

    /// Parse the JSON header into tensor metadata.
    fn parse_header(header_bytes: &[u8]) -> Result<HashMap<String, TensorInfo>> {
        let header_json = std::str::from_utf8(header_bytes).map_err(|e| {
            HypEmbedError::Model(format!("Invalid UTF-8 in SafeTensors header: {}", e))
        })?;

        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;

        let mut tensors = HashMap::new();
        for (name, value) in &raw {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
                HypEmbedError::Model(format!(
                    "Failed to parse tensor info for '{}': {}",
                    name, e
                ))
            })?;
            tensors.insert(name.clone(), info);
        }

        Ok(tensors)
    }

    /// Get the raw data bytes for tensor extraction.
    fn data_bytes(&self) -> &[u8] {
        let full = self.data.as_ref();
        &full[self.data_offset..]
    }

    /// Extract a tensor by name as f32.
    ///
    /// Supports F32, F16, and BF16 dtypes. F16/BF16 are converted to F32.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            HypEmbedError::Model(format!("Tensor '{}' not found in SafeTensors file", name))
        })?;

        let start = info.data_offsets[0];
        let end = info.data_offsets[1];
        let data = self.data_bytes();

        if end > data.len() {
            return Err(HypEmbedError::Model(format!(
                "Tensor '{}' data offsets [{}, {}) exceed data section size {}",
                name, start, end, data.len()
            )));
        }

        let raw = &data[start..end];
        let shape = Shape::new(info.shape.clone());

        match info.dtype.as_str() {
            "F32" => {
                let expected_bytes = shape.numel() * 4;
                if raw.len() != expected_bytes {
                    return Err(HypEmbedError::Model(format!(
                        "Tensor '{}': expected {} bytes for F32 shape {:?}, got {}",
                        name, expected_bytes, info.shape, raw.len()
                    )));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape)
            }
            "F16" => {
                let expected_bytes = shape.numel() * 2;
                if raw.len() != expected_bytes {
                    return Err(HypEmbedError::Model(format!(
                        "Tensor '{}': expected {} bytes for F16 shape {:?}, got {}",
                        name, expected_bytes, info.shape, raw.len()
                    )));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect();
                Tensor::from_vec(floats, shape)
            }
            "BF16" => {
                let expected_bytes = shape.numel() * 2;
                if raw.len() != expected_bytes {
                    return Err(HypEmbedError::Model(format!(
                        "Tensor '{}': expected {} bytes for BF16 shape {:?}, got {}",
                        name, expected_bytes, info.shape, raw.len()
                    )));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect();
                Tensor::from_vec(floats, shape)
            }
            other => Err(HypEmbedError::Model(format!(
                "Unsupported dtype '{}' for tensor '{}'",
                other, name
            ))),
        }
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Convert an IEEE 754 half-precision (f16) value to f32.
///
/// This implements the standard conversion:
/// - Sign: bit 15
/// - Exponent: bits 14-10 (5 bits, biased by 15)
/// - Mantissa: bits 9-0 (10 bits)
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            return f32::from_bits(sign << 31);
        }
        // Subnormal: (−1)^sign × 2^(−14) × (0.mantissa)
        // Convert to f32 by normalizing
        let mut m = mant;
        let mut e = -14i32 + 127; // f32 bias
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // Remove leading 1
        let f32_bits = (sign << 31) | ((e as u32) << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }

    if exp == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal: adjust exponent from f16 bias (15) to f32 bias (127)
    let f32_exp = exp + 127 - 15;
    let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Convert a Brain Floating Point (BF16) value to f32.
///
/// BF16 has the same exponent range as f32 (8 bits) but only 7 bits of mantissa.
/// Conversion is trivial: shift left by 16 bits to fill the f32 bit pattern.
///
/// Layout: sign(1) + exponent(8) + mantissa(7) = 16 bits
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_neg_two() {
        let val = f16_to_f32(0xC000);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_half() {
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_one() {
        // BF16 1.0 = 0_01111111_0000000 = 0x3F80
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_neg_two() {
        // BF16 -2.0 = 1_10000000_0000000 = 0xC000
        let val = bf16_to_f32(0xC000);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_bf16_to_f32_half() {
        // BF16 0.5 = 0_01111110_0000000 = 0x3F00
        let val = bf16_to_f32(0x3F00);
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_pi_approx() {
        // BF16 approximation of pi: 0x4049 → should be ~3.140625
        let val = bf16_to_f32(0x4049);
        assert!((val - std::f32::consts::PI).abs() < 0.02);
    }
}
