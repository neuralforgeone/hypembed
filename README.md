# HypEmbed

Pure-Rust text embedding inference for local-first applications.

[![CI](https://github.com/neuralforgeone/hypembed/actions/workflows/ci.yml/badge.svg)](https://github.com/neuralforgeone/hypembed/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hypembed.svg)](https://crates.io/crates/hypembed)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](./LICENSE-MIT)
[![Docs](https://img.shields.io/badge/docs-github.io-black.svg)](https://neuralforgeone.github.io/hypembed/)

HypEmbed is a Rust library for generating BERT-compatible text embeddings without Python, ONNX Runtime, libtorch, or hosted inference services. Load local model weights, tokenize input, run the encoder, and get normalized vectors from a small API surface.

## Why HypEmbed

- Pure Rust from tokenizer to encoder forward pass
- Local-first inference with no external ML runtime dependency
- BERT-family support for common embedding models such as MiniLM
- Correctness-focused math with stable softmax, layer norm, and normalization
- Performance-aware implementation with SIMD primitives, memory-mapped weights, and batch tokenization

## Current Scope

- Supports BERT-style encoder models, including BERT, MiniLM, and DistilBERT-style layouts
- Loads `config.json`, `vocab.txt`, and `model.safetensors` from a local model directory
- Offers mean pooling and CLS pooling
- Accepts F32, F16, and BF16 weights, converting to `f32` for inference
- Runs on CPU only

HypEmbed does not currently handle training, quantization, GPU execution, or direct Hugging Face Hub downloads.

## Installation

```bash
cargo add hypembed
```

## Quick Start

```rust
use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};

let model = Embedder::load("./model").unwrap();

let options = EmbeddingOptions::default()
    .with_normalize(true)
    .with_pooling(PoolingStrategy::Mean);

let embeddings = model
    .embed(&["hello world", "rust embeddings"], &options)
    .unwrap();

println!("Embedding dim: {}", embeddings[0].len());
println!("First 5 values: {:?}", &embeddings[0][..5]);
```

To try a complete example locally:

```bash
cargo run --example basic_embed -- ./path/to/model
```

## Model Directory

HypEmbed expects a local directory with:

| File | Description |
| --- | --- |
| `config.json` | Hugging Face style model configuration |
| `vocab.txt` | BERT WordPiece vocabulary |
| `model.safetensors` | SafeTensors weights |

Example compatible model:

- `sentence-transformers/all-MiniLM-L6-v2`

## Documentation

- Project site: https://neuralforgeone.github.io/hypembed/
- API docs: https://neuralforgeone.github.io/hypembed/api/hypembed/
- Crate docs (local): `cargo doc --open`

The public API is intentionally small: [`Embedder`](https://neuralforgeone.github.io/hypembed/api/hypembed/struct.Embedder.html), [`EmbeddingOptions`](https://neuralforgeone.github.io/hypembed/api/hypembed/struct.EmbeddingOptions.html), [`PoolingStrategy`](https://neuralforgeone.github.io/hypembed/api/hypembed/enum.PoolingStrategy.html), and [`HypEmbedError`](https://neuralforgeone.github.io/hypembed/api/hypembed/enum.HypEmbedError.html). Tensor, tokenizer, and model internals are not part of the stable contract.

## Design Notes

HypEmbed follows a simple pipeline:

```text
input text
  -> pre-tokenize and normalize
  -> WordPiece tokenize
  -> add special tokens, truncate, and pad
  -> embedding layer
  -> encoder stack
  -> mean or CLS pooling
  -> optional L2 normalization
  -> embedding vector
```

The project favors explicit behavior and stable numerics:

- softmax subtracts the row maximum before exponentiation
- layer norm uses epsilon guards
- pooling and vector normalization avoid divide-by-zero edge cases
- typed errors keep load and inference failures inspectable

## Open Source Status

HypEmbed is early-stage but already includes:

- cross-platform CI (test, clippy, fmt, doc, MSRV)
- self-contained tiny-model integration tests (no downloaded weights required)
- benchmark compilation checks
- generated API documentation with runnable examples

## License

Licensed under either of:

- Apache License, Version 2.0, see [LICENSE-APACHE](./LICENSE-APACHE)
- MIT license, see [LICENSE-MIT](./LICENSE-MIT)
