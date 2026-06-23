// Shared test fixtures for byte-based model loading.

use std::collections::HashMap;

use hypembed::tokenizer::vocab::{CLS_TOKEN, MASK_TOKEN, PAD_TOKEN, SEP_TOKEN, UNK_TOKEN};

/// Minimal BERT config for integration tests.
pub const TINY_CONFIG_JSON: &str = r#"{
    "vocab_size": 20,
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 16,
    "max_position_embeddings": 32,
    "type_vocab_size": 2,
    "model_type": "bert"
}"#;

/// Minimal vocab with required special tokens (20 entries, vocab_size=20).
pub fn tiny_vocab_txt() -> String {
    [
        "",
        "[unused1]",
        "[unused2]",
        "[unused3]",
        "[unused4]",
        "[unused5]",
        PAD_TOKEN,
        UNK_TOKEN,
        CLS_TOKEN,
        SEP_TOKEN,
        MASK_TOKEN,
        "hello",
        "world",
        "rust",
        "embedding",
        "machine",
        "learning",
        "semantic",
        "search",
        "the",
    ]
    .join("\n")
        + "\n"
}

/// Build a minimal valid SafeTensors file for the tiny BERT config above.
pub fn tiny_safetensors_bytes() -> Vec<u8> {
    let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    let add = |map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
               name: &str,
               shape: &[usize],
               fill: f32| {
        let n: usize = shape.iter().product();
        map.insert(name.to_string(), (shape.to_vec(), vec![fill; n]));
    };

    let mut word_vals = Vec::with_capacity(20 * 8);
    for i in 0..(20 * 8) {
        word_vals.push(0.01 * ((i % 17) as f32 + 1.0));
    }
    tensors.insert(
        "embeddings.word_embeddings.weight".to_string(),
        (vec![20, 8], word_vals),
    );
    add(&mut tensors, "embeddings.position_embeddings.weight", &[32, 8], 0.01);
    add(&mut tensors, "embeddings.token_type_embeddings.weight", &[2, 8], 0.01);
    add(&mut tensors, "embeddings.LayerNorm.weight", &[8], 1.0);
    add(&mut tensors, "embeddings.LayerNorm.bias", &[8], 0.0);

    let layer = "encoder.layer.0";
    for name in [
        "attention.self.query.weight",
        "attention.self.key.weight",
        "attention.self.value.weight",
        "attention.output.dense.weight",
    ] {
        add(
            &mut tensors,
            &format!("{layer}.{name}"),
            &[8, 8],
            0.01,
        );
    }
    for name in [
        "attention.self.query.bias",
        "attention.self.key.bias",
        "attention.self.value.bias",
        "attention.output.dense.bias",
        "attention.output.LayerNorm.weight",
        "attention.output.LayerNorm.bias",
    ] {
        let shape = if name.ends_with("weight") {
            vec![8]
        } else {
            vec![8]
        };
        let fill = if name.ends_with("weight") { 1.0 } else { 0.0 };
        add(&mut tensors, &format!("{layer}.{name}"), &shape, fill);
    }

    add(
        &mut tensors,
        &format!("{layer}.intermediate.dense.weight"),
        &[16, 8],
        0.01,
    );
    add(
        &mut tensors,
        &format!("{layer}.intermediate.dense.bias"),
        &[16],
        0.0,
    );
    add(
        &mut tensors,
        &format!("{layer}.output.dense.weight"),
        &[8, 16],
        0.01,
    );
    add(
        &mut tensors,
        &format!("{layer}.output.dense.bias"),
        &[8],
        0.0,
    );
    add(
        &mut tensors,
        &format!("{layer}.output.LayerNorm.weight"),
        &[8],
        1.0,
    );
    add(
        &mut tensors,
        &format!("{layer}.output.LayerNorm.bias"),
        &[8],
        0.0,
    );

    build_safetensors(&tensors)
}

fn build_safetensors(tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();

    for (name, (shape, values)) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }

    let header_json = serde_json::to_string(&header).expect("header json");
    let header_bytes = header_json.as_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data);
    out
}

/// Write tiny model files to a directory for CLI integration tests.
pub fn write_tiny_model_dir(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("create model dir");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).expect("write config");
    std::fs::write(dir.join("vocab.txt"), tiny_vocab_txt()).expect("write vocab");
    std::fs::write(dir.join("model.safetensors"), tiny_safetensors_bytes()).expect("write weights");
}