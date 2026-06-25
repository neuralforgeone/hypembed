mod common {
    include!("../../tests/common/mod.rs");
}

use hypembed_wasm::WasmEmbedder;

#[test]
fn wasm_embedder_embeds_from_bytes_on_native_host() {
    let weights = common::tiny_safetensors_bytes();
    let vocab = common::tiny_vocab_txt();
    let embedder = WasmEmbedder::new(common::TINY_CONFIG_JSON, &vocab, &weights)
        .expect("construct wasm embedder");

    assert_eq!(embedder.hidden_size(), 8);
    let vector = embedder.embed("hello rust").expect("embed");
    assert_eq!(vector.len(), 8);
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
}
