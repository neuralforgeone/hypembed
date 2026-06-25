use wasm_bindgen::prelude::*;

use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};

/// WASM embedder wrapping the hypembed core pipeline.
#[wasm_bindgen]
pub struct WasmEmbedder {
    inner: Embedder,
}

#[wasm_bindgen]
impl WasmEmbedder {
    /// Create an embedder from in-memory model bytes (no filesystem access).
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        vocab_txt: &str,
        weights: &[u8],
    ) -> Result<WasmEmbedder, JsValue> {
        let inner = Embedder::from_bytes(config_json, vocab_txt, weights, true)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmEmbedder { inner })
    }

    /// Return the model hidden size (embedding dimension).
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    /// Embed a single text and return the vector as a JS Float32Array-compatible Vec.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        let max_length = self
            .inner
            .max_position_embeddings()
            .min(EmbeddingOptions::default().max_length);
        let options = EmbeddingOptions::default()
            .with_max_length(max_length)
            .with_pooling(PoolingStrategy::Mean)
            .with_normalize(true);
        let out = self
            .inner
            .embed(&[text], &options)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        out.into_iter()
            .next()
            .ok_or_else(|| JsValue::from_str("empty embedding result"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_embedder_api_compiles_on_native() {
        // Structural check: constructor signature is available on native host builds.
        let _ = WasmEmbedder::new;
    }
}
