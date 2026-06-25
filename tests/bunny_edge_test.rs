use std::path::PathBuf;

#[test]
fn bunny_edge_handler_wires_wasm_bindgen_glue() {
    let handler = include_str!("../examples/bunny-edge/embed-handler.js");
    let glue = include_str!("../examples/bunny-edge/hypembed_wasm.js");

    assert!(handler.contains("import init, { WasmEmbedder } from \"./hypembed_wasm.js\""));
    assert!(handler.contains("await init()"));
    assert!(handler.contains("async fetch(request, env)"));
    assert!(handler.contains("model.embed(text)"));

    assert!(glue.contains("export class WasmEmbedder"));
    assert!(glue.contains("__wbg_ptr"));
    assert!(glue.contains("hypembed_wasm_bg.wasm"));
    assert!(
        !glue.contains("fnv1a"),
        "bunny-edge glue must be wasm-bindgen output, not the dev shim"
    );
}

#[test]
fn bunny_edge_wasm_artifact_present() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wasm_path = manifest
        .join("examples")
        .join("bunny-edge")
        .join("hypembed_wasm_bg.wasm");

    let meta = std::fs::metadata(&wasm_path)
        .unwrap_or_else(|e| panic!("missing wasm artifact {}: {e}", wasm_path.display()));

    assert!(
        meta.len() > 10_000,
        "wasm artifact too small ({} bytes); rebuild hypembed-wasm",
        meta.len()
    );
}
