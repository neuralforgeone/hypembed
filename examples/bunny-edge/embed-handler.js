/**
 * Bunny Edge Script example: POST JSON { "text": "..." } -> embedding vector.
 *
 * Prerequisites:
 * - Regenerate glue: scripts\generate_bunny_glue.cmd (or verify_wasm.cmd)
 * - Deploy hypembed_wasm.js + hypembed_wasm_bg.wasm alongside this handler
 * - Provide model bytes (config.json, vocab.txt, model.safetensors) via edge storage or KV
 */

import init, { WasmEmbedder } from "./hypembed_wasm.js";

let embedder = null;
let wasmReady = false;

async function ensureWasm() {
  if (!wasmReady) {
    await init();
    wasmReady = true;
  }
}

async function loadEmbedder(env) {
  if (embedder) {
    return embedder;
  }

  await ensureWasm();

  const configJson = await env.MODEL_CONFIG.get("config.json", { type: "text" });
  const vocabTxt = await env.MODEL_VOCAB.get("vocab.txt", { type: "text" });
  const weights = await env.MODEL_WEIGHTS.get("model.safetensors", { type: "arrayBuffer" });

  embedder = new WasmEmbedder(configJson, vocabTxt, new Uint8Array(weights));
  return embedder;
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "POST required" }), {
        status: 405,
        headers: { "content-type": "application/json" },
      });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response(JSON.stringify({ error: "invalid JSON body" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const text = body?.text;
    if (typeof text !== "string" || text.trim().length === 0) {
      return new Response(JSON.stringify({ error: "text field required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    try {
      const model = await loadEmbedder(env);
      const vector = model.embed(text);
      return new Response(
        JSON.stringify({
          dim: vector.length,
          embedding: Array.from(vector),
        }),
        { headers: { "content-type": "application/json" } },
      );
    } catch (err) {
      return new Response(
        JSON.stringify({ error: String(err?.message ?? err) }),
        { status: 500, headers: { "content-type": "application/json" } },
      );
    }
  },
};