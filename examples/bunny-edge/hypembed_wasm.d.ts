/* tslint:disable */
/* eslint-disable */

/**
 * WASM embedder wrapping the hypembed core pipeline.
 */
export class WasmEmbedder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Embed a single text and return the vector as a JS Float32Array-compatible Vec.
     */
    embed(text: string): Float32Array;
    /**
     * Return the model hidden size (embedding dimension).
     */
    hidden_size(): number;
    /**
     * Create an embedder from in-memory model bytes (no filesystem access).
     */
    constructor(config_json: string, vocab_txt: string, weights: Uint8Array);
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmembedder_free: (a: number, b: number) => void;
    readonly wasmembedder_embed: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmembedder_hidden_size: (a: number) => number;
    readonly wasmembedder_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
