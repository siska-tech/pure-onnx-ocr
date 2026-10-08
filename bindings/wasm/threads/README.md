# Multi-threaded browser build

This directory builds `pure-onnx-ocr-wasm` with the `threads` feature: a
WebAssembly module with shared memory whose inference runs on a pool of Web
Workers (through [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon)
and tract's `Executor::RayonGlobal`). Running cargo from here picks up:

- `rust-toolchain.toml`: a pinned nightly with `rust-src`. Shared memory needs
  a standard library compiled with atomics, so the build uses
  `-Z build-std`, which is nightly-only. The rest of the repository keeps
  building on stable.
- `.cargo/config.toml`: the `wasm32-unknown-unknown` target, the
  `+atomics,+bulk-memory,+mutable-globals,+simd128` target features, the
  linker flags for shared memory (2 GiB maximum), `build-std`, and
  `target/wasm-threads` as the target directory, so the stable build's
  artifacts are left alone.

## Build

```bash
cargo install wasm-bindgen-cli --version 0.2.105 --locked   # = wasm-bindgen in Cargo.lock

# From the repository root:
scripts/build_wasm.sh --threads              # -> examples/web/pkg-threads
scripts/build_wasm.sh --threads --out-dir DIR

# or by hand:
cd bindings/wasm/threads
cargo build --release -p pure-onnx-ocr-wasm --features threads
cd ../../..
wasm-bindgen --target web --out-dir examples/web/pkg-threads \
  target/wasm-threads/wasm32-unknown-unknown/release/pure_onnx_ocr_wasm.wasm
```

rustup installs the pinned nightly on first use (`rustup toolchain install`
from this directory does it explicitly). The first build compiles the
standard library and takes a few minutes.

## Output

| File | Purpose |
| :--- | :--- |
| `pure_onnx_ocr_wasm.js` | ES module: `init`, `initThreadPool`, `threadsSupported`, `OcrEngineBuilder`, `OcrEngine` |
| `pure_onnx_ocr_wasm_bg.wasm` | The module (imports a shared `WebAssembly.Memory`) |
| `pure_onnx_ocr_wasm.d.ts`, `pure_onnx_ocr_wasm_bg.wasm.d.ts` | TypeScript declarations |
| `snippets/wasm-bindgen-rayon-<hash>/src/workerHelpers.no-bundler.js` | Script of the thread-pool Workers |

Deploy the directory as is: `workerHelpers.no-bundler.js` finds the main
module through relative URLs. The hash in the snippet directory changes only
with the wasm-bindgen-rayon version.

## Use

```js
// In a Web Worker of a cross-origin isolated page:
import init, { initThreadPool, OcrEngineBuilder } from "./pkg-threads/pure_onnx_ocr_wasm.js";
await init();
await initThreadPool(navigator.hardwareConcurrency); // once per Worker, before build()
const engine = new OcrEngineBuilder()
  .detModel(detOnnx, detYml)
  .recModel(recOnnx, recYml)
  // .inferenceThreads(n)  default: the pool size; larger values are capped to it
  .build();
const results = engine.run(imageBytes);
```

- The page needs `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`), so that
  `crossOriginIsolated` is `true`. Without it there is no `SharedArrayBuffer`
  and the module cannot be instantiated: load the single-threaded build
  instead (`examples/web/worker.js` does this).
- Call `initThreadPool` and run inference from a Web Worker. The calling
  thread blocks while the pool works, which the main thread may not do. The
  pool threads are nested Workers started from that Worker.
- `initThreadPool` can be called only once per Worker. An engine built
  before it runs single-threaded and makes a later `initThreadPool` fail.
- The single-threaded build exports the same functions: `initThreadPool`
  resolves without doing anything and `threadsSupported()` returns `false`.

## Memory

Shared memory has to declare its maximum when it is created; it is 2 GiB
here (`--max-memory` in `.cargo/config.toml`). Browsers reserve address
space for the maximum, and 4 GiB reservations fail on some mobile devices.
PP-OCRv6 medium uses about 1 GiB. Measurements are in
`docs/devlog/perf/task-perf-010-wasm-threads.md`.
