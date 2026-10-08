# Browser demo

Runs PaddleOCR PP-OCRv6 entirely in the browser through the
`pure-onnx-ocr-wasm` bindings (`bindings/wasm`). OCR runs in a Web Worker, and
the detected boxes are drawn on a canvas.

## Build

Requirements:
- the `wasm32-unknown-unknown` Rust target
- `wasm-bindgen-cli` at the same version as the `wasm-bindgen` crate in
  `Cargo.lock` (currently 0.2.105)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.105 --locked

# From the repository root. .cargo/config.toml enables WebAssembly SIMD.
cargo build --release -p pure-onnx-ocr-wasm --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir examples/web/pkg \
  target/wasm32-unknown-unknown/release/pure_onnx_ocr_wasm.wasm
# or: scripts/build_wasm.sh
```

### Multi-threaded build (optional)

Cross-origin isolated pages can run inference on several threads. The build
needs a pinned nightly toolchain, which rustup installs from
`bindings/wasm/threads/rust-toolchain.toml` (details in
[bindings/wasm/threads/README.md](../../bindings/wasm/threads/README.md)):

```bash
scripts/build_wasm.sh --threads   # -> examples/web/pkg-threads
```

`worker.js` loads `pkg-threads` when the page is cross-origin isolated and
falls back to `pkg` otherwise (or when `pkg-threads` is missing), so build
both.

## Models

The page loads models from `examples/web/models/`, which is git-ignored. It
expects the same layout as the test fixtures:

```
examples/web/models/
  ppocrv6/{tiny,small,medium}_{det,rec}/inference.onnx, inference.yml
  PP-LCNet_x0_25_textline_ori/inference.onnx, inference.yml   # optional
  PP-LCNet_x1_0_doc_ori/inference.onnx, inference.yml         # optional
```

You can copy them from `tests/fixtures/models/` (see
`tests/fixtures/README.md` for the download commands).

## Run

Browsers refuse to load module workers and WebAssembly from `file://`, so
serve the directory over HTTP:

```bash
node examples/web/serve.mjs 8080          # single-threaded
node examples/web/serve.mjs 8080 --coi    # cross-origin isolated: multi-threaded
# open http://localhost:8080/
```

`--coi` adds `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` to every response. The status
line shows which build runs and on how many threads. The thread selector
starts a new Worker for each setting ("auto" = `navigator.hardwareConcurrency`).

Pick an image with the file input. The page can also run on load, which is
handy for automated checks: the output is stored in `window.ocrOutput`.

```
http://localhost:8080/?image=models/sample.jpg&tier=small&threads=4&textline=1&doc=1
```

## Cross-origin isolation on static hosting (GitHub Pages)

The multi-threaded build needs `SharedArrayBuffer`, which browsers only
enable when `crossOriginIsolated` is `true`, i.e. when the page is served
with COOP and COEP headers. Hosts that cannot set headers, such as GitHub
Pages, can add them with
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker): a
Service Worker that re-serves the site's responses with the headers.

1. Copy `coi-serviceworker.js` (MIT, keep its license notice) next to
   `index.html`, and load it first in `<head>`:
   ```html
   <script src="coi-serviceworker.js"></script>
   ```
   On the first visit it registers the Service Worker and reloads the page
   once; from then on `crossOriginIsolated` is `true`.
2. Put `worker.js`, `pkg/` and `pkg-threads/` inside the Service Worker's
   scope (the directory of `coi-serviceworker.js` and below). Only
   responses served through the Service Worker get the COEP header, and a
   Worker script without it is rejected in an isolated page
   (`ERR_BLOCKED_BY_RESPONSE`). This applies to `worker.js` and to the
   thread-pool Workers (`pkg-threads/snippets/*/src/workerHelpers.no-bundler.js`),
   which `worker.js` starts as nested Workers. Modules that a Worker
   imports may live outside the scope as long as they are same-origin (for
   example, a small `worker.js` inside the scope can `import` a shared one
   elsewhere).
3. Cross-origin files (models from Hugging Face, ...) must be fetched with
   CORS (`fetch()` does this by default) from a server that sends
   `Access-Control-Allow-Origin`, as Hugging Face does, redirects included.
4. coi-serviceworker uses `Cross-Origin-Embedder-Policy: credentialless` on
   Chrome and Firefox and `require-corp` on Safari, which does not support
   `credentialless`. With `require-corp`, cross-origin resources loaded
   without CORS (images, iframes) also need
   `Cross-Origin-Resource-Policy` headers.

Inference runs in a Web Worker as before (`worker.js`). Blocking waits
(`Atomics.wait`) are not allowed on the main thread, so call
`initThreadPool` and `run` from the Worker; rayon's threads are nested
Workers started from it (Chrome, Firefox, Safari 16.4+).

## Measured (headless Chrome 153, 896x528 boarding pass)

| Configuration | Total | Detection | Recognition |
| :--- | ---: | ---: | ---: |
| PP-OCRv6 tiny | 2.1 s | 0.7 s | 1.4 s |
| PP-OCRv6 small + text-line + document orientation | 7.7 s | 1.1 s | 5.9 s (orientation 0.6 s) |
| PP-OCRv6 medium | 30.4 s | 5.5 s | 24.9 s |

The recognised text matches the native build. The first run of each input
shape also includes compiling the inference plan.

## Multi-threaded build (headless Chromium 141, 4 vCPU)

Measured with `bench.mjs` (`serve.mjs --coi`) on a 1536x839 Japanese image
(`ja.jpg`, 50-55 regions). Each cell is the first run (includes compiling
the inference plans) / the median of the later runs / the WebAssembly memory
size after the runs, which is its peak. 8 threads exceed the 4 vCPUs of the
machine. More images and the comparison with v0.3.0 are in
[task-perf-010](../../docs/devlog/perf/task-perf-010-wasm-threads.md).

| Model | single (`pkg`) | threads: 1 | 2 | 4 | 8 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| PP-OCRv6 tiny | 2.7 / 2.1 s / 128 MiB | 3.0 / 1.9 s / 130 MiB | 2.2 / 1.4 s / 133 MiB | 1.8 / 0.9 s / 154 MiB | 1.9 / 0.9 s / 173 MiB |
| PP-OCRv6 small | 10.5 / 8.7 s / 277 MiB | 9.5 / 8.1 s / 279 MiB | 6.4 / 5.1 s / 289 MiB | 4.4 / 2.9 s / 302 MiB | 4.3 / 2.9 s / 345 MiB |
| PP-OCRv6 medium | 43.3 / 40.7 s / 613 MiB | 43.9 / 41.3 s / 676 MiB | 28.0 / 25.7 s / 685 MiB | 16.6 / 14.2 s / 847 MiB | 15.2 / 12.7 s / 833 MiB |
| PP-OCRv5 mobile | 10.8 / 8.6 s / 273 MiB | 9.9 / 8.1 s / 272 MiB | 6.5 / 4.7 s / 284 MiB | 4.7 / 2.6 s / 291 MiB | 4.4 / 2.6 s / 318 MiB |

Every configuration returned the same text and boxes as the native build
(`examples/ocr_json.rs`). PP-OCRv6 medium peaked at 1.2 GiB (1000x700
image, 8 threads), within the 2 GiB shared-memory maximum.

```bash
node examples/web/bench.mjs --models v6-tiny,v6-small,v6-medium,v5-mobile \
  --threads 1,2,4,8 --image models/ja.jpg [--native-dir DIR]
```
