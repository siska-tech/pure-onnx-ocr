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
```

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
node examples/web/serve.mjs 8080
# open http://localhost:8080/
```

Pick an image with the file input. The page can also run on load, which is
handy for automated checks: the output is stored in `window.ocrOutput`.

```
http://localhost:8080/?image=models/sample.jpg&tier=small&textline=1&doc=1
```

## Measured (headless Chrome 153, 896x528 boarding pass)

| Configuration | Total | Detection | Recognition |
| :--- | ---: | ---: | ---: |
| PP-OCRv6 tiny | 2.1 s | 0.7 s | 1.4 s |
| PP-OCRv6 small + text-line + document orientation | 7.7 s | 1.1 s | 5.9 s (orientation 0.6 s) |
| PP-OCRv6 medium | 30.4 s | 5.5 s | 24.9 s |

The recognised text matches the native build. The first run of each input
shape also includes compiling the inference plan.
