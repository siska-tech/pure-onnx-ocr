# `pure-onnx-ocr`

Author: Shion Watanabe  
First version: 2025-11-09  
Revised: 2026-10-08 (v0.3.0)
Repository: http://github.com/siska-tech/pure-onnx-ocr

Pure Rust OCR pipeline that re-implements the PaddleOCR detection (DBNet) and CTC recognition models without relying on C/C++ runtimes. **PP-OCRv5 and PP-OCRv6 (tiny / small / medium) ONNX exports are supported.** The crate provides a high-level `OcrEngine` facade that hides detection and recognition stages behind a builder-style configuration API.

## Highlights

- **Pure Rust runtime** – no native shared libraries or FFI bindings; `cargo build` is enough.
- **DBNet + CTC pipeline** – mirrors PaddleOCR 3.x pre- and post-processing: BGR input, ImageNet normalisation, variable-width recognition, `box_thresh` filtering, and the space class.
- **PaddleOCR 3.x model directories** – point the builder at a directory with `inference.onnx` + `inference.yml` and the preprocessing settings and dictionary are read from `inference.yml`.
- **Extensible architecture** – detection, recognition, and geometry utilities are separated so you can swap or extend individual stages.
- **Portable** – runs where shipping C++ runtimes is difficult (embedded, serverless). **Verified in browsers (`wasm32-unknown-unknown`) and on WASI**; see [WebAssembly](#webassembly).

## Prerequisites

- Rust 1.91 or newer (stable channel; required by `tract-onnx` 0.23)
- CPU inference on x86\_64 or aarch64
- PaddleOCR ONNX models (PP-OCRv6 recommended; PP-OCRv5 also works)

## Installation

```toml
[dependencies]
pure_onnx_ocr = "0.3.0"
image = "0.25"       # recommended for image I/O
geo-types = "0.7"    # recommended for working with polygon results
```

### PP-OCRv6 (recommended)

Download `inference.onnx` and `inference.yml` from the Hugging Face repositories `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx` and keep each pair in its own directory. The dictionary is embedded in the recognition `inference.yml`, so no separate dictionary file is needed.

```bash
for kind in det rec; do
  mkdir -p models/ppocrv6/small_${kind}
  for f in inference.onnx inference.yml; do
    curl -L -o models/ppocrv6/small_${kind}/${f} \
      https://huggingface.co/PaddlePaddle/PP-OCRv6_small_${kind}_onnx/resolve/main/${f}
  done
done
```

| Tier | Notes | CPU time per 896x528 image (Core i7-1360P, 16 threads, after the first run) |
| :--- | :--- | :--- |
| `tiny` | Smallest. 6,904-character dictionary **without hiragana/katakana, so it cannot read Japanese** | ~0.4 s |
| `small` | 50 languages including Japanese. Good balance | ~1.1 s |
| `medium` | 50 languages. Most accurate (PaddleOCR 3.x default) | ~4.1 s |

See [Performance](#performance) for many-image throughput and a comparison with OpenVINO.

### PP-OCRv5

Individual files (`det.onnx`, `rec.onnx`, `ppocrv5_dict.txt`) still work. Directory-style exports such as `PaddlePaddle/PP-OCRv5_{mobile,server}_{det,rec}_onnx` can be used exactly like PP-OCRv6.

## Quick Start

With PP-OCRv6 model directories:

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError};

fn main() -> Result<(), OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_dir("models/ppocrv6/small_det") // inference.onnx + inference.yml
        .rec_model_dir("models/ppocrv6/small_rec") // dictionary comes from inference.yml
        .build()?;

    for result in engine.run_from_path("examples/demo.jpg")? {
        println!("{} ({:.3})", result.text, result.confidence);
    }
    Ok(())
}
```

With individual files (PP-OCRv5 text dictionary):

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrResult};

fn main() -> Result<(), pure_onnx_ocr::OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_path("models/ppocrv5/det.onnx")
        .rec_model_path("models/ppocrv5/rec.onnx")
        .dictionary_path("models/ppocrv5/ppocrv5_dict.txt")
        .det_limit_side_len(960)
        .det_unclip_ratio(1.5)
        .rec_batch_size(1)       // default; one crop per batch is fastest
        .inference_threads(8)    // default: logical CPUs, at most 16
        .build()?;

    let results: Vec<OcrResult> = engine.run_from_path("examples/demo.jpg")?;
    for (idx, result) in results.iter().enumerate() {
        println!(
            "#{} text={} confidence={:.4} polygon={:?}",
            idx,
            result.text,
            result.confidence,
            result.bounding_box.exterior().points()
        );
    }

    Ok(())
}
```

## Performance

Measured on a Core i7-1360P (16 threads) with PP-OCRv6 and tract 0.23.8, and compared with OpenVINO Runtime 2026.4.1 on the same PC, ONNX files and pre/post-processing. Every setup produced exactly the same output as OpenVINO.

| Usage | tiny | small | medium |
| :--- | ---: | ---: | ---: |
| One image at a time (896x528, after the first run) | ~0.4 s | ~1.1 s | ~4.1 s |
| Many images (`run_many_from_images`, 16 images) | 5.3 img/s | 1.5 img/s | 0.34 img/s |
| Same, relative to OpenVINO's fastest setup | 64% | 67% | 81% |

- With many images, peak memory is 23-65% of OpenVINO's.
- With improvements headed for the next tract release (including proposed fixes such as [sonos/tract#2976](https://github.com/sonos/tract/pull/2976)), many-image throughput reaches 79-99% of OpenVINO (medium on par).
- Details: [benchmark-openvino.md](docs/devlog/perf/benchmark-openvino.md) (one image at a time) and [benchmark-openvino-throughput.md](docs/devlog/perf/benchmark-openvino-throughput.md) (many images), both in Japanese. The tool lives in `tools/openvino-bench`.

Getting the most out of it:

- **Process several images with `run_many_from_paths` / `run_many_from_images`.** It is 1.5-2.4x faster than a `run_*` call per image with identical results, at 2.5-3.7x the peak memory while it runs.
- **Call `warmup(width, height)` at startup in servers.** Inference plans compile per input shape on first use, which slows the first image down; warming up makes the first image of that size 16-33% faster.
- Reuse the `OcrEngine`: it owns the loaded models and the compiled-plan cache.

```rust
let engine = OcrEngineBuilder::new()
    .det_model_dir("models/ppocrv6/small_det")
    .rec_model_dir("models/ppocrv6/small_rec")
    .build()?;
engine.warmup(1280, 720)?; // compile the plans for a common image size up front

let paths = ["a.jpg", "b.jpg", "c.jpg"];
for (path, results) in paths.iter().zip(engine.run_many_from_paths(&paths)) {
    match results {
        Ok(results) => println!("{path}: {} regions", results.len()),
        Err(error) => eprintln!("{path}: {error}"), // one failure does not stop the others
    }
}
```

## Smoke Testing with `ocr_smoke`

If you want to replicate the behaviour of the original `test_ocr.py` without leaving the Rust ecosystem, you can use the bundled `ocr_smoke` binary.

- By default it points to `models/ppocrv5/det.onnx`, `models/ppocrv5/rec.onnx`, and `models/ppocrv5/ppocrv5_dict.txt`.
- Example usage:

```bash
cargo run --bin ocr_smoke -- path/to/image.jpg

# Override model paths and runtime options
cargo run --bin ocr_smoke -- path/to/image.jpg \
  --det-model models/ppocrv5/det.onnx \
  --rec-model models/ppocrv5/rec.onnx \
  --dictionary models/ppocrv5/ppocrv5_dict.txt \
  --det-limit-side-len 960 \
  --det-unclip-ratio 1.5 \
  --rec-batch-size 1 \
  --threads 8

# PP-OCRv6 model directories
cargo run --release --bin ocr_smoke -- path/to/image.jpg \
  --det-model-dir models/ppocrv6/small_det \
  --rec-model-dir models/ppocrv6/small_rec

# Thresholds (defaults match the PaddleOCR 3.x pipeline: 0.3 / 0.6)
cargo run --release --bin ocr_smoke -- path/to/image.jpg \
  --det-model-dir models/ppocrv6/small_det \
  --rec-model-dir models/ppocrv6/small_rec \
  --det-thresh 0.3 --det-box-thresh 0.6
```

The CLI prints inference timing, recognised texts with confidences, and polygon coordinates. It exits with a descriptive error when the image or models are missing.

Detection resizes the long side, normalises in BGR order with ImageNet statistics, and pads to multiples of 32. Recognition keeps the 48 px height and aspect ratio, widens the input up to 3200 px for long lines, and batches crops sorted by aspect ratio.

### PaddleOCR 3.x options

| Feature | Builder | `ocr_smoke` | Default |
| :--- | :--- | :--- | :--- |
| Rotation-corrected crops (vertical regions rotated by 90 degrees) | `rec_crop_mode(RecCropMode::Rotated)` | `--crop-mode rotated\|axis` | on |
| Native-resolution detection (as PaddleOCR 3.x) | `det_limit_type(DetLimitType::Min).det_limit_side_len(64)` | `--det-limit-type min --det-limit-side-len 64` | downscale to 960 px long side |
| Detection thresholds from `inference.yml` | `det_postprocess_from_model_config(true)` | `--det-params-from-config` | pipeline defaults (0.3 / 0.6 / 1.5) |
| Page orientation correction (0/90/180/270) | `doc_orientation_model_dir("models/PP-LCNet_x1_0_doc_ori")` | `--doc-ori-model-dir DIR` | off |
| Text-line flip correction (0/180) | `textline_orientation_model_dir("models/PP-LCNet_x0_25_textline_ori")` | `--textline-ori-model-dir DIR` | off |
| Inference threads | `inference_threads(8)` | `--threads N` | logical CPUs, at most 16 (WebAssembly: the thread pool size; 1 in the single-threaded build) |
| Compiled plan cache limit | `plan_cache_capacity(4, 16)` | n/a | 4 detection / 16 recognition |
| Compile plans ahead of the first run | `engine.warmup(width, height)` | n/a | compiled on first use |
| Many images in one call (same results as one by one) | `engine.run_many_from_paths(&paths)` / `run_many_from_images(&images)` | n/a | one `run_*` call per image |
| Loading and inference logs | emitted through the `log` crate | `-v` / `--verbose` | warnings only |

The orientation classifiers are available on Hugging Face as `PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx` and `PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx`. An `x1_0` text-line classifier also exists, but `x0_25` is about 3x faster on tract and is recommended.

> **Known limitations:**
> - Inference uses as many threads as logical CPUs (at most 16) by default; `inference_threads(1)` runs single-threaded. In browsers (WebAssembly), multi-threading needs the thread-enabled build and a cross-origin isolated page ([Multi-threading in browsers](#multi-threading-in-browsers)).
> - PP-OCRv6 medium takes about 4.1 s per image on CPU (tract, 16 threads). Prefer tiny or small when speed matters. See `docs/devlog/ppocrv6/benchmark-v5-vs-v6.md` for a comparison with PP-OCRv5.
> - Text-line flip correction can miss short all-uppercase lines such as `TAIYUAN`.
> - Document unwarping (UVDoc) and layout analysis are not supported.
>
> Research notes and design decisions are in `docs/devlog/ppocrv6/`.

## WebAssembly

The crate runs in browsers (`wasm32-unknown-unknown`) and on WASI (`wasm32-wasip1`). Browsers have no file system, so models, `inference.yml` files and images are passed as bytes or text:

```rust
let engine = OcrEngineBuilder::new()
    .det_model_bytes(det_onnx)       // Vec<u8>
    .det_config_yaml(det_yaml)       // String
    .rec_model_bytes(rec_onnx)
    .rec_config_yaml(rec_yaml)       // also provides the dictionary
    .build()?;
let results = engine.run_from_bytes(&jpeg_bytes)?;
```

From JavaScript, use the wasm-bindgen bindings in `bindings/wasm`. Build steps and a demo running OCR in a Web Worker are in [examples/web/README.md](examples/web/README.md).

```js
const engine = new OcrEngineBuilder()
  .detModel(detOnnxBytes, detYamlText)
  .recModel(recOnnxBytes, recYamlText)
  .build();
const results = engine.run(imageBytes); // [{ text, confidence, box, polygon }, ...]
```

`.cargo/config.toml` enables WebAssembly SIMD (`simd128`), which is about 2x faster than without it. Measured in headless Chrome 153 on an 896x528 image:

| Model | Time |
| :--- | ---: |
| PP-OCRv6 tiny | 2.1 s |
| PP-OCRv6 small with page and text-line orientation | 7.7 s |
| PP-OCRv6 medium | 30.4 s |

`OcrEngine` is `Send + Sync`, so one engine wrapped in an `Arc` can serve several threads at once.

### Multi-threading in browsers

Cross-origin isolated pages can run inference on several threads (Web Workers): recognition batches and matrix multiplications run in parallel, with the same output as the single-threaded build. With PP-OCRv5 mobile and 4 threads, it was 2.4-3.0x faster than v0.3.0 (single-threaded) in headless Chromium on 4 vCPUs ([task-perf-010](docs/devlog/perf/task-perf-010-wasm-threads.md)).

- **Build**: shared memory needs nightly Rust and `-Z build-std`. `bindings/wasm/threads/` pins a nightly for this build only; the rest of the repository stays on stable. `scripts/build_wasm.sh --threads` writes `examples/web/pkg-threads`. See [bindings/wasm/threads/README.md](bindings/wasm/threads/README.md).
- **Headers**: serve the page (and the Worker scripts) with the headers below. The threaded build cannot load unless `crossOriginIsolated` is `true`.
  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp   (or credentialless)
  ```
  Hosts that cannot set headers, such as GitHub Pages, can use coi-serviceworker ([examples/web/README.md](examples/web/README.md)).
- **Usage**: in a Web Worker, call `initThreadPool` once before building the engine. rayon's threads are nested Workers started from that Worker.

```js
import init, { initThreadPool, OcrEngineBuilder } from "./pkg-threads/pure_onnx_ocr_wasm.js";
await init();
await initThreadPool(navigator.hardwareConcurrency);
const engine = new OcrEngineBuilder()
  .detModel(detOnnxBytes, detYamlText)
  .recModel(recOnnxBytes, recYamlText)
  .inferenceThreads(4)   // default: the pool size; larger values are capped to it
  .build();
```

The single-threaded build exports an `initThreadPool` that resolves without doing anything; `threadsSupported()` tells the builds apart. The demo (`examples/web/worker.js`) loads the threaded build on isolated pages and the single-threaded one elsewhere. Shared memory is capped at 2 GiB.

### Troubleshooting

- `ModelLoad`: `tract` rejected an operator that the ONNX graph requires (e.g., `LayerNormalization`, `Scan`). Try a simplified model or file an issue with model details.
- `ModelConfig`: an `inference.yml` could not be parsed. Only the block-style YAML that PaddleOCR emits is supported.
- `Dictionary`: ensure the dictionary file is encoded in UTF-8 without BOM.

## API Overview

| Symbol             | Description                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `OcrEngineBuilder` | Configures model paths and runtime parameters. Produces an `OcrEngine`. `det_model_dir` / `rec_model_dir` accept PaddleOCR model directories. |
| `PaddleInferenceConfig` | Reads preprocessing parameters, thresholds, and the dictionary from a PaddleOCR `inference.yml`.         |
| `OcrEngine`        | Facade that executes detection + recognition: `run_from_path` / `run_from_image` / `run_from_bytes`, `run_many_from_paths` / `run_many_from_images` for several images, and `warmup` to compile plans ahead of time. |
| `OcrResult`        | Holds the text, confidence score, and `Polygon` bounding box for a single region.                             |
| `OcrError`         | Enumerates all errors emitted by the library (I/O, model loading, preprocessing, inference, post-processing). |
| `Polygon`          | Re-export of `geo-types::Polygon`. Useful for downstream geometry processing.                                 |

For detailed behavior and error semantics, see `docs/interface_design_en.md`.

## Documentation Set

- Architecture: `docs/architecture_en.md`
- Detailed design: `docs/detail_design_en.md`
- Interface design: `docs/interface_design_en.md`
- Requirements: `docs/requirements_en.md`
- References: `docs/references_en.md`
- Test specification: `docs/test_specification_en.md`

Each English document mirrors the Japanese source to help international contributors understand the project.

## Project Status

- 2025-11-09: Completed PoC for `det.onnx` (DBNet) loading via `tract-onnx`.
- 2025-11-09: Validated `rec.onnx` (SVTR\_HGNet) dummy inference; confirmed output shape `[1, 40, 18385]`.
- 2025-11-09: Implemented detection preprocessing (`DetPreProcessor`) with resizing, normalization, and NCHW transforms.
- 2025-11-09: Implemented detection inference session with runnable caching per input resolution.
- 2025-11-09: Implemented detection post-processing (contour extraction and filtering).
- 2025-11-09: Implemented polygon unclipping via `i_overlay` buffering.
- 2025-11-09: Implemented polygon scaling back to original coordinates.
- 2025-11-09: Implemented recognition preprocessing with cropping, force resize, normalization, and batching.
- 2025-11-09: Implemented recognition inference session with batch execution.
- 2025-11-09: Implemented dictionary loader with dedupe and bidirectional mapping.
- 2025-11-09: Implemented Pure Rust CTC greedy decoder with duplicate suppression and blank removal.
- 2025-11-09: Implemented recognition post-processor that combines logits, CTC decoding, and dictionary lookup.
- 2025-11-09: Implemented `OcrEngineBuilder`, `OcrEngine`, and public error surface.
- 2025-11-09: Refreshed README and added bilingual documentation set (`task-doc-001`).
- 2025-11-09: Enhanced public Rustdoc coverage (`task-doc-002`) and validated `cargo doc` output.
- 2025-11-09: Completed Cargo metadata (`task-doc-003`) and `cargo package --no-verify` validation.
- 2025-11-09: Added integration tests (`task-doc-004`) with fixture strategy and CI guidance.
- 2025-11-10: Fixed the CTC blank index (`task-fix-001`), softmax-based confidence (`task-fix-002`) and benchmark timings (`task-fix-003`).
- 2026-10-03: **v0.2.0** (see `CHANGELOG.md` and `docs/devlog/`):
  - PP-OCRv6 tiny/small/medium and PaddleOCR 3.x model directories (`inference.yml`).
  - PaddleOCR 3.x-compatible processing with a parity test against PaddleOCR 3.7; v6 medium matches it exactly on the test images.
  - Rotation-corrected crops, page and text-line orientation classifiers, native-resolution detection.
  - Browser WebAssembly support (in-memory inputs, wasm-bindgen bindings, demo).
  - Multi-threaded inference with batch size 1: whole pipeline 2.9-4.7x faster.
  - GitHub Actions CI and a fixture download script.
- 2026-10-03: **v0.2.1**: fixed recognition region overflow and detection output
  shape validation; added regression tests and expanded source documentation.
- 2026-10-08: **v0.3.0**: performance work, benchmarked against OpenVINO under identical conditions (`docs/devlog/perf/`); output unchanged:
  - Default inference threads capped at 16 instead of 8 (3-18% faster end to end).
  - Each plan shape compiles once (first run 10-23% faster); `OcrEngine::warmup` added.
  - `run_many_from_paths` / `run_many_from_images` for several images (1.5-2.4x throughput).
  - Proposed tract PRs for depthwise convolutions and packing (sonos/tract#2976-#2978).

## Contributing

Issues and pull requests are welcome. Please:

- Run `cargo fmt` and `cargo clippy` before submitting patches.
- Add unit tests where possible.
- Update the corresponding task file in `docs/devlog/` when documentation or feature work progresses.

## License

Licensed under `Apache-2.0`, aligning with PaddleOCR, OnnxOCR, and tract licensing.

## Testing

- Fixtures: `scripts/fetch_fixtures.sh` (about 35 MB for the default suite; `--all` adds small/medium and more)
- Tests: `cargo test --release` (release mode recommended, inference is slow in debug)
- CI: GitHub Actions (`.github/workflows/ci.yml`) runs fmt, clippy, tests on Linux and Windows, an MSRV check and WebAssembly builds.
- PP-OCRv6 tests (`tests/ppocrv6.rs`): the tiny pipeline runs by default; small and medium run with `cargo test --release --test ppocrv6 -- --ignored`.
- Integration tests: provide PP-OCRv5 models and a demo image via the `PURE_ONNX_OCR_FIXTURE_DIR` environment variable or `tests/fixtures/`. See `tests/fixtures/README.md` for the expected directory structure. Tests skip automatically when fixtures are missing.
