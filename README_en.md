# `pure-onnx-ocr`

Author: Shion Watanabe  
Date: 2025-11-09  
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
pure_onnx_ocr = "0.1.0"
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

| Tier | Notes | CPU time per 896x528 image (Core i7-1360P, 8 threads) |
| :--- | :--- | :--- |
| `tiny` | Smallest. 6,904-character dictionary **without hiragana/katakana, so it cannot read Japanese** | ~0.5 s |
| `small` | 50 languages including Japanese. Good balance | ~1.3 s |
| `medium` | 50 languages. Most accurate (PaddleOCR 3.x default) | ~4.5 s |

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
        .inference_threads(8)    // default: logical CPUs, at most 8
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
| Inference threads | `inference_threads(8)` | `--threads N` | logical CPUs, at most 8 (1 on WebAssembly) |
| Compiled plan cache limit | `plan_cache_capacity(4, 16)` | n/a | 4 detection / 16 recognition |
| Loading and inference logs | emitted through the `log` crate | `-v` / `--verbose` | warnings only |

The orientation classifiers are available on Hugging Face as `PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx` and `PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx`. An `x1_0` text-line classifier also exists, but `x0_25` is about 3x faster on tract and is recommended.

> **Known limitations:**
> - Inference uses as many threads as logical CPUs (at most 8) by default; `inference_threads(1)` runs single-threaded. Browsers (WebAssembly) always run single-threaded.
> - PP-OCRv6 medium takes about 4.5 s per image on CPU (tract, 8 threads). Prefer tiny or small when speed matters. See `docs/devlog/ppocrv6/benchmark-v5-vs-v6.md` for a comparison with PP-OCRv5.
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

### Troubleshooting

- `ModelLoad`: `tract` rejected an operator that the ONNX graph requires (e.g., `LayerNormalization`, `Scan`). Try a simplified model or file an issue with model details.
- `ModelConfig`: an `inference.yml` could not be parsed. Only the block-style YAML that PaddleOCR emits is supported.
- `Dictionary`: ensure the dictionary file is encoded in UTF-8 without BOM.

## API Overview

| Symbol             | Description                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `OcrEngineBuilder` | Configures model paths and runtime parameters. Produces an `OcrEngine`. `det_model_dir` / `rec_model_dir` accept PaddleOCR model directories. |
| `PaddleInferenceConfig` | Reads preprocessing parameters, thresholds, and the dictionary from a PaddleOCR `inference.yml`.         |
| `OcrEngine`        | Facade that executes detection + recognition. Provides `run_from_path` and `run_from_image`.                  |
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

## Contributing

Issues and pull requests are welcome. Please:

- Run `cargo fmt` and `cargo clippy` before submitting patches.
- Add unit tests where possible.
- Update the corresponding task file in `docs/devlog/` when documentation or feature work progresses.

## License

Licensed under `Apache-2.0`, aligning with PaddleOCR, OnnxOCR, and tract licensing.

## Testing

- Unit tests: `cargo test`
- PP-OCRv6 tests (`tests/ppocrv6.rs`): the tiny pipeline runs by default; small and medium run with `cargo test --release --test ppocrv6 -- --ignored`.
- Integration tests: provide PP-OCRv5 models and a demo image via the `PURE_ONNX_OCR_FIXTURE_DIR` environment variable or `tests/fixtures/`. See `tests/fixtures/README.md` for the expected directory structure. Tests skip automatically when fixtures are missing.

