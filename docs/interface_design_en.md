# Interface Design (API): Pure Rust OnnxOCR

Author: Shion Watanabe  
First version: 2025-11-09  
Revised: 2026-10-03 (v0.2.0)  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

- Define the public API contract: types, methods, errors and defaults.
- Record what v0.2.0 adds and changes: PP-OCRv6 model directories, in-memory inputs, orientation classifiers and thread settings.
- rustdoc (`cargo doc`) is authoritative for exact signatures. This document captures intent and defaults.

## 1. Public API overview

### Modules

The crate is `pure_onnx_ocr`, and the main types are re-exported at the root.

| Module | Role |
| :--- | :--- |
| `engine` | `OcrEngineBuilder`, `OcrEngine`, `OcrResult`, `OcrError`, timing types |
| `preprocessing` / `postprocessing` | Detection and recognition stages (`DetPreProcessor`, `DetPostProcessor::db_boxes`, `RecPreProcessor`, …) |
| `detection` / `recognition` / `ctc` | tract inference sessions, CTC decoding |
| `dictionary` | `RecDictionary` (text file or `inference.yml`) |
| `paddle_config` | `PaddleInferenceConfig` (`inference.yml` reader) |
| `crop` | Minimum-area rectangle and perspective crop |
| `orientation` | Page and text-line orientation classifiers |
| `imgproc` | OpenCV-compatible bilinear resize |

### Structs

| Type | Main members |
| :--- | :--- |
| `OcrEngineBuilder` | `new`, model sources and parameters (below), `build` |
| `OcrEngine` | `run_from_path`, `run_from_image`, `run_from_bytes`, `run_with_metrics_{path,image,bytes}`, `config`, `det_model_path`, `rec_model_path`, `dictionary_path`, `rec_batch_size` |
| `OcrResult` | `text: String`, `confidence: f32`, `bounding_box: Polygon<f64>` |
| `OcrRunWithMetrics` | `results`, `timings: OcrTimings`, `doc_orientation_angle: Option<u32>` |
| `OcrTimings` / `StageTimings` | Total time, decode time, orientation time, and preprocess / inference / post-process time for each pipeline |
| `OcrEngineConfig` | The effective configuration (`OcrEngine::config()`) |
| `PaddleInferenceConfig` | `from_path`, `from_yaml_str` |
| `RecDictionary` | `from_path`, `from_text`, `from_inference_yml(_str)`, `from_tokens`, `with_space_char` |
| `OrientationClassifier` | `from_model_dir`, `from_bytes`, `classify` |

### Enums and constants

| Item | Values |
| :--- | :--- |
| `DetLimitType` | `Max` (long-side cap, default) / `Min` (short-side minimum, as in PaddleOCR 3.x) |
| `RecCropMode` | `Rotated` (default, perspective crop) / `AxisAligned` |
| `ColorOrder` | `Rgb` / `Bgr` |
| `PADDLE_MODEL_FILE` / `PADDLE_CONFIG_FILE` | `"inference.onnx"` / `"inference.yml"` |
| `IMAGENET_MEAN` / `IMAGENET_STD` | Detection normalisation |
| `MULTITHREAD_SUPPORTED` | Whether this build can run inference on several threads |

## 2. API details

### `OcrError`

| Variant | When |
| :--- | :--- |
| `MissingField` | No detection model, recognition model or dictionary was given |
| `Io` | A file is missing or unreadable |
| `ModelLoad` | ONNX parsing or analysis failed (`path` is `<memory>` for in-memory models) |
| `ModelConfig` | `inference.yml` could not be parsed |
| `Dictionary` | Empty dictionary, duplicate entry or missing `character_dict` |
| `OrientationLoad` / `OrientationInference` | An orientation classifier failed to load or run |
| `InvalidConfiguration` | `rec_batch_size == 0`, unsupported post-process name, or invalid `image_shape` |
| `ImageDecode` | The image cannot be decoded |
| `Detection*` / `Recognition*` | A detection or recognition stage failed |
| `PipelineMismatch` | Internal inconsistency between the number of detections and recognitions |

### `OcrResult`

- `text`: the recognised string. Spaces come from the space class appended to the dictionary.
- `confidence`: the mean softmax probability of the decoded characters, from 0 to 1.
- `bounding_box`: the minimum-area rectangle in input-image pixels, given as `tl, tr, br, bl` in a closed ring. It is rounded and clamped like PaddleOCR's `DBPostProcess`, and always uses the input image's coordinates, even after page orientation correction.

### `OcrEngineBuilder`

**Model sources** (between a path and its in-memory counterpart, the last call wins):

| Method | Notes |
| :--- | :--- |
| `det_model_dir` / `rec_model_dir` | PaddleOCR 3.x directory (`inference.onnx` + `inference.yml`); the recognition YAML also provides the dictionary |
| `det_model_path` / `rec_model_path` / `dictionary_path` | Individual files (the dictionary may be text or `.yml`) |
| `det_config_path` / `rec_config_path` | An individual `inference.yml` |
| `det_model_bytes` / `rec_model_bytes` / `det_config_yaml` / `rec_config_yaml` / `dictionary_text` | In-memory inputs |
| `doc_orientation_model_dir` / `_bytes`, `textline_orientation_model_dir` / `_bytes` | Optional classifiers |

**Parameters:**

| Method | Default | Notes |
| :--- | :--- | :--- |
| `det_limit_side_len` | 960 | Detection size limit |
| `det_limit_type` | `Max` | `Min` with 64 reproduces PaddleOCR 3.x (native resolution) |
| `det_max_side_limit` | 4000 | Hard cap on the long side |
| `det_threshold` / `det_box_threshold` / `det_unclip_ratio` | 0.3 / 0.6 / 1.5 | PaddleOCR pipeline defaults |
| `det_postprocess_from_model_config` | false | Use the detection YAML thresholds; explicit setters still win |
| `rec_batch_size` | 1 | Fastest with tract |
| `rec_use_space_char` | true | Append `" "` to the dictionary |
| `rec_crop_mode` | `Rotated` | Crop strategy |
| `inference_threads` | logical CPUs, at most 16 | Always 1 on WebAssembly |
| `plan_cache_capacity(det, rec)` | 4, 16 | Compiled plans kept per model |

### `OcrEngine`

- `run_*` runs synchronously. The engine is `Send + Sync`, so it can be shared in an `Arc`.
- `config()` returns the effective configuration, after the YAML values are applied.
- The path getters return `Option<&Path>`, which is `None` for in-memory inputs.

## 3. Example

```toml
[dependencies]
pure_onnx_ocr = "0.2"
# single-threaded: pure_onnx_ocr = { version = "0.2", default-features = false }
```

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError};

fn main() -> Result<(), OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_dir("models/ppocrv6/small_det")
        .rec_model_dir("models/ppocrv6/small_rec")
        .build()?;
    for result in engine.run_from_path("receipt.jpg")? {
        println!("{} ({:.3})", result.text, result.confidence);
    }
    Ok(())
}
```

For JavaScript, see `bindings/wasm` and `examples/web`.
