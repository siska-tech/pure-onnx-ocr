# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
Before 1.0, minor versions may contain breaking changes.

## [Unreleased]

## [0.3.1] - 2026-10-08

### Added

- Multi-threaded browser build. `pure-onnx-ocr-wasm` gets a `threads`
  feature built from `bindings/wasm/threads` (pinned nightly,
  `-Z build-std`, shared memory with a 2 GiB maximum): inference runs on a
  wasm-bindgen-rayon pool of nested Web Workers, with recognition batches
  in parallel and tract's matrix multiplications on rayon's global pool
  (`Executor::RayonGlobal`). It needs a cross-origin isolated page
  (COOP/COEP headers). Output is identical to the single-threaded build.
  With 4 threads, runs after the first were 2.0-3.3x faster than the
  single-threaded build on PP-OCRv6 tiny / small / medium and PP-OCRv5
  mobile (headless Chromium, 4 vCPUs), and PP-OCRv6 medium peaked at
  1.2 GiB of WebAssembly memory.
  `scripts/build_wasm.sh [--threads]` builds either variant, and CI builds
  both.
- WebAssembly bindings: `initThreadPool(n)` (from wasm-bindgen-rayon; a
  no-op that resolves immediately in the single-threaded build),
  `threadsSupported()`, `OcrEngineBuilder.inferenceThreads(n)` and
  `OcrEngine.inferenceThreads`.
- Web demo: `worker.js` loads the multi-threaded build when the page is
  cross-origin isolated and falls back to the single-threaded one; a thread
  selector; `serve.mjs --coi` serves with COOP/COEP headers; the README
  explains coi-serviceworker for GitHub Pages. `bench.html` / `bench.mjs`
  measure both builds in headless Chrome and check their output against
  the native build (`examples/ocr_json.rs`).

### Changed

- WebAssembly builds with the `atomics` target feature now report
  `MULTITHREAD_SUPPORTED == true` and honour `inference_threads`, capped to
  the size of rayon's global pool (which is also the default). Other
  WebAssembly builds and native builds are unchanged.

## [0.3.0] - 2026-10-08

CPU performance work, measured against OpenVINO Runtime on the same PC, ONNX
files and pre/post-processing (`docs/devlog/perf/`). OCR output is unchanged.

### Added

- `OcrEngine::run_many_from_images` and `run_many_from_paths` process several
  images in one call with the same results as one call per image. Detection
  and the orientation classifiers run on several images at once and the
  recognition batches of all images run together: throughput rose 1.5-2.4x
  on PP-OCRv6 (16 images, 16-thread i7-1360P), at 2.5-3.7x the peak memory
  while it runs. `examples/throughput_bench.rs` measures it.
- `OcrEngine::warmup(width, height)` compiles the detection plan for an
  image size, the recognition plan for text lines of the minimum width and
  the orientation classifiers' plans ahead of the first run.
- `tools/openvino-bench` (OpenVINO comparison, A/B runs between two builds,
  multi-image throughput) and `tools/tract-profile` (per-node timings and an
  output hash for detection and recognition models). Neither is part of the
  published crate.

### Changed

- The default number of inference threads is now the number of logical CPUs
  capped at 16 (was 8). Recognition runs batches in parallel and keeps
  scaling past 8 threads: end-to-end time dropped by 3-18% on a 16-thread
  i7-1360P with identical output. Machines with 8 or fewer logical CPUs are
  unaffected. `inference_threads(8)` restores the old behaviour.

### Fixed

- Threads that need the same not-yet-compiled plan now wait for a single
  compilation instead of each compiling its own copy. On a cold engine,
  most recognition batches share one width, so an image used to compile
  the same plan up to 16 times: the first run is now 10-23% faster and its
  peak memory is back to the 8-thread level (PP-OCRv6 medium 1.6 GB to
  1.0 GB).

## [0.2.1] - 2026-10-03

### Fixed

- Recognition region bounds checks now reject overflowing widths and heights
  with `RegionOutOfBounds` instead of panicking or wrapping integer additions.
- Detection output conversion validates `[1, 1, H, W]` with positive spatial
  dimensions before indexing, returning an error for incompatible model outputs.
- The CTC blank-ID error test now asserts the error variant and field values.

### Added

- Seven regression tests covering invalid regions, image-edge crops, invalid
  detection output shapes and preservation of score-map axes and values.

### Changed

- Expanded API and implementation comments for tensor layouts, CTC decoding,
  coordinate scaling, plan caching, configuration precedence and stage timings.
- Corrected dictionary whitespace and WebAssembly polygon documentation.

## [0.2.0] - 2026-10-03

PaddleOCR PP-OCRv6 support, PaddleOCR 3.x-compatible pre/post-processing,
browser WebAssembly support and multi-threaded inference. Design notes and
measurements are under `docs/devlog/ppocrv6/`, `docs/devlog/wasm/` and
`docs/devlog/perf/`.

### Added
- PP-OCRv6 tiny / small / medium detection and recognition models (ONNX
  exports from Hugging Face), and PaddleOCR 3.x model directories in general:
  `OcrEngineBuilder::det_model_dir` / `rec_model_dir` read `inference.onnx`
  plus `inference.yml` (preprocessing parameters and the embedded dictionary).
- `PaddleInferenceConfig`: dependency-free reader for PaddleOCR
  `inference.yml`; `RecDictionary::from_inference_yml`, `from_tokens`,
  `with_space_char`.
- Rotation-corrected text crops (minimum-area rectangle + perspective warp,
  vertical regions rotated by 90°): `RecCropMode`, `min_area_quad`,
  `crop_quad`.
- Document orientation (0/90/180/270) and text-line orientation (0/180)
  classifiers: `OrientationClassifier`,
  `OcrEngineBuilder::doc_orientation_model_dir` /
  `textline_orientation_model_dir`, `OcrRunWithMetrics::doc_orientation_angle`,
  `OcrTimings::orientation`.
- Detection options: `DetLimitType` (`Max` / `Min`, native-resolution mode as
  in PaddleOCR 3.x), `det_max_side_limit`, `det_threshold`,
  `det_box_threshold`, `det_postprocess_from_model_config`.
- In-memory inputs for environments without a file system:
  `det_model_bytes`, `rec_model_bytes`, `det_config_yaml`, `rec_config_yaml`,
  `dictionary_text`, `*_orientation_model_bytes`,
  `OcrEngine::run_from_bytes` / `run_with_metrics_from_bytes`, and the
  matching `from_bytes` constructors on the sessions.
- Browser WebAssembly support: wasm-bindgen bindings (`bindings/wasm`,
  crate `pure-onnx-ocr-wasm`), WebAssembly SIMD enabled by default through
  `.cargo/config.toml`, and a Web Worker demo in `examples/web`.
- Multi-threaded inference (default feature `multithread`):
  `OcrEngineBuilder::inference_threads`, `default_inference_threads`,
  `MULTITHREAD_SUPPORTED`; recognition batches run in parallel.
- `OcrEngineBuilder::plan_cache_capacity` to bound compiled-plan memory.
- `ocr_smoke` options: `--det-model-dir`, `--rec-model-dir`, `--det-thresh`,
  `--det-box-thresh`, `--det-limit-type`, `--det-max-side-limit`,
  `--det-params-from-config`, `--crop-mode`, `--doc-ori-model-dir`,
  `--textline-ori-model-dir`, `--threads`, `--no-space-char`, `-v/--verbose`.
- `examples/ocr_bench` (PP-OCRv5 vs PP-OCRv6 benchmark),
  `scripts/fetch_fixtures.sh`, GitHub Actions CI.
- PaddleOCR parity test (`tests/paddle_parity.rs`) against PaddleOCR 3.7
  outputs in `tests/reference/`, generated by `scripts/paddleocr_reference.py`;
  `imgproc::resize_bilinear` (OpenCV `INTER_LINEAR` clone) and
  `DetPostProcessor::db_boxes`.

### Changed
- Preprocessing and postprocessing now follow PaddleOCR 3.x, which also
  improves PP-OCRv5 results: BGR input with ImageNet normalisation for
  detection, `box_thresh` filtering (0.6) and outer contours only,
  variable-width recognition input (up to 3200 px, previously squeezed to
  320 px), zero padding after normalisation, aspect-ratio-sorted batches.
- The space class is appended to the dictionary by default
  (`rec_use_space_char(false)` restores the old behaviour); spaces were
  previously decoded as `[UNK]`.
- Default recognition batch size is 1 (fastest with tract), and inference
  uses up to 8 threads by default.
- Library logging goes through the `log` crate instead of `println!`.
- Detection boxes are computed like PaddleOCR `DBPostProcess` (minimum-area
  rectangle, rectangle unclip, size filters) and `OcrResult::bounding_box` is
  now the 4-point rectangle; detection input is stretched to the nearest
  multiple of 32 instead of padded; crops use bicubic warps and OpenCV-style
  bilinear resizing. PP-OCRv6 medium now matches PaddleOCR exactly on the test
  images.
- `tract-onnx` 0.20 → 0.23 (required for PP-OCRv6 medium; also ~1.5–2× faster)
  and `ndarray` 0.15 → 0.17.

### Breaking
- MSRV raised from 1.70 to 1.91 (required by tract 0.23).
- `OcrEngine::det_model_path` / `rec_model_path` / `dictionary_path` return
  `Option<&Path>` (`None` for in-memory inputs).
- New public fields on `DetPreProcessorConfig`, `DetPostProcessorConfig`,
  `RecPreProcessorConfig`, `OcrEngineConfig`, `OcrTimings` and
  `OcrRunWithMetrics`; struct literals need `..Default::default()`.
- `RecPreProcessorConfig::pad_value` defaults to 0.5 (normalised zero).
- `OcrEngineConfig::det_polygon_scaler` was removed (boxes are rescaled the
  PaddleOCR way); `OcrResult::bounding_box` has 4 corners instead of the
  unclipped contour.
- Default outputs differ from 0.1.0 because of the PaddleOCR 3.x-compatible
  processing above.

### Fixed
- crates.io metadata: `categories` now uses valid slugs (`computer-vision`,
  `science`); the previous values were rejected and had to be removed for the
  0.1.0 upload.
- PaddleOCR 3.x ONNX exports (PP-OCRv5 and PP-OCRv6) failed to load because
  their symbolic `value_info` shape hints conflicted with concrete inputs.
- `OcrEngine` is now `Send + Sync`; the documentation claimed it could be
  shared across threads, but `RefCell` plan caches made it neither.
- `image_decode_seconds` reported the whole pipeline time.
- The PP-OCRv5 integration tests silently skipped because they expected a
  `demo.png` that nothing provided.
- `std::time::Instant` panics on `wasm32-unknown-unknown`.

## [0.1.0] - 2025-11-11

- Initial release: Pure Rust DBNet detection + SVTR/CTC recognition
  pipeline for PaddleOCR PP-OCRv5 ONNX models on `tract-onnx`, the
  `OcrEngineBuilder` / `OcrEngine` API and the `ocr_smoke` CLI.

[Unreleased]: https://github.com/siska-tech/pure-onnx-ocr/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/siska-tech/pure-onnx-ocr/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/siska-tech/pure-onnx-ocr/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/siska-tech/pure-onnx-ocr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/siska-tech/pure-onnx-ocr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/siska-tech/pure-onnx-ocr/releases/tag/v0.1.0
