# Test Specification: Pure Rust OnnxOCR

Author: Shion Watanabe  
First version: 2025-11-09  
Revised: 2026-10-03 (v0.2.0)  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

- Define the test strategy and the test cases that actually exist.
- Continuously check output parity with PaddleOCR (Python) through automated tests.
- Verify Linux, Windows, the MSRV and WebAssembly builds in CI.

## 1. Strategy

- **Kinds of tests**
  1. Unit tests (`src/**`): stages, YAML parsing, dictionary, CTC, rectangles and crops, resampling, plan cache, orientation helpers.
  2. Model-based integration tests (`tests/*.rs`): end to end with real ONNX models and images.
  3. PaddleOCR parity (`tests/paddle_parity.rs`): comparison with PaddleOCR 3.7 outputs stored in `tests/reference/*.json`.
  4. CLI test (`tests/ocr_smoke.rs`).
  5. Build checks in CI: MSRV 1.91, `--no-default-features`, `wasm32-unknown-unknown`, `wasm32-wasip1`.
- **Missing fixtures**: tests that need missing models are skipped with a message. Slow tests (small/medium, tilt) are `#[ignore]`.
- **Harness**: the standard Rust harness. `cargo test --release` is recommended.

## 2. Environment

- **OS**: Linux (ubuntu-latest) and Windows (windows-latest; development on Windows 11).
- **Rust**: stable, with MSRV 1.91.
- **Fixtures** (`scripts/fetch_fixtures.sh`; `tests/fixtures/` is git-ignored):
  - Default set (about 35 MB): PP-OCRv6 tiny det/rec, PP-OCRv5 mobile det/rec/dictionary (legacy single-file layout), the doc-orientation classifier, the `x0_25` text-line classifier, and `images/general_ocr_002.jpg`.
  - `--all` adds PP-OCRv6 small/medium, PP-OCRv5 mobile/server in directory form, the `x1_0` text-line classifier, and `images/ja.jpg`.
- **PaddleOCR references**: run `scripts/paddleocr_reference.py` in a uv `.venv` built from `scripts/requirements-reference.txt`. The generated JSON files are committed.

## 3. Test cases

### 3.1 Proof of concept

| Test | What | Expected |
| :--- | :--- | :--- |
| `tests::dbnet_dummy_inference_runs_successfully` / `svtr_…` (ignored) | Zero-input inference on `models/ppocrv5/*.onnx` | Outputs are produced and finite |
| `detection::tests::detection_inference_runs` / `recognition::tests::recognition_inference_runs` | Real-shaped inputs | Output shapes are consistent with the inputs |

### 3.2 Unit tests

| Area | Tests | Checks |
| :--- | :--- | :--- |
| Detection preprocessing | `resize_long_side_to_limit`, `keep_original_size_when_within_limit`, `detection_dims_round_to_nearest_multiple_of_32`, `min_limit_*`, `max_side_limit_caps_native_resolution`, `tensor_shape_and_normalization`, `detection_uses_bgr_channel_order_by_default` | Rounding to multiples of 32 (ties to even), stretching, per-axis scale, ImageNet normalisation, BGR order |
| Recognition preprocessing | `recognition_*` (6 tests) | Variable width (1200 px → 1216), cap, padding value, error cases |
| Detection post-processing | contour, box threshold, max candidates, unclip and scaler tests | Candidate extraction and filtering |
| Crop | `crop::tests::*` (5 tests) | Corner order, rotated rectangle recovery, degenerate input, warp, vertical rotation |
| Resampling | `imgproc::tests::*` (2 tests) | Same values as OpenCV bilinear |
| YAML / dictionary | `paddle_config::tests::*` (7 tests), `dictionary::*` (10 tests) | Quoting, U+3000, nesting, anchors, classifier fields, space class, duplicates |
| CTC | `ctc::tests::*` (5 tests) | Blank and repeat removal, confidence, out-of-range classes |
| Other | `onnx_model::tests::*`, `orientation::tests::*`, `engine::thread_safety::*` | LRU cache, angles, `Send + Sync` |

### 3.3 Integration tests

| Test | Condition | Expected |
| :--- | :--- | :--- |
| `engine::tests::*` | PP-OCRv5 mobile, individual files | Build succeeds, error cases, blank image handled, timings reported |
| `integration_test::*` | PP-OCRv5 mobile, boarding pass | Result contains `BOARDING`; missing image or models produce errors |
| `ppocrv6::detection_configs_*` / `recognition_configs_*` | PP-OCRv6 YAML files | BGR, ImageNet values, dictionary sizes (6,904 / 18,708) |
| `ppocrv6::recognition_class_count_matches_dictionary` | tiny_rec | Number of classes = blank + dictionary + space |
| `ppocrv6::tiny_pipeline_reads_boarding_pass` (small/medium ignored) | Boarding pass | Key strings and the footer line with spaces are read |
| `ppocrv6::model_config_postprocess_values_respect_explicit_overrides` | tiny YAML | Precedence: default < YAML < explicit setter |
| `ppocrv6::rotated_crops_read_tilted_text` (ignored) | Image tilted by 10° | Rotated crops reproduce more upright lines than axis-aligned crops |
| `ppocrv6::doc_orientation_restores_rotated_pages` | Pages rotated by 90/180/270° | Correct angle, and the text is read |
| `ppocrv6::textline_orientation_fixes_upside_down_lines` | Upside-down page | Key strings are read with the classifier |
| `ppocrv6::in_memory_models_match_file_based_engine` | Inputs given as bytes | Same output as file inputs; path getters return `None` |
| `ppocrv6::engine_can_be_shared_between_threads` / `thread_count_does_not_change_results` | 3 concurrent threads / 1 vs 4 threads | Identical outputs |
| `ppocrv6::ppocrv5_yaml_dictionary_matches_text_dictionary` | PP-OCRv5 YAML vs text dictionary | Same 18,383 entries in the same order |
| `paddle_parity::matches_paddleocr_reference_outputs` | Every (model, image) pair with fixtures | Detection F1 ≥ 0.90, character similarity ≥ 0.93 (text not checked for v6 tiny on Japanese) |
| `ocr_smoke::ocr_smoke_help_succeeds` | `--help` | Exit code 0 and usage text |

### 3.4 How to run

```bash
scripts/fetch_fixtures.sh
cargo test --release --workspace
cargo test --release --test ppocrv6 -- --ignored
cargo test --release --test paddle_parity -- --nocapture   # prints the parity table
```
