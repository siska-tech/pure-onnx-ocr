# Detailed Design: Pure Rust OnnxOCR

Author: Shion Watanabe  
First version: 2025-11-09  
Revised: 2026-10-03 (v0.2.0)  
Repository: http://github.com/siska-tech/pure-onnx-ocr

## Purpose

Describe the modules and algorithms as implemented, together with the PaddleOCR 3.x (PaddleX) step that each one mirrors. Parity is checked by `tests/paddle_parity.rs`.

## 1. Internal modules

| Module | Main items | Responsibility |
| :--- | :--- | :--- |
| `engine` | `OcrEngineBuilder`, `OcrEngine`, private `DetectionPipeline` / `RecognitionPipeline` | Configuration and orchestration |
| `onnx_model` (private) | `load_paddle_onnx(_from_bytes)`, `PlanCache`, `lock_cache` | ONNX loading with `value_info` removal; LRU cache of compiled plans per input shape |
| `detection` / `recognition` | `DetInferenceSession`, `RecInferenceSession`, `RecPostProcessor` | DBNet `[1,3,H,W]` and CTC recognition `[N,3,48,W]` inference |
| `preprocessing` | `DetPreProcessor`, `RecPreProcessor` | Resizing, normalisation, tensors |
| `postprocessing` | `DetPostProcessor::db_boxes`, `DetPolygonUnclipper` | DB post-processing |
| `crop` | `min_area_quad`, `crop_quad` | Rotated rectangles and perspective crops |
| `imgproc` | `resize_bilinear` | `cv2.resize(INTER_LINEAR)` clone |
| `paddle_config` | `PaddleInferenceConfig`, minimal YAML parser | `inference.yml` |
| `dictionary` / `ctc` | `RecDictionary`, `CtcGreedyDecoder` | blank + characters + space; greedy decoding and confidence |
| `orientation` | `OrientationClassifier`, `rotate_ccw`, `unrotate_point` | PP-LCNet classifiers |
| `threading` / `time` (private) | `executor_for`, `run_with`, `parallel_map`, `Instant` | Thread pool and tract executor; browser-safe clock |

## 2. Data structures

| Type | Contents |
| :--- | :--- |
| `PreprocessedDetInput` | `tensor`, `resized_dims`, `scale_ratio`, `scale_xy` (per-axis scale) |
| `DetBox` | `quad` (tl, tr, br, bl), `score` |
| `PreprocessedRecBatch` | `tensor [N,3,48,W]`, `valid_widths`, `max_width` |
| `RecInferenceOutput` | `logits [N,T,C]`, `valid_timesteps` |
| `DecodedSequence` | `text`, `token_indices`, `confidence`, `fallback_count` |

## 3. Algorithms

### 3.1 `OcrEngineBuilder::build`

1. Fail with `MissingField` when a model or the dictionary is missing. The dictionary source is searched in this order: `dictionary_text`, `dictionary_path`, `rec_config_yaml`, `rec_config_path`.
2. Parse the `inference.yml` files. Apply detection colour order and mean/std, and recognition colour order and `image_shape`. Apply the YAML thresholds only when `det_postprocess_from_model_config` is set.
3. Load the ONNX models. `load_paddle_onnx` clears the facts of every non-input, non-constant outlet: PaddleOCR 3.x exports carry `DynamicDimension.*` `value_info` hints that otherwise conflict with concrete input shapes.
4. Keep a base model with symbolic batch and width. Plans are compiled per concrete shape and kept in LRU caches (4 for detection, 16 for recognition).
5. Build the dictionary (appending the space class by default) and the thread pool.

### 3.2 `OcrEngine::run_from_image`

1. Optional page orientation: classify, then rotate the page upright with `rotate_ccw`.
2. Detection → boxes in input coordinates.
3. Crop with `crop_quad` (falling back to the axis-aligned box).
4. Optional text-line flip. The last chunk is padded to the batch size so a single compiled plan is reused.
5. Recognition.
6. When the page was rotated, map the boxes back with `unrotate_point`.

### 3.3 Detection preprocessing (PaddleX `DetResizeForTest`)

1. Compute the scale: for `Max`, `limit / long side` if the long side exceeds the limit; for `Min`, `limit / short side` if the short side is below it.
2. `resize = int(side × ratio)`, then cap the long side at `max_side_limit`.
3. Round each side to `max(round_half_even(side / 32) × 32, 32)`.
4. **Stretch** the image to that size with `resize_bilinear`.
5. Normalise in BGR order with `(x/255 − mean) / std` (ImageNet values). Record `scale_xy`.

### 3.4 Detection post-processing (`db_boxes`, PaddleX `DBPostProcess`)

1. Binarise with `probability > thresh`.
2. Extract all contours (`RETR_LIST`), at most `max_candidates`.
3. For each contour, take the minimum-area rectangle (convex hull plus rotating calipers). Drop it if the short side is < 3.
4. Score the rectangle with the mean probability inside it (`box_score_fast`). Drop it if the score is < `box_thresh`.
5. Unclip the **rectangle** (distance = area × ratio / perimeter, round joins via `i_overlay`). Take the minimum-area rectangle of the result and drop it if its short side is < 5.
6. Scale each point by `inverse_scale`, round, and clamp to the image.

### 3.5 Crop (`crop_quad`, PaddleX `get_rotate_crop_image`)

- Width = `int(max(|tl−tr|, |bl−br|))`, height = `int(max(|tl−bl|, |tr−br|))`.
- Perspective-warp with bicubic interpolation. If `h/w ≥ 1.5`, rotate 90° counter-clockwise (`np.rot90`).

### 3.6 Recognition preprocessing (PaddleX `OCRReisizeNormImg`)

1. Crop width: `int(48r)` when `r > 320/48`, else `ceil(48r)`, where `r` = width / height.
2. Batch width: `max(320, widest)`, capped at 3200 and rounded up to a multiple of 32. The rounding is for plan reuse and was verified not to change results.
3. Resize to height 48 with `resize_bilinear`, normalise BGR with `(x/255 − 0.5)/0.5`, and pad with 0.
4. Crops are sorted by aspect ratio and chunked by `rec_batch_size`. Preprocess, inference and decode each run in parallel with `parallel_map`, and the parallel inference uses `run_single_threaded`.

### 3.7 CTC decoding

1. Take the arg-max class at each time step within the valid width.
2. Drop blanks (index 0) and consecutive repeats, and map indices through the dictionary. Out-of-range indices become `[UNK]`.
3. Confidence is the mean probability of the kept characters. When the output is logits, softmax is computed with log-sum-exp.

### 3.8 Orientation classifiers

- Preprocessing follows `inference.yml`: `ResizeImage.size`, or `resize_short` followed by `CropImage`.
- Input is RGB with ImageNet normalisation. The arg-max label is parsed into an angle.
- Page correction uses `rotate_ccw(angle)`, like PaddleX `rotate_image`.

## 4. Error handling

- Public APIs return `Result<_, OcrError>` and do not panic. If the thread pool cannot be created, the panic is caught and inference falls back to a single thread with a warning.
- Errors from in-memory inputs report the path as `<memory>`.
- Unsupported YAML constructs (flow collections, block scalars) are reported as `PaddleConfigError::Syntax` with the line number.
- Poisoned plan-cache locks are recovered, since the cache cannot be left inconsistent.
