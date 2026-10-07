# pure-onnx-ocr vs OpenVINO: multi-image throughput (measured)

- 2026-10-07 22:18:54 / 13th Gen Intel(R) Core(TM) i7-1360P / Windows-11-10.0.26200-SP0
- OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4 / rustc 1.99.0 (b940084d7 2026-09-28) / pure-onnx-ocr 442d13a
- `pure-seq-tfix` runs `C:\Users\Shion\Documents\Projects\pure-onnx-ocr-tractmain\tools\openvino-bench\target\release\openvino-bench.exe`
- `pure-many-tfix` runs `C:\Users\Shion\Documents\Projects\pure-onnx-ocr-tractmain\tools\openvino-bench\target\release\openvino-bench.exe`
- images general_ocr_002.jpg, ja.jpg repeated: 16 images per pass; 2 timed passes per process after one warm-up pass; 3 rounds (medians over rounds)

## Throughput (images/s, median over rounds)

| Model | Config | img/s | min-max | vs pure-seq | vs best OV | avg cores busy | threads | peak WS MB | peak private MB | private after MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6-tiny | pure-seq | 2.20 | 2.10-2.52 | 1.00 | 0.27 | 4.3 | 36 | 187 | 181 | 104 |
| v6-tiny | pure-many | 5.27 | 5.06-5.32 | 2.40 | 0.64 | 14.4 | 20 | 597 | 664 | 101 |
| v6-tiny | ov-seq | 4.90 | 3.74-5.63 | 2.23 | 0.59 | 7.2 | 58 | 369 | 589 | 519 |
| v6-tiny | ov-many | 6.53 | 5.44-7.49 | 2.97 | 0.79 | 10.9 | 45 | 1517 | 1765 | 821 |
| v6-tiny | pure-seq-tfix | 3.45 | 3.02-3.69 | 1.57 | 0.42 | 6.5 | 36 | 179 | 171 | 95 |
| v6-tiny | pure-many-tfix | 6.52 | 6.30-6.68 | 2.97 | 0.79 | 14.2 | 20 | 476 | 524 | 95 |
| v6-tiny | ov-many-d4r8 | 7.45 | 6.85-8.26 | 3.39 | 0.90 | 11.4 | 49 | 1589 | 1846 | 909 |
| v6-tiny | ov-many-d8r16 | 8.26 | 8.04-8.83 | 3.76 | 1.00 | 12.0 | 55 | 2076 | 2378 | 1436 |
| v6-small | pure-seq | 0.92 | 0.79-0.94 | 1.00 | 0.41 | 6.6 | 33 | 443 | 439 | 253 |
| v6-small | pure-many | 1.51 | 1.50-1.54 | 1.64 | 0.67 | 14.7 | 17 | 985 | 1106 | 249 |
| v6-small | ov-seq | 1.62 | 1.43-1.80 | 1.75 | 0.72 | 9.1 | 58 | 705 | 1035 | 865 |
| v6-small | ov-many | 1.89 | 1.69-2.08 | 2.05 | 0.84 | 11.7 | 45 | 3168 | 3559 | 1307 |
| v6-small | pure-seq-tfix | 1.18 | 1.08-1.30 | 1.28 | 0.53 | 8.5 | 33 | 445 | 440 | 244 |
| v6-small | pure-many-tfix | 1.82 | 1.80-1.84 | 1.98 | 0.81 | 14.4 | 20 | 798 | 867 | 245 |
| v6-small | ov-many-d4r8 | 2.12 | 1.91-2.24 | 2.30 | 0.94 | 12.6 | 49 | 3340 | 3804 | 1574 |
| v6-small | ov-many-d8r16 | 2.25 | 2.06-2.29 | 2.44 | 1.00 | 13.4 | 54 | 4213 | 4845 | 2609 |
| v6-medium | pure-seq | 0.23 | 0.22-0.23 | 1.00 | 0.55 | 7.9 | 33 | 1078 | 1072 | 866 |
| v6-medium | pure-many | 0.34 | 0.34-0.35 | 1.48 | 0.81 | 14.9 | 17 | 2939 | 3227 | 873 |
| v6-medium | ov-seq | 0.40 | 0.40-0.41 | 1.74 | 0.96 | 11.8 | 55 | 1097 | 1808 | 1575 |
| v6-medium | ov-many | 0.41 | 0.39-0.41 | 1.77 | 0.97 | 12.9 | 42 | 3926 | 4692 | 2423 |
| v6-medium | pure-seq-tfix | 0.34 | 0.34-0.34 | 1.47 | 0.81 | 11.0 | 33 | 1073 | 1069 | 870 |
| v6-medium | pure-many-tfix | 0.42 | 0.42-0.42 | 1.80 | 0.99 | 15.1 | 17 | 2826 | 3256 | 869 |
| v6-medium | ov-many-d4r8 | 0.42 | 0.38-0.42 | 1.82 | 1.00 | 13.6 | 46 | 4114 | 4970 | 2730 |
| v6-medium | ov-many-d8r16 | 0.42 | 0.39-0.43 | 1.81 | 1.00 | 14.2 | 52 | 5361 | 6310 | 4016 |

Best OV per model (fastest OpenVINO config measured): v6-tiny: `ov-many-d8r16`, v6-small: `ov-many-d8r16`, v6-medium: `ov-many-d4r8`

## Output parity (relative to `pure-seq`)

| Model | Image | Config | regions (ref / config) | identical text | identical box |
|---|---|---|---:|---:|---:|
| v6-tiny | general_ocr_002.jpg | pure-many | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | pure-many | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | pure-many | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | pure-many | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | pure-many | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | pure-many | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | ov-seq | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | ov-seq | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | ov-seq | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | ov-seq | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | ov-seq | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | ov-seq | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | ov-many | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | ov-many | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | ov-many | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | ov-many | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | ov-many | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | ov-many | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | pure-seq-tfix | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | pure-seq-tfix | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | pure-seq-tfix | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | pure-seq-tfix | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | pure-seq-tfix | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | pure-seq-tfix | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | pure-many-tfix | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | pure-many-tfix | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | pure-many-tfix | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | pure-many-tfix | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | pure-many-tfix | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | pure-many-tfix | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | ov-many-d4r8 | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | ov-many-d4r8 | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | ov-many-d4r8 | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | ov-many-d4r8 | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | ov-many-d4r8 | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | ov-many-d4r8 | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | ov-many-d8r16 | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | ov-many-d8r16 | 55 / 55 | 55 | 55 |
| v6-small | general_ocr_002.jpg | ov-many-d8r16 | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | ov-many-d8r16 | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | ov-many-d8r16 | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | ov-many-d8r16 | 55 / 55 | 55 | 55 |

## Configs

- `pure-seq`: `--backend pure --throughput 8`
- `pure-many`: `--backend pure --many --throughput 8`
- `ov-seq`: `--backend ov --ov-rec-hint THROUGHPUT --ov-rec-requests 0 --throughput 8`
- `ov-many`: `--backend ov --many --ov-det-hint THROUGHPUT --ov-det-requests 0 --ov-rec-hint THROUGHPUT --ov-rec-requests 0 --throughput 8`
- `pure-seq-tfix`: `--backend pure --throughput 8`
- `pure-many-tfix`: `--backend pure --many --throughput 8`
- `ov-many-d4r8`: `--backend ov --many --ov-det-hint THROUGHPUT --ov-det-requests 0 --ov-rec-hint THROUGHPUT --ov-rec-requests 0 --ov-det-streams 4 --ov-rec-streams 8 --throughput 8`
- `ov-many-d8r16`: `--backend ov --many --ov-det-hint THROUGHPUT --ov-det-requests 0 --ov-rec-hint THROUGHPUT --ov-rec-requests 0 --ov-det-streams 8 --ov-rec-streams 16 --throughput 8`

## Settings

- v6-tiny / pure-seq: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-tiny / pure-many: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-tiny / ov-seq: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "NO", "INFERENCE_NUM_THREADS": "12", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "1", "PERFORMANCE_HINT": "LATENCY", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 1, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-tiny / ov-many: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-tiny / pure-seq-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-tiny / pure-many-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-tiny / ov-many-d4r8: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 8}}`
- v6-tiny / ov-many-d8r16: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 8, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "16", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "16", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 16}}`
- v6-small / pure-seq: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-small / pure-many: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-small / ov-seq: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "NO", "INFERENCE_NUM_THREADS": "12", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "1", "PERFORMANCE_HINT": "LATENCY", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 1, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-small / ov-many: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-small / pure-seq-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-small / pure-many-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-small / ov-many-d4r8: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 8}}`
- v6-small / ov-many-d8r16: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 8, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "16", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "16", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 16}}`
- v6-medium / pure-seq: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-medium / pure-many: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-medium / ov-seq: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "NO", "INFERENCE_NUM_THREADS": "12", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "1", "PERFORMANCE_HINT": "LATENCY", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 1, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-medium / ov-many: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 4}}`
- v6-medium / pure-seq-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-medium / pure-many-tfix: `{"pure": {"default_inference_threads": 16, "inference_threads": 16, "multithread_supported": true}}`
- v6-medium / ov-many-d4r8: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "4", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "4", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 4, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 8}}`
- v6-medium / ov-many-d8r16: `{"openvino": {"det": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "8", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "8", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "det_requests": 8, "preprocess_threads": 16, "rec": {"ENABLE_CPU_PINNING": "NO", "ENABLE_HYPER_THREADING": "YES", "INFERENCE_NUM_THREADS": "16", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "16", "OPTIMAL_NUMBER_OF_INFER_REQUESTS": "16", "PERFORMANCE_HINT": "THROUGHPUT", "SCHEDULING_CORE_TYPE": "ANY_CORE"}, "rec_requests": 16}}`
