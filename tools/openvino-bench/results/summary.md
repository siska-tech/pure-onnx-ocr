# pure-onnx-ocr vs OpenVINO (measured)

- 2026-10-07 09:42:24 / 13th Gen Intel(R) Core(TM) i7-1360P / Windows-11-10.0.26200-SP0
- OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4 / rustc 1.99.0 (b940084d7 2026-09-28) / pure-onnx-ocr de05070
- rounds 5 x warm runs 5 per process (warm medians over 25 samples)

## Headline: end-to-end warm latency (median), OpenVINO = best config (`ov`)

| Model | Image | pure-onnx-ocr | OpenVINO | Ratio (pure / OV) |
|---|---|---:|---:|---:|
| v6-small | general_ocr_002.jpg | 1.37 s | 0.58 s | 2.35 |
| v6-small | ja.jpg | 1.54 s | 0.79 s | 1.94 |
| v6-medium | general_ocr_002.jpg | 4.99 s | 2.47 s | 2.02 |
| v6-medium | ja.jpg | 5.80 s | 3.23 s | 1.80 |
| v6-tiny | general_ocr_002.jpg | 0.49 s | 0.18 s | 2.70 |
| v6-tiny | ja.jpg | 0.55 s | 0.25 s | 2.18 |

| Model | pure-onnx-ocr (mean of images) | OpenVINO | Ratio | OpenVINO `ov-latency` | Ratio |
|---|---:|---:|---:|---:|---:|
| v6-small | 1.45 s | 0.69 s | 2.11 | 1.05 s | 1.39 |
| v6-medium | 5.39 s | 2.85 s | 1.89 | 3.91 s | 1.38 |
| v6-tiny | 0.52 s | 0.22 s | 2.40 | 0.35 s | 1.49 |

## Stage breakdown (warm median, ms)

| Model | Image | Config | regions | first run | total | det pre | det inf | det post | rec pre (+crop) | rec inf | rec post |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6-small | general_ocr_002.jpg | pure | 31 | 2260 | 1373 | 15.2 | 557.3 | 4.5 | 9.8 | 742.7 | 30.2 |
| v6-small | general_ocr_002.jpg | ov | 31 | 667 | 585 | 16.8 | 88.1 | 5.7 | 11.8 | 399.0 | 39.8 |
| v6-small | general_ocr_002.jpg | ov-latency | 31 | 992 | 834 | 14.7 | 78.4 | 5.2 | 11.1 | 685.4 | 35.0 |
| v6-small | ja.jpg | pure | 55 | 1995 | 1535 | 24.7 | 570.5 | 5.7 | 29.1 | 858.0 | 39.9 |
| v6-small | ja.jpg | ov | 55 | 858 | 791 | 33.3 | 91.5 | 7.7 | 35.2 | 581.9 | 48.5 |
| v6-small | ja.jpg | ov-latency | 55 | 1378 | 1260 | 28.3 | 83.4 | 6.8 | 33.0 | 1069.5 | 38.7 |
| v6-medium | general_ocr_002.jpg | pure | 33 | 6898 | 4986 | 13.2 | 2084.2 | 4.3 | 8.2 | 2853.1 | 28.9 |
| v6-medium | general_ocr_002.jpg | ov | 33 | 2566 | 2466 | 15.6 | 431.6 | 5.8 | 11.7 | 1883.3 | 38.5 |
| v6-medium | general_ocr_002.jpg | ov-latency | 33 | 3629 | 3140 | 15.1 | 421.2 | 5.7 | 11.4 | 2651.8 | 38.7 |
| v6-medium | ja.jpg | pure | 55 | 6350 | 5799 | 23.3 | 2248.9 | 5.7 | 25.8 | 3415.3 | 35.9 |
| v6-medium | ja.jpg | ov | 55 | 3253 | 3227 | 29.3 | 460.6 | 7.7 | 35.4 | 2633.0 | 44.5 |
| v6-medium | ja.jpg | ov-latency | 55 | 4554 | 4677 | 29.4 | 448.5 | 7.5 | 34.8 | 4058.2 | 41.9 |
| v6-tiny | general_ocr_002.jpg | pure | 34 | 1017 | 486 | 15.2 | 299.2 | 4.4 | 9.4 | 142.6 | 13.0 |
| v6-tiny | general_ocr_002.jpg | ov | 34 | 266 | 180 | 15.8 | 41.1 | 5.5 | 11.2 | 88.7 | 14.3 |
| v6-tiny | general_ocr_002.jpg | ov-latency | 34 | 433 | 280 | 15.2 | 41.1 | 5.3 | 10.9 | 194.5 | 13.4 |
| v6-tiny | ja.jpg | pure | 55 | 860 | 549 | 27.2 | 312.2 | 5.8 | 27.8 | 154.8 | 14.4 |
| v6-tiny | ja.jpg | ov | 55 | 292 | 252 | 29.6 | 43.8 | 7.1 | 33.6 | 122.4 | 14.7 |
| v6-tiny | ja.jpg | ov-latency | 55 | 441 | 414 | 29.0 | 42.6 | 7.0 | 32.0 | 284.3 | 13.8 |

## Load, cold start, resources

| Model | Config | load ms | first run ms (img1) | load+first ms | throughput img/s | avg cores busy | threads (after run) | peak WS MB | peak private MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v6-small | pure | 94 | 2260 | 2354 | 0.70 | 4.3 | 28 | 412 | 404 |
| v6-small | ov | 747 | 667 | 1414 | 1.45 | 8.5 | 50 | 648 | 974 |
| v6-small | ov-latency | 629 | 992 | 1621 | 0.97 | 5.9 | 47 | 519 | 798 |
| v6-medium | pure | 245 | 6898 | 7143 | 0.19 | 4.8 | 25 | 1040 | 1033 |
| v6-medium | ov | 863 | 2566 | 3430 | 0.34 | 10.4 | 47 | 1046 | 1781 |
| v6-medium | ov-latency | 843 | 3629 | 4473 | 0.25 | 6.7 | 47 | 911 | 1664 |
| v6-tiny | pure | 40 | 1017 | 1057 | 1.94 | 3.2 | 28 | 160 | 151 |
| v6-tiny | ov | 448 | 266 | 714 | 4.45 | 7.1 | 50 | 340 | 559 |
| v6-tiny | ov-latency | 449 | 433 | 882 | 2.83 | 6.3 | 45 | 291 | 518 |

## A. Model only (identical tensors, median ms)

| Model | det pure | det OV | ratio | rec 8x320 pure | rec 8x320 OV | ratio | rec 1x320 pure | rec 1x320 OV | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6-small | 525.2 | 88.2 | 5.96 | 650.4 | 115.4 | 5.64 | 42.0 | 20.7 | 2.03 |
| v6-medium | 2166.7 | 431.9 | 5.02 | 1774.3 | 483.3 | 3.67 | 125.0 | 76.0 | 1.65 |
| v6-tiny | 275.2 | 33.9 | 8.12 | 186.0 | 25.2 | 7.38 | 12.2 | 4.3 | 2.84 |

Shapes: det [1, 3, 512, 896], rec [8, 3, 48, 320] / [1, 3, 48, 320]. pure: crate default threads; OV: LATENCY hint, 1 request, automatic threads.

## A'. Model only, 1 thread each (kernel efficiency, median ms)

| Model | det pure | det OV | ratio | rec 8x320 pure | rec 8x320 OV | ratio | rec 1x320 pure | rec 1x320 OV | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6-small | 675.0 | 173.8 | 3.88 | 811.0 | 247.9 | 3.27 | 56.9 | 31.3 | 1.82 |
| v6-medium | 2805.0 | 1095.7 | 2.56 | 2484.9 | 1222.2 | 2.03 | 209.4 | 149.3 | 1.40 |
| v6-tiny | 338.3 | 70.6 | 4.79 | 209.2 | 48.7 | 4.30 | 10.5 | 6.5 | 1.61 |

## Output parity (first run, pure vs ov)

| Model | Image | regions (pure / ov) | identical text | identical box |
|---|---|---:|---:|---:|
| v6-small | general_ocr_002.jpg | 31 / 31 | 31 | 31 |
| v6-small | ja.jpg | 55 / 55 | 55 | 55 |
| v6-medium | general_ocr_002.jpg | 33 / 33 | 33 | 33 |
| v6-medium | ja.jpg | 55 / 55 | 55 | 55 |
| v6-tiny | general_ocr_002.jpg | 34 / 34 | 34 | 34 |
| v6-tiny | ja.jpg | 55 / 55 | 55 | 55 |

## Models (same ONNX file for both backends)

| Model | bytes | sha256 |
|---|---:|---|
| v6-small/det | 9880512 | `d73e0058b7a8086b…` |
| v6-small/rec | 21159378 | `5435fd747c9e0efe…` |
| v6-medium/det | 62032837 | `eb13b44b25bb36f8…` |
| v6-medium/rec | 76554979 | `9c09abf0957f7968…` |
| v6-tiny/det | 1780590 | `193bab7a04fca699…` |
| v6-tiny/rec | 4462639 | `9ef676d6ed3c8825…` |
