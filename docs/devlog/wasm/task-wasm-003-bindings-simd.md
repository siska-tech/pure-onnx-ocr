---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [wasm, bindings, performance]
depends_on: task-wasm-002
---

# タスク概要
JavaScript から呼べるバインディングを用意し、`wasm32-unknown-unknown` で実際に動作することを確認する。

## 実装メモ
- `bindings/wasm`（クレート名 `pure-onnx-ocr-wasm`、`publish = false`）を追加し、ルートの `Cargo.toml` を workspace にした。
  - 本体クレートは wasm-bindgen に依存しない。
  - 本体側では `exclude = ["bindings/"]` を指定し、パッケージに含めない。
- JS 側の API:
  - `new OcrEngineBuilder()` に、次のメソッドをチェーンして設定する。
    - `.detModel(bytes, yaml)` / `.recModel(bytes, yaml)`
    - `.dictionaryText` / `.docOrientationModel` / `.textlineOrientationModel`
    - `.detLimitSideLen` / `.detLimitType("max"|"min")` / `.cropMode("rotated"|"axis")` / `.recBatchSize`
    - `.build()`
  - `engine.run(imageBytes)` は `[{ text, confidence, box, polygon }]` を返す。
    - `box` は、回転補正に使う最小面積の矩形（4 点、左上から時計回り）。
    - `polygon` は検出の輪郭で、数百点になることがある。UI で描画するなら `box` を使う。
  - `engine.runWithMetrics(imageBytes)` は `{ results, timings(ms), docOrientationAngle }` を返す。
- `.cargo/config.toml` で、`wasm32-unknown-unknown` 向けに `-C target-feature=+simd128` を指定した。

## 検証（Node.js 24、`wasm-bindgen --target nodejs`、PP-OCRv6 tiny、搭乗券）

| ビルド | 合計 | 検出 | 認識 |
| :--- | ---: | ---: | ---: |
| SIMD なし | 6.7 秒 | 1.4 秒 | 5.2 秒 |
| SIMD あり | 3.1 秒 | 1.0 秒 | 2.1 秒 |

- 認識結果のテキスト（37 領域）は、ネイティブ版と完全に一致した。
- 方向分類器（ページ + 行の向き）を有効にしても動作した。
- rayon（image と imageproc が利用）は、スレッドがない環境では現在のスレッドで処理するように切り替わり、問題なく動作した。
- 同じソースから wasm32-wasip1 向けにビルドした `ocr_smoke` も、Node.js の WASI 上で動作した。
