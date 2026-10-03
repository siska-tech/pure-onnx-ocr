---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, tract, onnx, dependency]
depends_on:
---

# タスク概要
PP-OCRv6 の ONNX モデル 6 種（tiny / small / medium × det / rec）を `tract-onnx` で読み込み、推論できるようにする。

## 調査結果
- tract 0.20.7 では、6 モデルすべてが `Impossible to unify Sym(DynamicDimension.0) with Val(1)` で失敗した。
  - 原因: PaddleOCR 3.x のエクスポートは、ほぼすべての中間テンソルに `value_info` でシンボリック次元を付けている。tract はこれを推論ルールとして採用する。
  - tract-onnx には `value_info` を無視するオプションがない（`ignore_output_shapes` は出力だけが対象）。
  - Hugging Face 配布の PP-OCRv5 mobile も同じ理由で失敗する（develop ブランチで確認）。
- value_info を破棄しても、tract 0.20.7 では medium_rec だけが失敗した。
  - `tract-core-0.20.7/src/ops/nn/reduce.rs:289` に `ensure!(!shape.iter().any(|d| *d == 768.to_dim()))` というデバッグ用のコードが残っている。medium_rec の隠れ次元は 768 なので、これに引っかかる。
- tract 0.21 は `DynamicDimension.0` をパースできずに失敗した。
- tract 0.23.8 では 6 モデルすべてが動作した。推論は 0.20 の約 1.5〜2 倍速い。

## 実装メモ
- `Cargo.toml`
  - `tract-onnx` を 0.20 から 0.23 へ、`ndarray` を 0.15 から 0.17 へ上げた。ndarray は tract と同じ版に揃えないと `Tensor::from(Array)` が使えない。
  - `rust-version` を 1.70 から 1.91 へ上げた。tract 0.23 の MSRV に合わせている。
- `src/onnx_model.rs`（新規）: `load_paddle_onnx()` で ONNX を読み込んだ後、入力でも定数でもない outlet の fact を `InferenceFact::default()` にリセットする。検出・認識・ダミー推論のすべてがこの関数を通る。
- tract 0.23 で変わった API に追従した。
  - `TypedRunnableModel<TypedModel>` は `TypedRunnableModel` に変わった。`into_runnable()` は `Arc` を返すようになった。
  - `symbol_table` は `symbols` に変わった。
  - `to_array_view` は `to_plain_array_view` に変わった。
  - `anyhow!` の import 元は `tract_core::internal` に変わった。
  - `InferenceFact::from(&Tensor)` は使えなくなったので、`InferenceFact::dt_shape` を使う。

## 検証
- 6 モデルとも、ゼロ入力で推論できることを確認した。検出は `[1, 1, 320, 320]`、認識は `[1, 40, C]` を出力する。
- ダミー推論テスト（`cargo test --release --lib -- --ignored`）は PP-OCRv5 mobile で成功した。0.20 では 60 秒以上かかるとされていたが、0.34 秒で完了した。

## 影響
- MSRV が 1.91 に上がるため、README の前提条件を更新した。
- 公開 API の型は変わらない。ただし `Tensor` などの tract の型は 0.23 のものになる。
