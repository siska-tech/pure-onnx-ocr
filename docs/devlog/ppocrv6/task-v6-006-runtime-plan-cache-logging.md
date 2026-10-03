---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, tract, performance, logging]
depends_on: task-v6-005
---

# タスク概要
ROADMAP の Follow-ups にあった次の 2 件に対応する。
- 推論計画の再コンパイル削減（「シンボリックな幅で 1 回だけ最適化する」案）
- ライブラリの `println!` を `log` クレートへ移す

## 調査: シンボリック形状の推論計画
入力の幅（認識）や画像サイズ（検出）をシンボルのまま最適化し、1 つの計画を使い回せるか検証した（tract 0.23.8、release ビルド）。

| モデル | シンボリック計画 | 形状ごとの計画 |
| :--- | :--- | :--- |
| PP-OCRv5 mobile rec `[8,3,48,320]` | コンパイル 769 ms、実行 2.60 s | コンパイル 177 ms、実行 1.00 s |
| PP-OCRv5 mobile rec `[6,3,48,640]` | 実行 4.16 s | コンパイル 150 ms、実行 1.59 s |
| PP-OCRv6 small rec `[8,3,48,320]` | コンパイル 454 ms、実行 1.20 s | コンパイル 147 ms、実行 0.83 s |
| PP-OCRv6 small rec `[2,3,48,1056]` | 実行 1.10 s | コンパイル 161 ms、実行 0.76 s |
| PP-OCRv6 small det | **解析に失敗**（`Concat` で `H/32` と `H/16` の関係を単一化できない） | 問題なし |

結論は次のとおり。
- シンボリック計画は実行が 1.4〜2.6 倍遅くなる。形状ごとのコンパイル（約 150 ms）を節約しても元が取れない。
- 検出モデルはシンボリック形状では解析自体ができない。
- このため **形状ごとの計画を維持する** ことにした。代わりに、問題の本体だった「計画のキャッシュが無制限に増え、メモリを消費し続ける」点を解消した。

## 実装メモ
- `onnx_model::PlanCache`: 入力形状をキーにした LRU キャッシュ。上限を超えると、最も長く使われていない計画を破棄する。
  - 計画ごとに最適化済みの重みを持つので、medium のような大きいモデルでは上限が重要になる。
  - 既定の上限は、検出が 4、認識が 16、方向分類器が 2。
- `DetInferenceSession` / `RecInferenceSession` に `set_plan_cache_capacity` と `cached_plan_count` を追加した。
- `OcrEngineBuilder::plan_cache_capacity(detection, recognition)` を追加した。
- ログの扱い:
  - ライブラリ側の `println!` を廃止した。読み込みは `log::info!`、推論とコンパイル時間は `log::debug!` で出す。
  - `ocr_smoke` は、依存を増やさない最小の stderr ロガーを内蔵した。既定では Warn 以上だけを出す。`-v` / `--verbose` を付けると Debug まで出す。

## 検証
- `PlanCache` の単体テストで、LRU による破棄と上限縮小時の挙動を確認した。
- `ocr_smoke` は既定では推論ログを出さず、`-v` で読み込み・コンパイル・実行時間のログが出ることを確認した。
