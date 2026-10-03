---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, detection, config]
depends_on: task-v6-006
---

# タスク概要
ROADMAP の Follow-ups にあった次の 2 件に対応する。
- 検出の原寸モード（PaddleOCR 3.x の `limit_type=min, limit_side_len=64, max_side_limit=4000`）を選べるようにする
- `inference.yml` の `PostProcess` のしきい値を明示的に適用するオプションを追加する

## 実装メモ
- `DetPreProcessorConfig` に `limit_type: DetLimitType` と `max_side_limit` を追加した。PaddleX の `DetResizeForTest`（type0）と同じ順序で計算する。
  1. `Max` の場合は、長辺が `limit_side_len` を超えると縮小する。`Min` の場合は、短辺が `limit_side_len` 未満だと拡大する。
  2. その後、長辺が `max_side_limit` を超えたら縮小する。
- 既定値は従来どおり `Max` / 960 / 4000 とした。CPU 推論の速度を優先したためで、詳しくは [research 3.4](research-ppocrv6.md)。
- `OcrEngineBuilder::det_postprocess_from_model_config(true)` を有効にすると、検出 YAML の `thresh` / `box_thresh` / `unclip_ratio` / `max_candidates` を適用する。
- しきい値の優先順位は「明示的な setter → YAML（オプション有効時）→ パイプライン既定値（0.3 / 0.6 / 1.5 / 1000）」とした。そのため、ビルダー内部ではしきい値を `Option` で保持している。
- `ocr_smoke` に `--det-limit-type max|min`、`--det-max-side-limit N`、`--det-params-from-config` を追加した。
- あわせて、`--help` に過去のタスクで追加したオプション（`--det-model-dir` など）が表示されていなかった問題を修正した。

## 検証
- 単体テスト:
  - `Min` では 1920x1080 が原寸のまま、200x32 が 2 倍に拡大されること。
  - `max_side_limit` によって 8000px が 4000px に抑えられること。
- 統合テスト: tiny_det の YAML（0.2 / 0.4 / 1.4 / 3000）が反映されること。明示的に指定した `det_box_threshold(0.5)` は YAML より優先されること。
- 日本語画像（1536x839）を PP-OCRv6 small で処理した結果:
  - 原寸モードでは検出時間が約 1.1 秒から約 2.7 秒に増えた。
  - その代わり、従来モードでは連結されていた `淹れたてスイート` と `もっちり` が別々の行として検出された。
