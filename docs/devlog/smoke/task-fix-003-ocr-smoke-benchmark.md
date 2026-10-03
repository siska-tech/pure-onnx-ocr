---
status: completed
priority: low
assignee: Backend
start_date: 2025-11-10
end_date: 2025-11-10
tags: [benchmark, tooling, ocr-smoke]
depends_on: task-fix-000
---

# タスク概要
`ocr_smoke` CLI に推論時間および主要ステージ（前処理／推論／後処理）の計測を追加し、ベンチマーク用途に利用できるよう整備する。

## 要件
- 1 回の実行につき、全体の経過時間と少なくとも DBNet・SVTR 推論の時間を `info` ログとして出力する。
- 計測は `Instant` など標準 API を用いて行い、Windows・Unix 双方で動作すること。
- オプション（例: `--benchmark`）で計測のオン／オフを切り替え可能にし、既存の既定出力フォーマットを壊さない。
- `docs/devlog/task-fix-001-ocr-smoke-quality.md` 等で計測手順とサンプル出力を共有する。
- 将来的に自動ベンチマークへ取り込めるよう、測定結果を JSON 形式で保存する仕組みの追加も検討事項として記録する。

## 実装メモ
- `ocr_smoke --benchmark` 実行時に `[INFO] benchmark.*` 形式のログで総時間と DBNet / SVTR の前処理・推論・後処理を出力する。
- `OcrEngine::run_with_metrics_from_path` / `run_with_metrics_from_image` により、CLI 以外の呼び出し元も同一メトリクスを取得できる。
- JSON 出力は follow-up として `docs/devlog/ROADMAP_fix.md` に記録済み。専用 CLI オプションを別タスクで検討する。

## サンプル出力
```
[INFO] benchmark.image=tests/fixtures/images/demo.png
[INFO] benchmark.total_seconds=0.412583
[INFO] benchmark.image_decode_seconds=0.003121
[INFO] benchmark.det.preprocess_seconds=0.044512
[INFO] benchmark.det.inference_seconds=0.221009
[INFO] benchmark.det.postprocess_seconds=0.012334
[INFO] benchmark.rec.preprocess_seconds=0.018775
[INFO] benchmark.rec.inference_seconds=0.094281
[INFO] benchmark.rec.postprocess_seconds=0.005237
```

