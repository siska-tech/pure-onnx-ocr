---
status: todo
priority: high
assignee: Backend
start_date:
end_date:
tags: [performance, multithread, rayon]
depends_on: perf/task-perf-002-openvino-bench
---

# タスク概要
既定の推論スレッド数を「論理 CPU 数、最大 8」から「論理 CPU 数、最大 16」に引き上げる。

## 背景
[benchmark-openvino](benchmark-openvino.md) の追加計測（v6 small、交互に 3 ラウンド、ミリ秒）:

| スレッド数 | 合計（搭乗券 / 日本語） | rec 推論 | det 推論 | 平均使用コア数 |
|---:|---:|---:|---:|---:|
| 8（既定） | 1,389 / 1,529 | 746 / 834 | 572 / 562 | 4.3 |
| 12 | 1,404 / 1,502 | 714 / 734 | 614 / 630 | 5.6 |
| 16 | 1,256 / 1,284 | 636 / 608 | 577 / 570 | 6.3 |

- 16 スレッドで、認識が 15〜27% 速くなり、合計は 10〜16% 縮んだ。検出は変わらない。
- task-perf-001 の「8 と 16 で差がない」は、行列演算の並列化だけを計測した結果だった。認識のバッチを並列に実行する現在の実装では、16 スレッドにも効果がある。

## 要件
- `src/threading.rs` の `default_inference_threads()` の上限を 16 にし、ドキュメントコメントの根拠を更新する。
- v6 tiny / small / medium と v5 mobile / server で、8 と 16 を交互に計測する。合計の時間、スレッド数、メモリのピークを記録する。
- 可能なら、論理 CPU が 8 以下の環境と、ハイブリッドでない CPU でも、悪化しないことを確かめる。
- 既定の動作が変わるので、CHANGELOG に記載する。task-perf-001 の「4. スレッド数の既定値」に、結論を修正したことを追記する。

## 完了の条件
- 上の計測で、どのモデルでも合計の時間が悪化しない。
- 出力が変わらない（`thread_count_does_not_change_results` を含む全テストが成功する）。

## 作業ログ

## テスト
