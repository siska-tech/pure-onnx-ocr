---
status: todo
priority: medium
assignee: Backend
start_date:
end_date:
tags: [performance, detection, tract, upstream]
depends_on: perf/task-perf-004-det-profile
---

# タスク概要
tract で、行列演算以外の重い演算子（task-perf-004 で特定したもの）も、演算子の中で並列に実行できるようにする。tract 本体への PR として提案する。

## 背景
- tract の並列化は行列演算の中だけで、検出は 8 スレッドでも 1.23〜1.29 倍にしかならない（[benchmark-openvino](benchmark-openvino.md)、task-perf-001）。OpenVINO は演算子の中でも空間方向に分割して、2〜2.5 倍に伸びる。
- この方法なら出力は変わらない。task-perf-005 のタイル分割と違い、既定の動作に入れられる。本クレート以外の tract の利用者にも効く。
- tract 側のレビューに時間がかかるので、task-perf-005 と並行して進める。

## 要件
- task-perf-004 で、8 スレッドにしても速くならなかった演算子を対象にする。候補は depthwise 畳み込み、要素ごとの演算、`Resize`、`ConvTranspose`。
- tract-linalg の `multithread` の executor（`multithread_tract_scope` で渡しているもの）の上で、出力をチャネル方向か空間方向に分割して並列に計算する。
- tract の作業領域が thread-local の `RefCell` であることに注意する（task-perf-001 で、入れ子の rayon 実行による panic があった）。
- tract 本体に提案する前に、fork で本クレートの検出の時間と出力の一致を確かめる。

## 完了の条件
- tract に PR を出し、取り込まれたら本クレートの tract を更新して効果を計測する。
- 取り込まれるまでの間、fork の tract に依存して公開することはしない。

## 作業ログ

## テスト
