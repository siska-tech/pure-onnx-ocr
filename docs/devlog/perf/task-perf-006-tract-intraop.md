---
status: todo
priority: high
assignee: Backend
start_date:
end_date:
tags: [performance, detection, tract, upstream, depthwise]
depends_on: perf/task-perf-004-det-profile
---

# タスク概要
tract の depthwise 畳み込みを中心に、検出で並列化されていない演算子を速くする（並列化と SIMD 化）。tract 本体への PR として提案する。

## 背景
- tract の並列化は行列演算の中だけで、検出は 8 スレッドでも 1.23〜1.29 倍にしかならない（[benchmark-openvino](benchmark-openvino.md)、task-perf-001）。OpenVINO は演算子の中でも空間方向に分割して、2〜2.5 倍に伸びる。
- この方法なら出力は変わらない。task-perf-005 のタイル分割と違い、既定の動作に入れられる。本クレート以外の tract の利用者にも効く。
- [task-perf-004](task-perf-004-det-profile.md) で、検出の最大のボトルネックは depthwise 畳み込みだと分かった。x86_64 では SIMD のカーネルがなく、チャネル方向の並列化もない。
- tract 側のレビューに時間がかかるので、fork での計測を先に進め、PR は効果を確かめたものから出す。

## 要件

[task-perf-004](task-perf-004-det-profile.md) のプロファイルにもとづき、効果の大きい順に進める。8 スレッドでは、次の 4 つの並列化されない処理が検出時間の約 70% を占める。

| 順 | 対象 | 内容 | 出力 | 見込める効果（small / medium、det 推論） |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `DepthWiseConv` | チャネル方向に並列化する（tract-core の `ops/cnn/conv/depth_wise.rs` のチャネルのループ） | 完全に一致する | −130 ms / −700 ms |
| 2 | `DepthWiseConv` | x86_64 の SIMD（AVX2 / FMA）のカーネルを追加する。タップ数の多い内側の領域（3x3、7x7、9x9）に効く形にする | FMA で丸めが変わる可能性。変化を測る | 1 スレッドで数倍 |
| 3 | `MultiBroadcastTo`（ConvTranspose のバイアス） | バイアスを出力全体に展開せず、DeconvSum の初期値などで足す | 足す順序が同じなら一致する | −34 ms / −100 ms |
| 4 | `OptMatMulPack`、`Pad` | 並列化する。またはパディングを畳み込みに取り込む | 一致する | −50 ms / −200 ms 程度 |

- 1 と 3 は小さい変更で、出力も変わらない。まずこの 2 つを tract の fork で実装し、本クレートの検出の時間と出力の一致を確かめてから、tract に PR を出す。
- 2 は tract-linalg への追加になる。arm64 には `depthwise_w_f32` があるので、同じ枠組み（`routines.rs` の `Func::DepthwiseW`）に x86_64 版を追加する形で提案する。ただし、今の枠組みはタップ数 4 以下の領域にしか使われていないので、呼び出し側も変える必要がある。
- tract の作業領域が thread-local の `RefCell` であることに注意する（task-perf-001 で、入れ子の rayon 実行による panic があった）。depthwise は行列演算の作業領域を使わないが、並列化には `multithread_tract_scope` で渡される実行器を使う。
- 計測には [tools/tract-profile](../../../tools/tract-profile/README.md)（演算子ごと）と [tools/openvino-bench](../../../tools/openvino-bench/README.md)（パイプライン全体の A/B）を使う。tract の fork は、計測用に `[patch.crates-io]` で差し替える。

## 完了の条件
- tract に PR を出し、取り込まれたら本クレートの tract を更新して効果を計測する。
- 取り込まれるまでの間、fork の tract に依存して公開することはしない。

## 作業ログ

## テスト
