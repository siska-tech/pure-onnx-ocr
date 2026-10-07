---
status: todo
priority: high
assignee: Backend
start_date:
end_date:
tags: [performance, detection, tract, profiling]
depends_on: perf/task-perf-002-openvino-bench
---

# タスク概要
検出モデル（DBNet）のどの演算子が遅いのかを特定する。検出を速くする方法（task-perf-005 のタイル分割、task-perf-006 の tract の演算子内の並列化、グラフの書き換え）のどれに注力するかを、この結果で決める。

## 背景
[benchmark-openvino](benchmark-openvino.md) より:

- パイプライン全体の差の 6〜9 割は、検出の推論から来ている。
- 検出モデル単体では、本クレートは OpenVINO の 5〜8 倍遅い。
  - 1 スレッド同士でも 2.6〜4.8 倍遅い（カーネルの効率の差）。
  - 1 → 8 スレッドで 1.23〜1.29 倍にしかならない（並列化の差）。OpenVINO は 2〜2.5 倍になる。

## 調べること
1. **演算子ごとの時間**
   - tract CLI の `--profile --cost` で、v6 tiny / small / medium の検出を 1 スレッドと 8 スレッドで測る。入力は搭乗券を前処理した `[1,3,512,896]`。
   - depthwise 畳み込み、通常の畳み込み、`ConvTranspose`、`Resize`、要素ごとの演算（`HardSigmoid` など）、SE ブロック（`GlobalAveragePool`）に分けて集計する。
   - 8 スレッドにしたときに速くなる演算子と、速くならない演算子を分ける。
2. **画像全体を使う演算子の確認（タイル分割で結果が同じになるか）**
   - v6 tiny と small の検出には `GlobalAveragePool`（SE ブロック）が含まれる。画像全体の平均を使うので、タイルに分けると確率マップは元と一致しない。
   - 全モデルに含まれる `ReduceMean` の軸を調べる。チャネル方向（LayerNorm 系）なら、タイル分割しても結果は変わらない。空間方向なら変わる。
   - medium で `GlobalAveragePool` が見当たらないことも確かめる。

## 成果物
- 演算子ごとの時間の表と、そこから見た「次にやること」を本書にまとめる。
- task-perf-005、task-perf-006 の優先度と方針を、結果に合わせて更新する。

## 作業ログ

## テスト
