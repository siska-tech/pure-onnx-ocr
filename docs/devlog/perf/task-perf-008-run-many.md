---
status: todo
priority: medium
assignee: Backend
start_date:
end_date:
tags: [performance, throughput, multithread, api]
depends_on: perf/task-perf-003-default-threads
---

# タスク概要
複数の画像をまとめて処理する API（例: `OcrEngine::run_many`）を追加し、スループットを上げる。

## 背景
[benchmark-openvino](benchmark-openvino.md) より:

- 本クレートの平均使用コア数は 3.2〜4.8 で、OpenVINO は 7.1〜10.4 だった。
- 1 枚あたりの CPU 時間の総量は、OpenVINO とほぼ同じだった（small 約 6.2 対 5.9 コア秒）。
- 実時間の差の多くは「同時に何コア使えているか」の差で、特に検出は 1 回の推論が並列化されない。

検出は画像ごとに独立しているので、画像をまたいで並列に実行すればよく並列化できる。結果も変わらない。

## 要件
- 複数の画像の検出を、それぞれシングルスレッドの推論で並列に実行する。
- 認識は、全画像の切り出しをまとめて、既存のバッチ並列の仕組みで実行する。
- 結果は入力の順に返す。1 枚ずつ `run` した結果と完全に一致させる。
- 画像の数に比例してメモリが増えすぎないように、同時に処理する画像の数に上限を設ける。

## 計測
- 10 枚以上の画像で、1 枚ずつ `run` した場合と、`run_many` の場合の枚/秒を比べる。可能なら OpenVINO の `ov` 設定とも比べる。

## 作業ログ

## テスト
