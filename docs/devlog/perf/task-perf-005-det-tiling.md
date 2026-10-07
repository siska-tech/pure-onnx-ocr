---
status: todo
priority: low
assignee: Backend
start_date:
end_date:
tags: [performance, detection, multithread, tiling]
depends_on: perf/task-perf-004-det-profile
---

# タスク概要
検出の入力を縦に分割し、それぞれをシングルスレッドの推論で並列に実行して、確率マップを合成する。まず試作して、速度と出力の変化を測る。採用する場合も、**オプトイン**の設定にする。

## 背景
- [benchmark-openvino](benchmark-openvino.md) は、この方法で det 推論が 557 → 約 200 ms（v6 small）になると見積もっている。認識をバッチ並列にしたとき（task-perf-001）と同じ考え方である。
- ただし、[task-perf-004](task-perf-004-det-profile.md) で、v6 の検出モデルはすべて画像全体の平均（空間方向の `ReduceMean`。tiny と small はさらに `GlobalAveragePool`）を使うことが分かった。タイルごとに平均が変わるので、どれだけ重ねても確率マップは元と一致しない。既定の動作で PaddleOCR と同じ出力を保つ方針に反するので、既定では使わない。
- task-perf-004 により、出力を変えずに同程度の効果が見込める [task-perf-006](task-perf-006-tract-intraop.md)（depthwise の並列化など）を優先する。本タスクは、006 の結果を見てから着手するかどうかを決める。
- 搭乗券くらいの小さい入力（`512x896`）では、重ねる部分の割合が大きく、計算量が増える。約 200 ms という見積もりは楽観的である。

## 要件
- `DetInferenceSession` に、分割数と重なりの幅を受け取る実行経路を追加する。各タイルは `threading::parallel_map` で並列に、シングルスレッドで推論する。
- 合成は、各タイルの重なりを除いた中央部分をつなぐ。タイルの高さは 32 の倍数にそろえる。
- 分割数（2、3、4）と重なりの幅を変えて、次を測る。
  - det 推論の時間と、パイプライン全体の時間
  - 元の推論と比べた出力の変化: 全フィクスチャでの領域数、文字列、box の IoU、日本語 57 語の一致数
- 採用する場合は `OcrEngineBuilder` にオプトインの設定（例: `detection_tiles(n)`）を追加し、出力が変わりうることをドキュメントに書く。

## 不採用にする条件
- 文字列の一致数が下がる、または速度の改善が 20% 未満の場合。

## 作業ログ

## テスト
