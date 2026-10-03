---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, preprocessing, postprocessing, quality]
depends_on: task-v6-002
---

# タスク概要
検出と認識の前処理・後処理を、PaddleOCR 3.x（PaddleX の `OCR.yaml` と `text_recognition/processors.py`）の挙動に合わせる。PP-OCRv6 の `inference.yml` が前提とする入力仕様を満たすことが目的。

## 差分と対応

| 項目 | 変更前 | 変更後 (= PaddleOCR 3.x) |
| :--- | :--- | :--- |
| 検出の色順 | RGB | BGR（`DetPreProcessorConfig::color_order`） |
| 検出の正規化 | `x / 255` | `(x / 255 - mean) / std`。ImageNet の値を使う（`mean` / `std`） |
| 検出のリサイズフィルタ | Lanczos3 | Triangle（bilinear。`cv2.resize` の既定に合わせた） |
| 検出のパディング | 生の 0（黒） | 正規化後の 0（平均色） |
| 輪郭 | 外側と穴の両方 | 外側の輪郭だけ |
| `box_thresh` | なし | 輪郭内の平均確率が 0.6 未満なら破棄（`box_threshold`） |
| `max_candidates` | なし | 1000（面積の大きい順に残す） |
| 認識の色順 | RGB | BGR |
| 認識の幅 | 320 に固定。超える分は押し潰す | `max(320, ceil(48 x 縦横比))`。上限 3200 で、32 の倍数に切り上げる |
| 認識のパディング | 正規化後に -1 | 正規化後に 0（`pad_value` = 0.5） |
| 認識のリサイズフィルタ | Lanczos3 | Triangle |

## 実装メモ
- `ColorOrder::source_channel(c)` で、出力チャネルを RGB 画像のどのチャネルから読むかを決める。`mean` と `std` は、PaddleOCR と同じくモデル側のチャネル順（BGR）で並べる。
- 検出の正規化は、`scale = 1 / (255 * std)` と `offset = mean / std` をあらかじめ計算し、画素ごとの割り算をなくした。
- `contour_score()`
  - 輪郭の外接矩形の中を走査線で塗りつぶしたマスクを作る。境界の画素も含める。
  - マスク内の確率の平均を求める。PaddleOCR の `box_score_fast` に相当する。
- `RecPreProcessorConfig` に `max_dynamic_width`（3200）と `width_alignment`（32）を追加した。`max_dynamic_width = max_width` にすると、従来の固定幅に戻る。
- `RecInferenceSession::load_with_input_height` を追加した。`inference.yml` の `image_shape[1]` を反映する。

## テスト
- 検出: ImageNet 正規化の値と、BGR 順（赤が第 3 チャネルに入ること）を確認した。
- 認識: 長い領域でバッチ幅が広がること（1200px の領域で幅 1216）と、上限が効くことを確認した。
- 後処理: 平均確率 0.4 の領域が `box_threshold = 0.6` で除外されることと、`max_candidates` で大きい領域が残ることを確認した。

## 影響（破壊的変更）
- 3 つの設定構造体（`DetPreProcessorConfig`、`DetPostProcessorConfig`、`RecPreProcessorConfig`）にフィールドを追加した。リテラルで初期化しているコードは `..Default::default()` が必要。
- 既定の挙動が変わるため、PP-OCRv5 を使っている場合も出力が変わる。実測では改善方向（[task-v6-005](task-v6-005-validation.md)）。
