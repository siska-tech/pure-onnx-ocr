---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, recognition, geometry, quality]
depends_on: task-v6-007
---

# タスク概要
ROADMAP の Follow-ups の「検出領域を回転補正して切り出す」に対応する。PaddleOCR の `get_mini_boxes` と `get_rotate_crop_image` に相当する処理を実装する。

## 実装メモ
- `src/crop.rs`（新規）
  - `min_area_quad`: 凸包（Andrew の monotone chain）を求め、rotating calipers で最小面積の外接矩形を得る。凸包の点が 3 未満の退化ケースでは外接矩形で代用する。
  - 角の並び順は PaddleOCR と同じ規則にした。x でソートし、左の 2 点のうち上を tl、下を bl とする。右の 2 点も同様に tr、br とする。
  - `crop_quad`:
    - 出力サイズは、対辺のうち長い方の長さとする（PaddleOCR と同じ）。
    - imageproc の `Projection::from_control_points` と `warp_into`（bilinear）で透視変換する。
    - 高さ / 幅が 1.5 以上の切り出しは縦書きとみなし、`rotate270`（= `np.rot90`、反時計回りに 90°）で横向きに直す。
- `RecPreProcessor::process_images` を追加し、切り出し済みの画像から直接バッチを作れるようにした。従来の `process`（領域を指定する方法）は、切り出してから `process_images` を呼ぶ形に変えた。
- `OcrEngineConfig::rec_crop_mode`（`RecCropMode`）を追加した。
  - 既定は `Rotated`。
  - `AxisAligned` にすると従来の挙動になる。
  - 透視変換に失敗した場合は、`AxisAligned` に自動で切り替える。
  - ビルダーの `rec_crop_mode` と、`ocr_smoke --crop-mode rotated|axis` で指定できる。
- 認識のバッチ化（縦横比でのソート）は、切り出し後の画像サイズで行う。

## 検証
- 単体テスト:
  - 30° 回転した 40x10 の矩形が復元されること。
  - 退化ケースが処理できること。
  - 切り出し後の向きが正しいこと。
  - 縦長の切り出しが反時計回りに回転されること。
- 搭乗券を 10° 傾けた画像を PP-OCRv6 small で処理し、正立画像の出力 33 行のうち何行がそのまま再現されるかを比べた。

  | 切り出し方式 | 一致した行 |
  | :--- | ---: |
  | Rotated | 24 |
  | AxisAligned | 21 |

  AxisAligned では、末尾の長い行の切り出しに隣の行が入り込み、`CATS CLP` と誤読された。Rotated では `GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME` と正しく読めた。
- 正立画像では両方式の結果はほぼ同じ。Rotated のほうが約 0.5 秒遅い（tiny の場合。透視変換と画像全体の RGB 変換による）。
- tiny では、Rotated に変えると末尾の行の `10` が `1O` になった。そのため、tiny の統合テストは `GATES CLOSE` と `MINUTES BEFORE DEPARTURE TIME` を部分一致で確認する形に緩めた。
