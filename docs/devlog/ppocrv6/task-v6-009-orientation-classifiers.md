---
status: completed
priority: low
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, orientation, classification]
depends_on: task-v6-008
---

# タスク概要
ROADMAP の Follow-ups の「検出・認識以外のモデル」のうち、PaddleOCR 3.x の OCR パイプラインに含まれる 2 つの方向分類器に対応する。

| モデル | 役割 | 入力 | ラベル |
| :--- | :--- | :--- | :--- |
| `PP-LCNet_x1_0_doc_ori` | ページ全体の向き | 短辺を 256 にリサイズし、中央 224x224 を切り出す | `0` / `90` / `180` / `270` |
| `PP-LCNet_x1_0_textline_ori` / `PP-LCNet_x0_25_textline_ori` | 切り出した行の上下 | 160x80 に変形リサイズ | `0_degree` / `180_degree` |

どちらも Hugging Face に `*_onnx` 版がある。

## 調査結果（PaddleX の develop ブランチ）
- 分類器の前処理は `ReadImage(format="RGB")` で始まる。パイプラインから渡される BGR 画像は RGB に変換されるので、**入力は RGB**（検出・認識の BGR とは異なる）。
- `NormalizeImage` は ImageNet の平均と標準偏差、scale は 1/255。
- ページの向きの補正には `rotate_image(img, angle)` を使う。これは `cv2.getRotationMatrix2D(center, angle)` による**反時計回り**の回転で、予測されたラベルの角度だけ回す。
- 行の向きは、0 か 1 の予測に 180 を掛けた角度だけ回転する。PaddleOCR 2.x にあった `cls_thresh` は 3.x では使われておらず、最も確率の高いラベルをそのまま採用する。

## 実装メモ
- `src/orientation.rs`（新規）
  - `OrientationClassifier::from_model_dir` で、`inference.yml` からリサイズ方式・正規化・ラベルを読み込む。
  - ラベル（`180_degree`、`90` など）は角度に変換する。
  - `classify` は、`[N, classes]` の出力から最も確率の高いラベルを返す。
- `paddle_config` で読み取る項目に、`ResizeImage.size` / `resize_short`、`CropImage.size`、`PostProcess.Topk.label_list` を追加した。
- エンジン:
  - **ページの向き**: 検出の前にページを分類し、ラベルの角度だけ反時計回りに回してから検出する。
    - 結果のポリゴンは `unrotate_point` で元画像の座標に戻す。
    - 判定した角度は `OcrRunWithMetrics::doc_orientation_angle` で返す。
  - **行の向き**: 切り出した後、`rec_batch_size` ごとに分類する。`180` と判定された行は 180° 回転する。
    - 最後のバッチは同じ画像で埋めて件数を揃え、推論計画のコンパイルを 1 回で済ませた（約 250 ms の短縮）。
  - 処理時間は `OcrTimings::orientation` に記録する。
- ビルダーの `doc_orientation_model_dir` / `textline_orientation_model_dir`、`ocr_smoke` の `--doc-ori-model-dir` / `--textline-ori-model-dir` で指定する。

## 検証
- ページの向き: 搭乗券を時計回りに 90° / 180° / 270° 回転した画像で、角度がそれぞれ 90 / 180 / 270 と判定された。補正後に `ZHANGQIWEI` が読め、ポリゴンが入力画像の範囲に収まることを確認した（tiny）。
- 行の向き: 上下を反転した搭乗券で確認した。
  - 分類器なしでは `NIIAAAAAOAANIWOISO...` のように全く読めない。
  - 分類器ありでは `BOARDING`、`ZHANGQIWEI`、`张祺伟`、`登机牌`、`GATES CLOSE ...` が読める。
  - ただし、短い大文字だけの行（`TAIYUAN`、`FUZHOU` など）は上下が判別しにくく、取りこぼしがあった。PP-LCNet の 0/180 分類の限界と考えられる。
- 速度（搭乗券、tiny、37 行）:

  | 構成 | 方向分類にかかった時間 |
  | :--- | ---: |
  | doc_ori | 約 0.1 秒 |
  | textline_ori x1_0 | 約 2.2 秒（1 行あたり約 50 ms） |
  | textline_ori x0_25 | 約 0.7 秒 |

  速度を重視する場合は x0_25 を推奨する。
- tract はこれらのモデル（opset 7）について「テスト対象外の opset」と警告を出すが、問題なく動作した。

## 対応しなかったもの
- 文書の歪み補正（UVDoc）とレイアウト解析（PP-DocLayout など）は、OCR パイプラインの外側にある「文書解析」の機能である。出力の形式（レイアウト領域、表、読み順）も含めて、別途設計が必要になる。本タスクの範囲外とし、ROADMAP に残した。
