# 詳細設計書 (モジュール設計書): Pure Rust OnnxOCR

作成者: Shion Watanabe  
初版: 2025-11-09  
改訂: 2026-10-03（v0.2.0）  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * 各モジュールの内部構造とアルゴリズムを、実装に合わせて記述する。
  * PaddleOCR 3.x（PaddleX）の参照実装との対応関係を明記する。前処理と後処理は本家と同じ結果になるように実装しており、`tests/paddle_parity.rs` で検証している。

-----

## 1\. 内部クラス・関数設計

| モジュール | 主な型・関数 | 責務 |
| :--- | :--- | :--- |
| `engine` | `OcrEngineBuilder`、`OcrEngine`、`DetectionPipeline`、`RecognitionPipeline`（非公開） | 設定の組み立てと、パイプライン全体の制御 |
| `onnx_model`（非公開） | `load_paddle_onnx(_from_bytes)`、`PlanCache`、`lock_cache` | ONNX の読み込みと `value_info` の破棄、入力形状ごとの推論計画の LRU キャッシュ |
| `detection` | `DetInferenceSession` | DBNet の推論（入力形状は `[1,3,H,W]`） |
| `recognition` | `RecInferenceSession`、`RecPostProcessor` | 認識モデルの推論（`[N,3,48,W]`）と CTC デコード |
| `preprocessing` | `DetPreProcessor`、`RecPreProcessor` | リサイズ、正規化、テンソルへの変換 |
| `postprocessing` | `DetPostProcessor::db_boxes`、`DetPolygonUnclipper` | DB 後処理（矩形、スコア、unclip） |
| `crop` | `min_area_quad`、`crop_quad` | 最小面積矩形の算出と、透視変換による切り出し |
| `imgproc` | `resize_bilinear` | `cv2.resize(INTER_LINEAR)` と同じ結果を返すリサイズ |
| `paddle_config` | `PaddleInferenceConfig`、最小限の YAML パーサ | `inference.yml` の読み込み |
| `dictionary` | `RecDictionary` | blank（0）+ 文字 + space（任意）の対応表 |
| `ctc` | `CtcGreedyDecoder` | 重複とブランクを取り除き、信頼度を計算する |
| `orientation` | `OrientationClassifier`、`rotate_ccw`、`unrotate_point` | PP-LCNet による方向分類 |
| `threading`（非公開） | `executor_for`、`run_with`、`parallel_map` | rayon のスレッドプールと、tract の実行器の差し替え |
| `time`（非公開） | `Instant` | ブラウザでは `web_time`、それ以外では `std` を使う |

## 2\. データ構造

| 型 | 内容 |
| :--- | :--- |
| `PreprocessedDetInput` | `tensor`、`resized_dims`、`scale_ratio`、`scale_xy`（縦横それぞれの倍率） |
| `DetBox` | `quad: [(f64,f64);4]`（tl, tr, br, bl）、`score` |
| `PreprocessedRecBatch` | `tensor [N,3,48,W]`、`valid_widths`、`max_width` |
| `RecInferenceOutput` | `logits [N,T,C]`、`valid_timesteps` |
| `DecodedSequence` | `text`、`token_indices`、`confidence`、`fallback_count` |
| `OcrEngineConfig` | 前処理・後処理の各設定、`rec_batch_size`、`rec_crop_mode`、`inference_threads` |

## 3\. アルゴリズム・ロジック

### 3.1. `OcrEngineBuilder::build`

1.  検出モデル・認識モデル・辞書のいずれかが指定されていなければ、`MissingField` を返す。辞書は `dictionary_text` → `dictionary_path` → `rec_config_yaml` → `rec_config_path` の順で探す。
2.  `inference.yml` を解析し、設定に反映する。
    * 検出: 色順と、正規化の mean / std
    * 認識: 色順と `image_shape`
    * `det_postprocess_from_model_config` が有効な場合は、しきい値も反映する。
3.  ONNX を読み込む。`load_paddle_onnx` は、入力と定数を除くすべてのテンソルの形状情報（fact）を消す。PaddleOCR 3.x のエクスポートは `DynamicDimension.*` というシンボル付きの `value_info` を持っており、そのままだと具体的な入力形状と矛盾して解析に失敗するためである。
4.  入力の高さ（48）を固定し、幅とバッチ数をシンボルのまま残した基本モデルを作る。推論計画は、実際の入力形状ごとにコンパイルし、LRU キャッシュに保持する（検出 4 個、認識 16 個）。
5.  辞書を構築し、既定で space を追加する。
6.  スレッドプールを生成する（`inference_threads`。WebAssembly では 1）。

### 3.2. `OcrEngine::run_from_image` (メインロジック)

1.  （任意）ページの向きを分類し、0 度以外なら `rotate_ccw` で画像を正立させる。
2.  `DetectionPipeline` で、入力画像の座標系の矩形の一覧を得る。
3.  `crop_regions` で切り出す（`Rotated` の場合は `crop_quad`、失敗した場合は外接矩形で切り出す）。
4.  （任意）行の上下を分類し、`180_degree` と判定された切り出しを 180 度回転する。最後のバッチは件数を揃えるために埋めて、推論計画を使い回す。
5.  `RecognitionPipeline::run_with_timings` で認識する。
6.  ページを回転していた場合は、`unrotate_point` で矩形を元の入力画像の座標に戻す。

### 3.3. 検出前処理（`DetPreProcessor`、PaddleX `DetResizeForTest` 相当）

1.  倍率 `ratio` を決める。
    * `Max` の場合: 長辺が `limit` を超えるなら `limit / 長辺`
    * `Min` の場合: 短辺が `limit` 未満なら `limit / 短辺`
2.  `resize = int(辺 × ratio)` とする。長辺が `max_side_limit` を超える場合は、さらに縮小する。
3.  各辺を `max(round_half_even(辺 / 32) × 32, 32)` にする。
4.  画像をそのサイズに**引き伸ばす**（`imgproc::resize_bilinear`）。
5.  BGR の順に並べ、`(x/255 - mean) / std` で正規化する（ImageNet の値）。
6.  `scale_xy = (変換後の幅 / 元の幅, 変換後の高さ / 元の高さ)` を保持する。

### 3.4. 検出後処理（`DetPostProcessor::db_boxes`、PaddleX `DBPostProcess` 相当）

1.  `probability > thresh` で二値化する。
2.  `find_contours` で輪郭を取り出す。外側の輪郭も穴の輪郭も対象にする（`RETR_LIST` 相当）。先頭から `max_candidates` 個まで処理する。
3.  輪郭ごとに、凸包を求め、rotating calipers で最小面積矩形を得る。短辺が 3 未満なら除外する。
4.  スコアを計算する（`box_score_fast` 相当）。矩形の外接範囲の中で、矩形に含まれる画素の確率を平均する。`box_thresh` 未満なら除外する。
5.  **矩形**を膨張させる（unclip）。距離は `面積 × unclip_ratio / 周長` で、`i_overlay` を使い角は丸める。膨張後の最大の多角形について最小面積矩形を求め、短辺が 5 未満なら除外する。
6.  エンジンで、各座標に `inverse_scale` を掛けて四捨五入し、画像の範囲に収める。

### 3.5. 切り出し（`crop::crop_quad`、PaddleX `get_rotate_crop_image` 相当）

* 幅 = `int(max(|tl-tr|, |bl-br|))`、高さ = `int(max(|tl-bl|, |tr-br|))` とする。
* 矩形を `(0,0)-(w,h)` に写す透視変換を求め、bicubic で切り出す。
* `h / w ≥ 1.5` の場合は、反時計回りに 90 度回転する（`np.rot90` と同じ）。

### 3.6. 認識前処理（`RecPreProcessor`、PaddleX `OCRReisizeNormImg` 相当）

1.  縦横比を `r = w / h` として、各切り出しの幅を決める。`r > 320/48` なら `int(48r)`、それ以外は `ceil(48r)` とする。
2.  バッチの幅は `max(320, 最大の幅)` を上限 3200 で抑え、32 の倍数に切り上げる。切り上げは推論計画の再利用のためで、結果には影響しないことを確認している。
3.  各切り出しを、`resize_bilinear` で高さ 48 にリサイズする。
4.  BGR の順に並べ、`(x/255 - 0.5) / 0.5` で正規化する。残りの幅は 0（正規化後の値）で埋める。
5.  `RecognitionPipeline` は、切り出しを縦横比でソートしてから `rec_batch_size` ごとに分ける。前処理・推論・デコードの各ステージを、`parallel_map` で並列に処理する。並列で動く推論は、`run_single_threaded` を使う。

### 3.7. 認識後処理（`CtcGreedyDecoder`）

1.  各時刻で、確率が最大のクラスを選ぶ。有効幅を超える時刻は無視する。
2.  blank（0）と連続する重複を取り除き、辞書で文字に変換する。辞書の範囲外のクラスは、代替文字列 `[UNK]` にする。
3.  信頼度は、選ばれた文字の確率の平均とする。出力が確率でない場合（ロジット）は、log-sum-exp で softmax を計算する。

### 3.8. 方向分類（`OrientationClassifier`）

* 前処理は `inference.yml` に従う。
  * `ResizeImage.size`: 指定サイズに変形リサイズ
  * `resize_short` と `CropImage`: 短辺をリサイズしてから中央を切り出す
* 入力は RGB で、ImageNet の値で正規化する。
* 出力は確率が最大のラベルとし、ラベル名（`180_degree`、`90` など）を角度に変換する。
* ページの向きの補正は、`rotate_ccw(angle)`（反時計回りの回転）で行う。これは PaddleX の `rotate_image` と同じである。

## 4\. エラー処理の詳細

* すべての公開 API は `Result<_, OcrError>` を返し、ライブラリ内で panic しない。
  * 例外として、スレッドプールを作れない場合は `catch_unwind` で受け止め、警告を出してシングルスレッドに切り替える。
* 入力元がメモリの場合、エラーメッセージのパスは `<memory>` と表示する。
* `inference.yml` に対応していない構文（フロー形式、ブロックスカラー）が含まれていれば、`PaddleConfigError::Syntax` として、行番号付きで報告する。
* 推論計画キャッシュのロックが汚染されていても（poison）、処理を続ける。キャッシュは不完全な状態にならないためである。
