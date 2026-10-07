# アーキテクチャ設計書：Pure Rust OnnxOCR

作成者: Shion Watanabe  
初版: 2025-11-09  
改訂: 2026-10-03（v0.2.0：PP-OCRv6、PaddleOCR 3.x 互換、WebAssembly、マルチスレッド）  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * **プログラム全体の構造（モジュール構成）を示す。**
    推論の中心は `OcrEngine` である。その周囲に、次のパイプラインを独立したモジュールとして配置する。
    * 検出: 前処理 → 推論 → 後処理
    * 認識: 切り出し → 前処理 → 推論 → CTC デコード
    * 任意の方向分類: ページの向き・行の上下

  * **主要なモジュール間の役割分担と依存関係を明確にする。**
    * `OcrEngine` は、パイプライン全体の流れを制御する。
    * 検出パイプラインは、画像からテキスト領域（4 点の矩形）を求める。
    * 認識パイプラインは、切り出した画像からテキストを求める。
    * ONNX の読み込み、`inference.yml` の解釈、画像のリサンプリング、スレッドプールは、それぞれ専用のモジュールに分離する。

  * **採用する設計原則やデザインパターンを定義する。**
    * 関心の分離と、Facade パターン（`OcrEngine`）、Builder パターン（`OcrEngineBuilder`）を採用する。
    * 前処理と後処理は、PaddleOCR 3.x（PaddleX）の参照実装と**同じ結果になること**を設計方針とし、それを `tests/paddle_parity.rs` で検証する。

-----

## 記載すべき項目

### 1\. システム構成図（コンポーネント図）

``` mermaid
flowchart TD
    A["利用者 (Rust / JavaScript)"] --> B["OcrEngineBuilder<br>モデル・設定・辞書を<br>パス / ディレクトリ / バイト列で受け取る"]
    W["bindings/wasm<br>(wasm-bindgen)"] --> B
    B --> C["OcrEngine (Facade, Send + Sync)<br>run_from_path / run_from_image / run_from_bytes"]

    C --> O1["ページの向き分類 (任意)<br>orientation::OrientationClassifier"]
    O1 --> D["検出パイプライン"]
    C --> D
    D --> D1["DetPreProcessor<br>32 の倍数に引き伸ばし (imgproc)<br>BGR + ImageNet 正規化"]
    D1 --> D2["DetInferenceSession<br>(tract, 推論計画の LRU キャッシュ)"]
    D2 --> D3["DetPostProcessor::db_boxes<br>二値化 → 輪郭 → 最小面積矩形<br>→ 平均スコア → unclip → 矩形"]

    D3 --> R0["crop::crop_quad<br>透視変換で切り出し、縦長は 90° 回転"]
    R0 --> O2["行の上下分類 (任意)"]
    O2 --> R["認識パイプライン"]
    R0 --> R
    R --> R1["RecPreProcessor<br>高さ 48、可変幅、BGR、(x/255-0.5)/0.5"]
    R1 --> R2["RecInferenceSession<br>(バッチを rayon で並列実行)"]
    R2 --> R3["RecPostProcessor<br>CTC greedy デコード + 辞書"]

    Y["paddle_config<br>inference.yml の読み込み"] -.-> B
    T["threading<br>rayon プール / tract 実行器"] -.-> D2
    T -.-> R2
```

### 2\. モジュール間の関係

#### 2.1. データフロー

1.  **入力**: 利用者は、パス・`DynamicImage`・エンコード済み画像のバイト列のいずれかを渡す。
2.  **ページの向き（任意）**: `OrientationClassifier` が 0/90/180/270 度を判定し、画像を正立させる。結果の座標は、最後に元の画像の座標系へ戻す。
3.  **検出**:
    1.  `DetPreProcessor` が、長辺の上限（既定 960）または短辺の下限（PaddleOCR と同じ設定）に従って倍率を決める。
    2.  各辺を最も近い 32 の倍数に**引き伸ばし**、OpenCV 互換の bilinear でリサイズする。
    3.  BGR の順に並べ、ImageNet の平均・標準偏差で正規化し、NCHW のテンソルにする。
    4.  `DetInferenceSession` が DBNet を実行し、確率マップ `[H, W]` を得る。
    5.  `DetPostProcessor::db_boxes` が、PaddleOCR の `DBPostProcess` と同じ手順で、4 点の矩形とスコアを求める。
    6.  矩形の座標を、縦横それぞれの倍率で元の画像の座標に戻す。
4.  **切り出し**: `crop_quad` が矩形を透視変換（bicubic）で切り出す。高さ ÷ 幅 ≥ 1.5 の領域は縦書きとみなし、反時計回りに 90 度回転する。
5.  **行の上下（任意）**: 上下逆と判定された切り出しを 180 度回転する。
6.  **認識**:
    1.  切り出した画像を縦横比でソートし、`rec_batch_size`（既定 1）ごとのバッチに分ける。
    2.  バッチごとに、高さ 48 の可変幅へリサイズ・正規化する。
    3.  推論・デコードを、スレッドプールで並列に実行する。
7.  **出力**: `Vec<OcrResult { text, confidence, bounding_box }>` を返す。`run_with_metrics_*` の場合は、各ステージの所要時間とページの角度も返す。

#### 2.2. 処理シーケンス（主要ユースケース: `OcrEngine::run_from_path`）

1.  `App -> OcrEngineBuilder.det_model_dir(..).rec_model_dir(..).build()`
    * `inference.yml` を解釈し、ONNX を読み込む（中間テンソルの形状情報 `value_info` は破棄する）。
    * 辞書を作り、スレッドプールを生成する。
2.  `App -> OcrEngine.run_from_path(path)`
3.  `OcrEngine -> image::open` で画像を読み込む。
4.  ページの向き分類（任意）
5.  `DetectionPipeline`: 前処理 → 推論（形状ごとの推論計画をキャッシュ）→ `db_boxes` → 座標の変換
6.  `crop_regions` で切り出し、必要なら行の上下分類を行う。
7.  `RecognitionPipeline`: 前処理・推論・デコードの各ステージを、バッチ単位で並列に処理する。
8.  `OcrEngine -> App`: `Vec<OcrResult>` を返す。

### 3\. 設計原則・デザインパターン

  * **関心の分離**: C/C++ のライブラリが担っていた役割を、Pure Rust のクレートと自前のモジュールで置き換える（4 章を参照）。
  * **カプセル化**: `OcrEngine` は、モデル・辞書・設定・スレッドプール・推論計画のキャッシュをすべて内部に持つ。利用者は `run_*` を呼ぶだけでよい。
  * **Facade パターン**: `OcrEngine` が、各パイプラインに対する単一の窓口になる。
  * **Builder パターン**: `OcrEngineBuilder` で設定する。入力元（パス、PaddleOCR のモデルディレクトリ、バイト列）と、各種パラメータを組み合わせられる。
  * **PaddleOCR との一致**: 前処理と後処理は、PaddleX の実装に合わせる。実装の違いによる結果の差は、`tests/reference/*.json`（PaddleOCR 3.7 の出力）との比較テストで検出する。
  * **スレッド安全性**: 推論計画のキャッシュは `Mutex` で守り、`OcrEngine` を `Send + Sync` にしている。並列実行するバッチの中では、tract をシングルスレッドで動かす。tract の作業領域はスレッドごとに `RefCell` で持たれており、入れ子の並列化でワーカーが別の仕事を割り込ませると二重借用で panic するためである。

### 4\. 技術選定

| カテゴリ | 選定 | 理由・備考 |
| :--- | :--- | :--- |
| ONNX 推論 | `tract-onnx` 0.23 | Pure Rust の推論エンジン。0.20 では PP-OCRv6 medium の認識モデルを実行できない |
| 並列処理 | `rayon` + `tract-linalg/multithread-mm` | 認識バッチの並列化（既定は最大 16 スレッド）。機能 `multithread` で切り替える |
| N 次元配列 | `ndarray` 0.17 | tract と同じ版 |
| 画像の入出力 | `image` | デコードと切り抜き |
| リサンプリング | 自前の `imgproc` | `cv2.resize(INTER_LINEAR)` と同じ結果にするため |
| 輪郭検出・透視変換 | `imageproc` | `cv2.findContours` と `cv2.warpPerspective` の代替 |
| ポリゴンの膨張 | `i_overlay` | `pyclipper` の代替（角は丸め） |
| ジオメトリ | `geo-types` | `OcrResult::bounding_box` の型 |
| YAML | 自前の `paddle_config` | PyYAML が出力する範囲だけに対応し、依存を増やさない |
| ログ | `log` | 実際の出力先は、利用者側のロガーに任せる |
| WebAssembly | `wasm-bindgen`（`bindings/wasm`）、`web-time`、`getrandom/wasm_js` | ブラウザ対応。SIMD128 を有効にする |
