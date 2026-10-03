# インターフェース設計書 (API設計書): Pure Rust OnnxOCR

作成者: Shion Watanabe  
初版: 2025-11-09  
改訂: 2026-10-03（v0.2.0）  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * 利用者（アプリケーション開発者）が使う公開 API の契約（型、メソッド、エラー、既定値）を定義する。
  * v0.2.0 で追加・変更した API を明記する。主な追加は、PP-OCRv6 のモデルディレクトリ、メモリからの入力、方向分類器、スレッド数の設定である。
  * 細かいシグネチャは rustdoc（`cargo doc`）を正とする。本書では、設計上の意図と既定値をまとめる。

-----

## 記載すべき項目

### 1\. 公開API一覧

#### 名前空間 / モジュール

クレート名は `pure_onnx_ocr`。主要な型はクレートのルートに再エクスポートしている。

| モジュール | 役割 |
| :--- | :--- |
| `engine` | `OcrEngineBuilder`、`OcrEngine`、`OcrResult`、`OcrError`、計測値の型 |
| `preprocessing` / `postprocessing` | 検出・認識の前処理と後処理（`DetPreProcessor`、`DetPostProcessor::db_boxes`、`RecPreProcessor` など） |
| `detection` / `recognition` / `ctc` | tract による推論セッション、CTC デコード |
| `dictionary` | `RecDictionary`（テキストまたは `inference.yml` から読み込む） |
| `paddle_config` | `PaddleInferenceConfig`（`inference.yml` の読み込み） |
| `crop` | 最小面積矩形の算出と、透視変換による切り出し |
| `orientation` | ページの向き・行の上下の分類器 |
| `imgproc` | OpenCV 互換の bilinear リサイズ |

#### クラス（Struct）と公開メソッド

| 型 | 主なメソッド |
| :--- | :--- |
| `OcrEngineBuilder` | `new`、モデルの指定（後述）、パラメータの指定（後述）、`build` |
| `OcrEngine` | `run_from_path`、`run_from_image`、`run_from_bytes`、`run_with_metrics_from_{path,image,bytes}`、`config`、`det_model_path`、`rec_model_path`、`dictionary_path`、`rec_batch_size` |
| `OcrResult` | `text: String`、`confidence: f32`、`bounding_box: Polygon<f64>` |
| `OcrRunWithMetrics` | `results`、`timings: OcrTimings`、`doc_orientation_angle: Option<u32>` |
| `OcrTimings` / `StageTimings` | 全体・画像デコード・方向分類・検出と認識の各ステージ（前処理、推論、後処理）の所要時間 |
| `OcrEngineConfig` | エンジンが実際に使っている設定（`OcrEngine::config()` で参照する） |
| `PaddleInferenceConfig` | `from_path`、`from_yaml_str` |
| `RecDictionary` | `from_path`、`from_text`、`from_inference_yml(_str)`、`from_tokens`、`with_space_char` |
| `OrientationClassifier` | `from_model_dir`、`from_bytes`、`classify` |

#### 公開する列挙型（Enum）

| 型 | 値 |
| :--- | :--- |
| `OcrError` | 2 章を参照 |
| `DetLimitType` | `Max`（長辺の上限、既定）/ `Min`（短辺の下限。PaddleOCR 3.x と同じ） |
| `RecCropMode` | `Rotated`（既定。透視変換で切り出す）/ `AxisAligned`（外接矩形で切り出す） |
| `ColorOrder` | `Rgb` / `Bgr` |

#### 公開する定数

| 定数 | 値 |
| :--- | :--- |
| `PADDLE_MODEL_FILE` / `PADDLE_CONFIG_FILE` | `"inference.onnx"` / `"inference.yml"` |
| `IMAGENET_MEAN` / `IMAGENET_STD` | 検出の正規化パラメータ |
| `MULTITHREAD_SUPPORTED` | このビルドでマルチスレッド推論が使えるか（機能 `multithread` が有効で、WebAssembly 以外の場合に true） |

#### 再エクスポート (Re-exports)

`geo_types::{Point, Polygon}`、主要な型（`OcrEngineBuilder`、`OcrEngine`、`OcrResult`、`OcrError`、`DetLimitType`、`RecCropMode`、`PaddleInferenceConfig`、`OrientationClassifier` など）、関数（`min_area_quad`、`crop_quad`、`default_inference_threads`）。

### 2\. 各APIの詳細定義

#### `OcrError` (Enum)

| バリアント | 発生条件 |
| :--- | :--- |
| `MissingField { field }` | 検出モデル・認識モデル・辞書のいずれかが指定されていない |
| `Io { source, path }` | 指定したファイルが存在しない、または読めない |
| `ModelLoad { source, path }` | ONNX の読み込み・解析に失敗した（メモリ入力の場合、`path` は `<memory>`） |
| `ModelConfig { source, path }` | `inference.yml` を解析できない |
| `Dictionary { source }` | 辞書が空、重複がある、`character_dict` がない |
| `OrientationLoad { source, path }` / `OrientationInference { source }` | 方向分類器の読み込み・推論に失敗した |
| `InvalidConfiguration { message }` | `rec_batch_size == 0`、対応していない後処理名（`DBPostProcess` / `CTCLabelDecode` 以外）、`image_shape` が不正 |
| `ImageDecode { source, path }` | 画像をデコードできない |
| `DetectionPreprocess` / `DetectionInference` / `DetectionPostProcess` | 検出の各ステージで失敗した |
| `RecognitionPreprocess` / `RecognitionInference` / `RecognitionPostProcess` | 認識の各ステージで失敗した |
| `PipelineMismatch` | 検出数と認識結果の数が一致しない（内部の不整合） |

#### `OcrResult` (Struct)

* `text`: 認識した文字列。空白は、辞書の末尾に追加した space クラスから出力される。
* `confidence`: CTC で選ばれた各文字の確率の平均（softmax 後）。0 から 1。
* `bounding_box`: 入力画像の座標系での最小面積矩形。4 点 `tl, tr, br, bl` の閉じたリングで、PaddleOCR の `DBPostProcess` と同じく整数に丸め、画像内に収めてある。ページの向き補正を行った場合も、元の入力画像の座標で返す。

#### `OcrEngineBuilder` (Struct)

**モデルの指定**（パスとメモリ入力のうち、後から呼んだ方が優先される）

| メソッド | 内容 |
| :--- | :--- |
| `det_model_dir(dir)` / `rec_model_dir(dir)` | PaddleOCR 3.x のモデルディレクトリ（`inference.onnx` + `inference.yml`）。認識モデルの YAML に含まれる辞書も使う |
| `det_model_path` / `rec_model_path` / `dictionary_path` | ファイルを個別に指定する。辞書はテキストまたは `.yml` |
| `det_config_path` / `rec_config_path` | `inference.yml` を個別に指定する |
| `det_model_bytes` / `rec_model_bytes` / `det_config_yaml` / `rec_config_yaml` / `dictionary_text` | メモリから渡す（ブラウザなど） |
| `doc_orientation_model_dir` / `_bytes` | ページの向きの分類器（任意） |
| `textline_orientation_model_dir` / `_bytes` | 行の上下の分類器（任意） |

**パラメータ**

| メソッド | 既定値 | 内容 |
| :--- | :--- | :--- |
| `det_limit_side_len` | 960 | 検出入力のサイズ制限 |
| `det_limit_type` | `Max` | `Min` + 64 にすると、PaddleOCR 3.x と同じ原寸検出になる |
| `det_max_side_limit` | 4000 | 長辺の絶対上限 |
| `det_threshold` / `det_box_threshold` / `det_unclip_ratio` | 0.3 / 0.6 / 1.5 | PaddleOCR パイプラインの既定値 |
| `det_postprocess_from_model_config` | false | 検出 YAML のしきい値を使う（明示的に指定した値が優先） |
| `rec_batch_size` | 1 | tract では 1 が最速 |
| `rec_use_space_char` | true | 辞書の末尾に `" "` を追加する |
| `rec_crop_mode` | `Rotated` | 切り出しの方法 |
| `inference_threads` | 論理 CPU 数（最大 8） | WebAssembly では 1 |
| `plan_cache_capacity(det, rec)` | 4, 16 | 推論計画（入力形状ごと）のキャッシュ数 |

`build()` は、ファイルの存在確認、YAML の解釈、ONNX の読み込み、辞書の構築、スレッドプールの生成を行う。

#### `OcrEngine` (Struct)

* `run_from_path(path)` / `run_from_image(&DynamicImage)` / `run_from_bytes(&[u8])` は、`Vec<OcrResult>` を返す。
* `run_with_metrics_*` は、`OcrRunWithMetrics` を返す（所要時間、ページの角度）。
* 処理は同期的に実行する。`OcrEngine` は `Send + Sync` で、`Arc` で包めば複数スレッドから同時に使える。
* `config()` は、実際に使っている `OcrEngineConfig`（YAML を反映した後の値）を返す。
* `det_model_path()` などは `Option<&Path>` を返す（メモリ入力の場合は `None`）。

### 3\. 使用例（Code Snippet）

#### `Cargo.toml`

```toml
[dependencies]
pure_onnx_ocr = "0.2"
# シングルスレッドにしたい場合:
# pure_onnx_ocr = { version = "0.2", default-features = false }
```

#### `src/main.rs`

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError};

fn main() -> Result<(), OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_dir("models/ppocrv6/small_det")
        .rec_model_dir("models/ppocrv6/small_rec")
        // 任意: .textline_orientation_model_dir("models/PP-LCNet_x0_25_textline_ori")
        .build()?;

    let run = engine.run_with_metrics_from_path("receipt.jpg")?;
    for result in &run.results {
        let corners: Vec<_> = result.bounding_box.exterior().points().take(4).collect();
        println!("{} ({:.3}) {:?}", result.text, result.confidence, corners);
    }
    println!("total {:?}", run.timings.total);
    Ok(())
}
```

JavaScript から使う場合は、`bindings/wasm` と `examples/web` を参照。
