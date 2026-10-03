# テスト仕様書 (Test Specification): Pure Rust OnnxOCR

作成者: Shion Watanabe  
初版: 2025-11-09  
改訂: 2026-10-03（v0.2.0）  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

## 🎯 目的

  * 品質を保証するためのテスト戦略と、実在するテストケースを定義する。
  * PaddleOCR 本体（Python）との出力の一致を、自動テストで継続的に確認する。
  * CI で、Linux と Windows、MSRV、WebAssembly のビルドを常に検証する。

-----

### 1\. テスト方針

  * **実施するテストの種類**
    1.  **単体テスト**（`src/**`）: 前処理・後処理、YAML の解析、辞書、CTC、矩形の算出と切り出し、リサイズ、計画キャッシュ、方向分類のラベル変換と座標の戻し。
    2.  **モデルを使う結合テスト**（`tests/*.rs`）: 実際の ONNX モデルと画像で、パイプライン全体を確認する。
    3.  **PaddleOCR との一致テスト**（`tests/paddle_parity.rs`）: PaddleOCR 3.7 の出力（`tests/reference/*.json`）と比較する。
    4.  **CLI テスト**（`tests/ocr_smoke.rs`）: `ocr_smoke --help`。
    5.  **ビルドの検証**（CI）: MSRV 1.91、`--no-default-features`、`wasm32-unknown-unknown`、`wasm32-wasip1`。
  * **フィクスチャがない場合**: 必要なモデルがないテストは、メッセージを出してスキップする。実行時間の長いもの（small / medium、傾き補正）は `#[ignore]` にしている。
  * **フレームワーク**: Rust 標準のテストハーネス。推論を含むため、`cargo test --release` を推奨する。

### 2\. テスト環境

  * **OS**: Linux（ubuntu-latest）、Windows（windows-latest、開発機は Windows 11）
  * **言語**: Rust stable（MSRV 1.91）
  * **テスト用アセット**（`scripts/fetch_fixtures.sh` で取得。`tests/fixtures/` は git 管理外）
    * 既定（約 35MB）:
      * PP-OCRv6 tiny の det / rec
      * PP-OCRv5 mobile の det / rec / 辞書（旧形式のファイル構成）
      * 方向分類器 2 種（`PP-LCNet_x1_0_doc_ori`、`PP-LCNet_x0_25_textline_ori`）
      * `images/general_ocr_002.jpg`
    * `--all` を付けると追加で取得する:
      * PP-OCRv6 small / medium
      * PP-OCRv5 mobile / server（ディレクトリ形式）
      * `PP-LCNet_x1_0_textline_ori`
      * `images/ja.jpg`
  * **PaddleOCR の参照データ**: uv の `.venv`（`scripts/requirements-reference.txt`）で `scripts/paddleocr_reference.py` を実行して作る。生成した JSON はコミットする。

### 3\. テストケース

#### 3.1. 技術検証 (PoC) テスト

| テスト | 内容 | 期待結果 |
| :--- | :--- | :--- |
| `tests::dbnet_dummy_inference_runs_successfully`（ignored） | `models/ppocrv5/det.onnx` をゼロ入力で推論する | 出力を得られる |
| `tests::svtr_dummy_inference_runs_successfully`（ignored） | `models/ppocrv5/rec.onnx` をゼロ入力で推論する | 出力の値が有限である |
| `detection::tests::detection_inference_runs` / `recognition::tests::recognition_inference_runs` | v5 の det / rec を実際の入力形状で推論する | 出力の形状が入力と整合する |

#### 3.2. 単体テスト (Unit Tests)

| 対象 | 主なテスト | 確認内容 |
| :--- | :--- | :--- |
| 検出前処理 | `resize_long_side_to_limit`、`keep_original_size_when_within_limit`、`detection_dims_round_to_nearest_multiple_of_32`、`min_limit_*`、`max_side_limit_caps_native_resolution`、`tensor_shape_and_normalization`、`detection_uses_bgr_channel_order_by_default` | 32 の倍数への丸め（偶数側への丸め）と引き伸ばし、縦横の倍率、ImageNet 正規化、BGR の順 |
| 認識前処理 | `recognition_*`（6 件） | 可変幅（1200px → 1216）、上限、パディングの値、範囲外・面積 0 のエラー |
| 検出後処理 | `extracts_single_square_contour`、`filters_small_regions`、`box_threshold_discards_low_confidence_regions`、`max_candidates_keeps_largest_regions`、`unclip_makes_polygon_larger`、`scaler_*` | 輪郭の抽出、スコアによる除外、膨張、座標の変換 |
| 切り出し | `crop::tests::*`（5 件） | 角の順序、30 度回転した矩形の復元、退化したケース、透視変換、縦長の回転 |
| リサイズ | `imgproc::tests::*`（2 件） | OpenCV の bilinear と同じ値になる（拡大・縮小） |
| YAML / 辞書 | `paddle_config::tests::*`（7 件）、`dictionary::*`（10 件） | クオートとエスケープ、U+3000、入れ子、アンカー、分類器の項目、space の追加、重複 |
| CTC | `ctc::tests::*`（5 件） | 重複とブランクの除去、信頼度、範囲外のクラス |
| その他 | `onnx_model::tests::*`、`orientation::tests::*`、`engine::thread_safety::engine_is_send_and_sync` | LRU キャッシュ、角度の変換、`Send + Sync` |

#### 3.3. 結合テスト (Integration Tests)

| テスト | 条件 | 期待結果 |
| :--- | :--- | :--- |
| `engine::tests::*` | PP-OCRv5 mobile（ファイル個別指定） | ビルドできること、異常系のエラー、白紙画像を処理できること、所要時間を報告すること |
| `integration_test::ocr_pipeline_smoke_test` | PP-OCRv5 mobile で搭乗券を処理する | `BOARDING` を含む結果が得られる |
| `integration_test::ocr_pipeline_reports_missing_image` / `ocr_builder_rejects_missing_models` | 存在しない画像やモデル | `Io` / `ImageDecode` / ビルドのエラー |
| `ppocrv6::detection_configs_*` / `recognition_configs_*` | v6 の YAML | BGR、ImageNet の値、辞書の件数（6,904 / 18,708） |
| `ppocrv6::recognition_class_count_matches_dictionary` | tiny_rec | 出力のクラス数が、blank + 辞書 + space と一致する |
| `ppocrv6::tiny_pipeline_reads_boarding_pass`（small / medium は ignored） | 搭乗券 | 主要な文字列と、空白を含む末尾の行を読める |
| `ppocrv6::model_config_postprocess_values_respect_explicit_overrides` | tiny の YAML | 既定値、YAML の値、明示的な指定の優先順位が正しい |
| `ppocrv6::rotated_crops_read_tilted_text`（ignored） | 10 度傾けた画像 | 回転補正ありの方が、正立画像の結果を多く再現する |
| `ppocrv6::doc_orientation_restores_rotated_pages` | 90 / 180 / 270 度回転したページ | 角度を正しく判定し、テキストを読める |
| `ppocrv6::textline_orientation_fixes_upside_down_lines` | 上下逆の画像 | 分類器を使うと主要な文字列を読める |
| `ppocrv6::in_memory_models_match_file_based_engine` | バイト列で渡す | ファイルから読み込んだ場合と結果が一致し、パスは `None` |
| `ppocrv6::engine_can_be_shared_between_threads` / `thread_count_does_not_change_results` | 3 スレッドから同時に使う / 1 スレッドと 4 スレッド | 結果が一致する |
| `ppocrv6::ppocrv5_yaml_dictionary_matches_text_dictionary` | v5 の YAML 辞書とテキスト辞書 | 全 18,383 件の順序が一致する |
| `paddle_parity::matches_paddleocr_reference_outputs` | 参照 JSON がある (モデル, 画像) の組 | 検出 F1 ≥ 0.90、文字類似度 ≥ 0.93（v6 tiny の日本語は文字の比較を除外） |
| `ocr_smoke::ocr_smoke_help_succeeds` | `--help` | 終了コード 0 で、使い方を表示する |

#### 3.4. 実行方法

```bash
scripts/fetch_fixtures.sh                    # 既定のテスト用（約 35MB）
cargo test --release --workspace             # CI と同じ
cargo test --release --test ppocrv6 -- --ignored            # small / medium / 傾き
cargo test --release --test paddle_parity -- --nocapture    # 本家との比較表を表示
```
