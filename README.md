# `pure-onnx-ocr` (Pure Rust OnnxOCR)

作成者: Shion Watanabe  
日付: 2025-11-09  
リポジトリ: http://github.com/siska-tech/pure-onnx-ocr

Pure RustでOCRパイプラインを構築するためのライブラリです。Baidu PaddleOCR 由来の検出モデル (DBNet) と認識モデル (CTC) を、Pure Rust エコシステムのみで実行できるよう再設計しています。**PP-OCRv5 と PP-OCRv6 (tiny / small / medium) の ONNX モデルに対応しています。**

> **English documentation is available in `README_en.md`.**  
> Other architectural documents also provide English counterparts (see [Documentation](#documentation)).

## 特長

- **Pure Rustのみで完結**: C/C++製オンプレミスライブラリやFFIの導入が不要です。`cargo build` だけでセットアップできます。
- **DBNet + CTC 認識パイプライン**: PaddleOCR が採用する検出・認識モデルを Rust 上で再現します。前処理 (BGR・ImageNet 正規化・可変幅認識) と後処理 (`box_thresh`・空白クラス) は PaddleOCR 3.x に合わせています。
- **PaddleOCR 3.x のモデルディレクトリをそのまま利用可能**: `inference.onnx` と `inference.yml` が入ったディレクトリを指定すると、前処理の設定と辞書を `inference.yml` から読み込みます。
- **モジュール構成が明確**: `OcrEngineBuilder` と `OcrEngine` を中心に、前処理・推論・後処理を分離しています。
- **移植性**: 組み込み環境、サーバーレス、WASMなど、C++依存が課題となる環境でも動作を想定しています。

## 導入手順

### 1. 前提条件

- Rust 1.91 以降 (stable。依存する `tract-onnx` 0.23 の要件)
- CPU推論を想定した x86\_64 / aarch64 環境
- PaddleOCR の ONNX モデル (PP-OCRv6 推奨。PP-OCRv5 も利用可)

### 2. 依存関係の追加

`Cargo.toml` の `[dependencies]` に以下を追加してください。

```toml
[dependencies]
pure_onnx_ocr = "0.1.0"         # crates.io リリース後に最新バージョンへ更新してください
image = "0.25"                  # OCR結果の描画や前処理に利用する場合
geo-types = "0.7"               # ポリゴン座標の操作に利用する場合
```

### 3. モデルの配置

#### PP-OCRv6 (推奨)

Hugging Face の `PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx` からモデルを取得します。各リポジトリの `inference.onnx` と `inference.yml` を、同じディレクトリに配置してください。辞書は `inference.yml` に含まれているため、別途用意する必要はありません。

```bash
for kind in det rec; do
  mkdir -p models/ppocrv6/small_${kind}
  for f in inference.onnx inference.yml; do
    curl -L -o models/ppocrv6/small_${kind}/${f} \
      https://huggingface.co/PaddlePaddle/PP-OCRv6_small_${kind}_onnx/resolve/main/${f}
  done
done
```

| 階層 | 特徴 | CPU 推論時間の目安 (896x528 の画像 1 枚) |
| :--- | :--- | :--- |
| `tiny` | 最軽量。辞書は 6,904 文字で、**ひらがな・カタカナを含まないため日本語には不向き** | 約 2.3 秒 |
| `small` | 50 言語 (日本語を含む)。精度と速度のバランスが良い | 約 6.5〜8 秒 |
| `medium` | 50 言語。最高精度 (PaddleOCR 3.x の既定) | 約 21〜26 秒 |

#### PP-OCRv5

従来どおり、`det.onnx`、`rec.onnx`、`ppocrv5_dict.txt` を個別に指定できます。Hugging Face の `PaddlePaddle/PP-OCRv5_{mobile,server}_{det,rec}_onnx` のようにディレクトリ単位で配布されているモデルは、PP-OCRv6 と同じ方法で指定できます。

## クイックスタート

PP-OCRv6 のモデルディレクトリを使う例:

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError};

fn main() -> Result<(), OcrError> {
    let engine = OcrEngineBuilder::new()
        .det_model_dir("models/ppocrv6/small_det") // inference.onnx + inference.yml
        .rec_model_dir("models/ppocrv6/small_rec") // 辞書は inference.yml から読み込む
        .build()?;

    for result in engine.run_from_path("examples/demo.jpg")? {
        println!("{} ({:.3})", result.text, result.confidence);
    }
    Ok(())
}
```

ファイルを個別に指定する例 (PP-OCRv5 のテキスト辞書):

```rust
use pure_onnx_ocr::{OcrEngineBuilder, OcrError, OcrResult};

fn main() -> Result<(), OcrError> {
    // 1. エンジンの初期化（アプリケーション起動時に一度だけ実行）
    let engine = OcrEngineBuilder::new()
        .det_model_path("models/ppocrv5/det.onnx")
        .rec_model_path("models/ppocrv5/rec.onnx")
        .dictionary_path("models/ppocrv5/ppocrv5_dict.txt")
        .det_limit_side_len(960)   // 任意調整: 入力画像の最大長辺
        .det_unclip_ratio(1.5)     // 任意調整: 検出ポリゴンのオフセット率
        .rec_batch_size(8)         // 任意調整: 認識推論のバッチサイズ
        .det_box_threshold(0.6)    // 任意調整: 検出領域の平均スコア下限 (PaddleOCR の box_thresh)
        .build()?;

    // 2. 画像ファイルからOCRを実行
    let results: Vec<OcrResult> = engine.run_from_path("examples/demo.jpg")?;

    // 3. OCR結果を活用
    for (idx, result) in results.iter().enumerate() {
        println!("#{} text={} confidence={:.4}", idx, result.text, result.confidence);
        println!("   polygon={:?}", result.bounding_box.exterior().points());
    }

    Ok(())
}
```

## 動作確認バイナリ `ocr_smoke`

`test_ocr.py` に相当する動作確認を Rust のみで実施したい場合は、付属の `ocr_smoke` バイナリを利用できます。

- 既定で `models/ppocrv5` 配下の `det.onnx`, `rec.onnx`, `ppocrv5_dict.txt` を参照します。
- 使い方:

```bash
cargo run --bin ocr_smoke -- path/to/image.jpg

# モデルや設定を上書きする例
cargo run --bin ocr_smoke -- path/to/image.jpg \
  --det-model models/ppocrv5/det.onnx \
  --rec-model models/ppocrv5/rec.onnx \
  --dictionary models/ppocrv5/ppocrv5_dict.txt \
  --det-limit-side-len 960 \
  --det-unclip-ratio 1.5 \
  --rec-batch-size 8
```

PP-OCRv6 のモデルディレクトリを使う場合は、次のように指定します。

```bash
cargo run --release --bin ocr_smoke -- path/to/image.jpg \
  --det-model-dir models/ppocrv6/small_det \
  --rec-model-dir models/ppocrv6/small_rec

# しきい値を調整する例 (既定値は PaddleOCR 3.x パイプラインと同じ 0.3 / 0.6)
cargo run --release --bin ocr_smoke -- path/to/image.jpg \
  --det-model-dir models/ppocrv6/small_det \
  --rec-model-dir models/ppocrv6/small_rec \
  --det-thresh 0.3 --det-box-thresh 0.6
```

ベンチマーク用途では `--benchmark` フラグを付与します。総時間・画像デコード・DBNet / SVTR の各ステージ（前処理・推論・後処理）が `[INFO] benchmark.*` 形式で出力され、既存のテキスト出力と併置されます。

```bash
cargo run --bin ocr_smoke -- path/to/image.jpg --benchmark

[INFO] benchmark.image=tests/fixtures/images/demo.png
[INFO] benchmark.total_seconds=0.412583
[INFO] benchmark.image_decode_seconds=0.003121
[INFO] benchmark.det.preprocess_seconds=0.044512
[INFO] benchmark.det.inference_seconds=0.221009
[INFO] benchmark.det.postprocess_seconds=0.012334
[INFO] benchmark.rec.preprocess_seconds=0.018775
[INFO] benchmark.rec.inference_seconds=0.094281
[INFO] benchmark.rec.postprocess_seconds=0.005237
```

推論時間、検出されたテキストと信頼度、ポリゴン座標が標準出力に整形されます。入力画像やモデルが見つからない場合はエラーメッセージと共に終了します。

内部では、検出前処理で次の順に処理しています。

1. 長辺を指定サイズにリサイズする
2. BGR 順で ImageNet 正規化する
3. 32px 単位でパディングして、DBNet の入力制約（32 の倍数）を満たす

認識前処理では、高さ 48px のまま縦横比を保ってリサイズします。行が長い場合は入力幅を最大 3200px まで広げ、縦横比でソートしたうえでバッチ化します。

### PaddleOCR 3.x 相当のオプション

| 機能 | ビルダー | `ocr_smoke` | 既定 |
| :--- | :--- | :--- | :--- |
| 回転補正つきの切り出し（縦長の領域は 90° 回転） | `rec_crop_mode(RecCropMode::Rotated)` | `--crop-mode rotated\|axis` | 有効 |
| 原寸での検出（PaddleOCR 3.x と同じ条件） | `det_limit_type(DetLimitType::Min).det_limit_side_len(64)` | `--det-limit-type min --det-limit-side-len 64` | 長辺 960 に縮小 |
| 検出 YAML のしきい値を使う | `det_postprocess_from_model_config(true)` | `--det-params-from-config` | パイプラインの既定値 (0.3 / 0.6 / 1.5) |
| ページの向き補正（0/90/180/270） | `doc_orientation_model_dir("models/PP-LCNet_x1_0_doc_ori")` | `--doc-ori-model-dir DIR` | 無効 |
| 行の上下補正（0/180） | `textline_orientation_model_dir("models/PP-LCNet_x0_25_textline_ori")` | `--textline-ori-model-dir DIR` | 無効 |
| 推論計画のキャッシュ上限 | `plan_cache_capacity(4, 16)` | なし | 検出 4 / 認識 16 |
| 読み込み・推論ログ | `log` クレートで出力 | `-v` / `--verbose` | Warn 以上のみ |

向きの分類器は、Hugging Face の `PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx` と `PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx` から取得します。行の向きの分類器は `x1_0` 版もありますが、tract 上では `x0_25` 版のほうが約 3 倍速いため、こちらを推奨します。

> **既知の制約:**
> - PP-OCRv6 medium は CPU (tract) 上で 1 枚あたり 20 秒前後かかります。速度を優先する場合は tiny / small を推奨します。処理時間はマシンの負荷によって大きく変動します。
> - 行の上下補正は、短い大文字だけの行（`TAIYUAN` など）で判定を誤ることがあります。
> - 文書の歪み補正（UVDoc）とレイアウト解析には対応していません。
>
> 調査の経緯と検討内容は `docs/devlog/ppocrv6/` を参照してください。

### よくあるエラー

- `ModelLoad`: `tract` が未対応のONNXオペレータ（例: `LayerNormalization`, `Scan`）を検出した場合に発生します。
- `ModelConfig`: `inference.yml` を解析できない場合に発生します。PaddleOCR が出力した YAML（ブロック形式）にのみ対応しています。
- `Dictionary`: 辞書ファイルの文字コードがUTF-8以外の場合に発生します。UTF-8 (BOM無し) で保存してください。

## API概要

`pure_onnx_ocr` クレートは次の構造体・列挙型を公開します。

| シンボル           | 概要                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `OcrEngineBuilder` | モデル・辞書・パラメータを設定し、`OcrEngine` を構築するためのビルダー。`det_model_dir` / `rec_model_dir` で PaddleOCR のモデルディレクトリを指定できます。 |
| `OcrEngine`        | 検出・認識パイプラインを統合したファサード。`run_from_path` と `run_from_image` を提供します。 |
| `OcrRunWithMetrics`| OCR 実行結果とステージ別メトリクス (`OcrTimings`) をまとめて返すヘルパー構造体。               |
| `OcrTimings`       | 全体時間・画像デコード時間・DBNet / SVTR の各ステージ時間を集約したメトリクス。                 |
| `StageTimings`     | 個別ステージ（前処理・推論・後処理）の所要時間を表すユーティリティ。                            |
| `OcrResult`        | 認識された単一テキスト領域の結果 (`text`, `confidence`, `bounding_box`) を保持します。         |
| `OcrError`         | ライブラリ全体で発生し得るエラーをカプセル化した列挙型です。                                   |
| `Polygon`          | `geo-types::Polygon` の再エクスポート。検出結果の座標表現に利用します。                        |
| `PaddleInferenceConfig` | PaddleOCR の `inference.yml` から、前処理パラメータ・しきい値・辞書を読み取ります。 |

詳細なAPI仕様については `docs/interface_design.md` および `docs/interface_design_en.md` を参照してください。

## Documentation

- `docs/architecture.md` / `docs/architecture_en.md`
- `docs/detail_design.md` / `docs/detail_design_en.md`
- `docs/interface_design.md` / `docs/interface_design_en.md`
- `docs/requirements.md` / `docs/requirements_en.md`
- `docs/references.md` / `docs/references_en.md`
- `docs/test_specification.md` / `docs/test_specification_en.md`

ドキュメントセット全体の英語版を整備し、国際的なコントリビューターでも参照可能な構成としています。

## 開発進捗

- 2025-11-09: `tract-onnx` を用いた `det.onnx` (DBNet) のロードとダミー推論 PoC (`task-poc-001`) を完了。
- 2025-11-09: `rec.onnx` (SVTR_HGNet) のロードとダミー推論 PoC (`task-poc-002`) を完了。出力形状 `[1, 40, 18385]` を確認。
- 2025-11-09: 検出前処理 `DetPreProcessor` (`task-det-001`) を実装。長辺制限リサイズ、正規化、NCHW変換に対応。
- 2025-11-09: DBNet 推論モジュール `DetInferenceSession` (`task-det-002`) を実装。解像度別にランナブルをキャッシュ。
- 2025-11-09: 検出後処理 `DetPostProcessor` (`task-det-003`) を実装。閾値処理と輪郭抽出を追加。
- 2025-11-09: ポリゴン拡張 `DetPolygonUnclipper` (`task-det-004`) を実装。`i_overlay` を利用したバッファリングを実現。
- 2025-11-09: 座標復元 `DetPolygonScaler` (`task-det-005`) を実装。逆スケーリングと丸め処理を追加。
- 2025-11-09: 認識前処理 `RecPreProcessor` (`task-rec-001`) を実装。クロップ、強制リサイズ、バッチ化を統合。
- 2025-11-09: 認識推論 `RecInferenceSession` (`task-rec-002`) を実装。`tract-onnx` によるバッチ推論を整備。
- 2025-11-09: 辞書ローダー `RecDictionary` (`task-rec-003`) を実装。重複検知やマッピングを追加。
- 2025-11-09: CTC Greedy デコーダー `CtcGreedyDecoder` (`task-rec-004`) を実装。重複圧縮とブランク除去をサポート。
- 2025-11-09: 認識ポストプロセッサ `RecPostProcessor` (`task-rec-005`) を実装。ロジット処理と辞書マッピングを統合。
- 2025-11-09: 公開ビルダー `OcrEngineBuilder` (`task-api-001`) を実装。パラメータ検証とモデル初期化を実装。
- 2025-11-09: `OcrEngine` ファサード (`task-api-002`) を実装。検出・認識パイプラインを統合。
- 2025-11-09: `OcrEngine::run_from_path` (`task-api-003`) を実装し、E2E OCR処理を完成。
- 2025-11-09: `OcrEngine::run_from_image` (`task-api-004`) を実装し、メモリ上の画像入力に対応。
- 2025-11-09: 公開エラー型 `OcrError` (`task-api-005`) を整備し、エラーパスを統一。
- 2025-11-09: ドキュメント整備タスク `task-doc-001` を完了。READMEの再構成と英語版ドキュメントを追加。
- 2025-11-09: 公開APIの Rustdoc コメント (`task-doc-002`) を整備し、`cargo doc` で生成物を確認。
- 2025-11-09: Cargo メタデータ (`task-doc-003`) を整備し、`cargo package --no-verify` で公開準備を確認。
- 2025-11-09: 結合テスト (`task-doc-004`) を追加し、フィクスチャ設計と CI 実行手順を文書化。
- 2025-11-10: `task-fix-001` で `RecDictionary` に blank トークンを追加し、`OcrEngineBuilder` と CTC デコーダーが PaddleOCR の仕様 (`blank_id = 0`) と一致するように修正。
- 2025-11-10: `task-fix-002` で認識信頼度を「確率出力を検出して最大値を直接集計し、ロジット出力は log-sum-exp で Softmax 後に算術平均化する」方式へ刷新し、`ocr_smoke` の信頼度出力が実測値を反映するよう改善。
- 2025-11-10: `task-fix-003` で `ocr_smoke` に `--benchmark` 計測フラグと `OcrEngine::run_with_metrics_*` API を追加し、主要ステージの所要時間を取得可能にした。

## コントリビューション

Pull Request や Issue を歓迎します。大規模な変更を提案する場合は、まず Issue で背景と目的を共有してください。

### 開発フローの指針

- `cargo fmt` と `cargo clippy` でスタイルと静的解析を行ってから PR を作成してください。
- 追加した機能には可能な限りユニットテストを付与してください。
- ドキュメント更新の場合は `docs/devlog` と関連タスクの進捗を同期してください。

## ライセンス

本プロジェクトは `Apache-2.0` ライセンスで提供します。リファレンス実装である `PaddleOCR`, `OnnxOCR`, `tract` と同一ファミリーのライセンス体系に準拠します。

## テスト

- ユニットテスト: `cargo test`
- PP-OCRv6 テスト (`tests/ppocrv6.rs`): tiny のパイプラインは既定で実行されます。small と medium は `cargo test --release --test ppocrv6 -- --ignored` で実行します。
- 結合テスト: PP-OCRv5 モデルとテスト画像を `PURE_ONNX_OCR_FIXTURE_DIR` または `tests/fixtures/` に配置してください。フィクスチャが見つからない場合、テストは自動的にスキップされます。必要なパス構成は `tests/fixtures/README.md` を参照してください。