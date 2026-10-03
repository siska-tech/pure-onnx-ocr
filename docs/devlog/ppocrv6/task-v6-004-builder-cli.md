---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, api, cli, batching]
depends_on: task-v6-003
---

# タスク概要
PaddleOCR 3.x の配布形態（モデルごとのディレクトリに `inference.onnx` と `inference.yml` を置く形）を、そのまま `OcrEngineBuilder` と `ocr_smoke` に渡せるようにする。あわせて、認識のバッチ処理を PaddleOCR と同じ方式に改める。

## 実装メモ

### `OcrEngineBuilder`
- `det_model_dir(dir)` / `rec_model_dir(dir)`
  - `dir/inference.onnx` をモデルとして使う。
  - `dir/inference.yml` があれば、設定として読み込む。
- `det_config_path` / `rec_config_path`: ONNX ファイルと YAML を個別に指定したい場合に使う。
- `dictionary_path` を省略した場合は、認識モデルの `inference.yml` に含まれる辞書を使う。
- `det_threshold` / `det_box_threshold` / `rec_use_space_char` を追加した。
- 設定ファイルの扱い:
  - 検出の YAML からは、色順と正規化パラメータを反映する。
  - 認識の YAML からは、色順と `image_shape` を反映する。
  - `PostProcess.name` が `DBPostProcess` / `CTCLabelDecode` 以外の場合は、`InvalidConfiguration` エラーにする。
  - しきい値は、パイプラインの既定値（0.3 / 0.6 / 1.5）を維持する。YAML の値は使わない（[research 3.3](research-ppocrv6.md)）。
- `OcrError::ModelConfig` を追加した。

### 認識のバッチ処理
- 変更前は、全領域を 1 バッチで推論しており、`rec_batch_size` を使っていなかった。
- 変更後は、領域を縦横比でソートし、`rec_batch_size` ずつ推論する。結果は元の順序に戻す。
- 幅の近い領域が同じバッチに入るので、可変幅にしてもパディングの無駄が少ない。

### `ocr_smoke`
- `--det-model-dir` / `--rec-model-dir` / `--det-thresh` / `--det-box-thresh` / `--no-space-char` を追加した。
- `--rec-model-dir` を指定し、`--dictionary` を省略した場合は、YAML の辞書を使う。

## 使用例
```bash
cargo run --release --bin ocr_smoke -- tests/fixtures/images/general_ocr_002.jpg \
  --det-model-dir tests/fixtures/models/ppocrv6/small_det \
  --rec-model-dir tests/fixtures/models/ppocrv6/small_rec --benchmark
```
