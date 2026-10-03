---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [wasm, api]
depends_on: task-wasm-001
---

# タスク概要
ファイルシステムのない環境（ブラウザ）でも使えるように、入力をすべてメモリ上のバイト列・テキストとして渡せるようにする。

## 実装メモ
- 低レベル API:
  - `DetInferenceSession::from_bytes`
  - `RecInferenceSession::from_bytes_with_input_height`
  - `OrientationClassifier::from_bytes(model, yaml)`
  - `RecDictionary::from_text` / `from_inference_yml_str`
  - 内部では tract の `model_for_read` を使い、value_info の破棄も同じように適用している。
- `OcrEngineBuilder`:
  - 追加したメソッド: `det_model_bytes`、`rec_model_bytes`、`det_config_yaml`、`rec_config_yaml`、`dictionary_text`、`doc_orientation_model_bytes`、`textline_orientation_model_bytes`。
  - パスとメモリ入力の両方を指定した場合は、後から指定したほうを優先する（もう一方はクリアされる）。
  - 辞書を決める優先順位は、`dictionary_text` → `dictionary_path` → `rec_config_yaml` → `rec_config_path`。
  - エラーメッセージでは、メモリ入力の場所を `<memory>` と表示する。
- `OcrEngine::run_from_bytes` / `run_with_metrics_from_bytes`: エンコードされた画像（PNG、JPEG など）をメモリ上でデコードして処理する。

## 破壊的変更
- `OcrEngine::det_model_path` / `rec_model_path` / `dictionary_path` は、`&Path` ではなく `Option<&Path>` を返すようになった。メモリから読み込んだ場合は `None` になる。

## 検証
- 統合テスト `in_memory_models_match_file_based_engine` を追加した。tiny のモデルと YAML をメモリから渡したエンジンの結果が、ファイルから読み込んだエンジンの結果と完全に一致することを確認する。パスの getter が `None` を返すことも確認する。
