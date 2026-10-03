---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, dictionary, config, yaml]
depends_on: task-v6-001
---

# タスク概要
PP-OCRv6 には辞書テキストが同梱されておらず、辞書は `inference.yml` の `PostProcess.character_dict` に埋め込まれている。この YAML から辞書と前処理のパラメータを読み込めるようにする。

## 要件
- 依存クレートを増やさない（Pure Rust・小さい依存ツリーを維持する）。
- 辞書は YAML のシーケンスとして記述されている。クオートやエスケープを含むエントリ（`''''`、`'"'`、`\`、`':'` など）を正しく復元する。
- PaddleOCR の `use_space_char=True` と同じく、辞書の末尾に `" "` を追加できるようにする。

## 実装メモ
- `src/paddle_config.rs`（新規）
  - PyYAML が出力するブロック形式だけを扱う、最小限の YAML パーサを実装した。
    - ブロックマップ。
    - ブロックシーケンス。親キーと同じインデントの `- x` 形式と、入れ子の `- - x` 形式を含む。
    - 引用符なしのスカラーと、`'...'` / `"..."` のスカラー。`''` と `\uXXXX` などのエスケープに対応する。
    - アンカーとエイリアス。
  - フロー形式とブロックスカラーは、`PaddleConfigError::Syntax` で明示的に拒否する。
  - 空白として扱うのは半角スペースとタブだけにした。U+3000 などの Unicode 空白は、辞書のエントリとして保持する必要があるため。
  - `PaddleInferenceConfig` に、次の値を抽出する。
    - `model_name`
    - `img_mode`（→ `ColorOrder`）
    - `NormalizeImage.mean` / `NormalizeImage.std`
    - `RecResizeImg.image_shape`
    - `PostProcess.name` / `thresh` / `box_thresh` / `unclip_ratio` / `max_candidates`
    - `character_dict`
  - `1./255.` のような分数表記も数値として読める。
- `src/dictionary.rs`
  - `from_path` は拡張子が `.yml` / `.yaml` なら `from_inference_yml` を、それ以外なら `from_text_file`（従来の処理）を呼ぶ。
  - `from_tokens` を追加した。PaddleOCR は辞書を位置で参照するため、重複したエントリも許容する。`index_of` は最初に出現した位置を返す。
  - `with_space_char` を追加した。辞書の末尾に `" "` を追加する。
  - エラー型 `DictionaryError::Config` と `DictionaryError::MissingCharacterDict` を追加した。

## 検証
- 単体テストで、クオート・エスケープ・U+3000・入れ子シーケンス・アンカー・フロー形式の拒否を確認した。
- 実ファイルの辞書件数を確認した。tiny は 6,904、small と medium は 18,708。これに blank と space を加えた数が、ONNX の出力クラス数（6,906 / 18,710）と一致した。
- PP-OCRv5 の `rec.yml` から読んだ辞書と `ppocrv5_dict.txt` が、全 18,383 件で順序まで一致した（`tests/ppocrv6.rs`）。

## 補足
- 以前の実装は space クラスを追加していなかった。そのため、PP-OCRv5 でも空白の位置が `[UNK]` として出力されていた（例: `序号[UNK]SERIAL[UNK]NO.`）。
