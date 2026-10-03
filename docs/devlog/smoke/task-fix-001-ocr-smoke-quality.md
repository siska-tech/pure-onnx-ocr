---
status: completed
priority: high
assignee: Backend
start_date: 2025-11-10
end_date: 2026-10-03
tags: [quality, investigation, ocr-smoke]
depends_on: task-fix-000
---

# タスク概要
`ocr_smoke` CLI で観測されている OCR 結果の乱れを調査し、DBNet/SVTR パイプラインのどこで情報が欠落しているのか原因を特定する。

## 事前調査ログ
- `ocr_smoke` で生成されたログを確認したところ、DBNet の検出ポリゴンは妥当ながら、CTC デコード結果が `[UNK]` で埋まるケースが多い。
- `RecPostProcessorConfig::default()` が `blank_id = 0` を固定しており、PaddleOCR 辞書仕様（最終クラスをブランクとして利用）と不整合な点を `src/recognition.rs` で確認。
- `OcrEngine` ビルド時に辞書サイズからブランク ID を導出していないため、辞書の先頭エントリが常に空白扱いになり、正しいテキストが脱落している。
- `docs/test_specification_en.md` の仕様でも「辞書長と同じインデックスを blank として扱う」ことが記載されており、実装との差異が明確。
- `tests/fixtures/images/demo.png` を `ocr_smoke` で実行すると 16 領域が検出されるが、出力テキストは `《一…` やローマ字交じりのノイズばかりで信頼度 0.0。`ppocrv5_dict.txt` の実体は 18,383 エントリ（約 73KB）あり、中国語・ラテン文字・記号が混在するフルサイズ辞書であることを確認した。
- `rec.onnx` の出力クラス数（logits の最終次元）は 6,625 前後だが、辞書は 18,000 超と大きいため一見ミスマッチに見える。ただし PaddleOCR の辞書には重複や結合用トークンが含まれ、`RecPostProcessor` は辞書長ぶんをそのまま参照する設計なので、境界クラスの扱いを調査する必要がある。
- `RecDictionary::from_path` は `trim()` で空白行を除去する一方、先頭に配置された全角スペース行はトークンとして欠落するため、インデックスのずれが発生している可能性がある。`ppocrv5_dict.txt` の先頭数行（全角スペース / 一 / 乙 …）が CTC ロジットのインデックスと一致しているか検証する。
- `ocr_smoke` 出力に `[UNK]` は現れていないが、辞書に含まれる漢字記号がそのまま並んでいる。`demo.png` 自体も中国語テキスト主体であるため、言語ミスマッチが原因ではなく、SVTR モデルと辞書の組み合わせ（トークン順序やクラス数のズレ）が文字化けの主因と推測される。

## 次のアクション候補
- `RecDictionary` から `blank_id` を取得するヘルパを追加し、`RecPostProcessorConfig` のデフォルト値を更新する。
- 回帰を防ぐために、辞書長と CTC 出力クラス数の整合性を検証するユニットテストを追加する。
- フィクスチャ画像を使った `ocr_smoke` の統合テストを拡張し、代表的な文字列が正しく復元されるか確認する。
- PaddleOCR 純正の `ppocr_keys_v1.txt`（6,625 トークン構成）を利用した別辞書と比較し、`ppocrv5_dict.txt`（18,383 トークン）とのずれを可視化する。必要に応じて、モデルに合わせた辞書（同一順序）へ差し替える。
- `ocr_smoke` の CLI オプションに辞書差し替えを促す警告を追加し、不整合な辞書サイズの場合は起動時に警告またはエラーを発行する。
- 長期的には、小さな辞書でも動作させる場合は認識モデルを再学習するか、推論後に外部辞書マッピングを行う（現状はモデルと辞書をワンセットで扱う想定）。
- `RecDictionary::from_path` のトリム処理を見直し、先頭の全角/半角スペースをトークンとして保持するよう修正済み（テスト `preserves_leading_space_token` を追加）。`cargo test` は全て成功するが、`ocr_smoke` の結果は依然としてノイズで、辞書・モデル間のクラス数ギャップ（6,625 vs 18,383）に起因する可能性が残る。

## 進捗ログ (2025-11-10)
- `RecDictionary::from_path` で `blank` トークンを先頭に自動追加し、`blank_id = 0` を返す API (`blank_id`, `blank_token`) を実装。空辞書検知を再調整。
- CTC デコーダー (`ctc.rs`) と認識ポストプロセッサ (`recognition.rs`) のユニットテストを更新し、blank 除去と重複除去が PaddleOCR 仕様と一致することを確認。
- `OcrEngineBuilder` が辞書長ではなく `blank_id` を参照するよう修正し、`cargo fmt`, `cargo test` で全パスの回帰を確認済み。

## スモークテスト結果 (2025-11-10)
- `tests/fixtures/images/demo.png` を `ocr_smoke` で再実行。16 領域すべてが中国語ラベルや商品説明のテキストとして復元され、`《一…` ノイズは解消。出力例:
  - `纯臻营养护发素`
  - `产品信息/参数`
  - `【品名】：纯臻营养护发素`
  - `【主要功能】：可紧致头发磷层，从而达到`
  - `即时持久改善头发光泽的效果，给干燥的头`
- プロジェクトルートに追加した `sample1.jpg` / `sample2.jpg` でもスモークテストを実行。
  - `sample1.jpg`: `25.0B.05-Y`, `c`, `保存方法`
  - `sample2.jpg`: `` (空文字列), `25.06.09-Y`
- 現状の `Confidence` 表示はすべて `0.000` のまま。ロジット差分に対して素朴な `1 / Σexp(...)` を平均している暫定実装で、Softmax 正規化を行っていないため有効値になっていない。信頼度評価は別タスクで要改善。


## 再評価とクローズ (2026-10-03)

PP-OCRv6 対応（`docs/devlog/ppocrv6/`）の調査で、PaddleOCR 3.x の参照実装と前処理・後処理を比較した。その結果、ノイズの主因は次の 3 点と判明し、すべて修正した。

| 原因 | 症状 | 修正 |
| :--- | :--- | :--- |
| 検出入力が RGB で、`x/255` のみ。BGR と ImageNet 正規化が欠けていた | 確率マップが劣化し、`ccaa` のようなノイズ領域が出る | `task-v6-003`。BGR と ImageNet 正規化に加え、`box_thresh` と外側輪郭のみの抽出を導入 |
| 辞書の末尾に space クラスがなかった（PaddleOCR の `use_space_char=True` 相当） | 空白が `[UNK]` になる | `task-v6-002`。`RecDictionary::with_space_char` |
| 認識入力の幅を 320 に固定していた | 長い行が押し潰されて読めない | `task-v6-003`。幅を `max(320, 48 x 縦横比)`、上限 3200 の可変にした |

事前調査ログにある「6,625 クラスと 18,383 語の不一致」は、Hugging Face 配布の PP-OCRv5 mobile では発生しない。出力クラス数は 18,385 で、`ppocrv5_dict.txt` の 18,383 語に blank と space を足した数と一致する。

再評価の条件と結果は次のとおり。
- 画像は `demo.png` が手元にないため、PaddleX のデモ画像 `general_ocr_002.jpg`（搭乗券）を使った。
- モデルは PP-OCRv5 mobile。
- 実行コマンドは `ocr_smoke`。

| 項目 | 修正前 | 修正後 |
| :--- | :--- | :--- |
| ノイズ領域 | `ccaa` / `cYaaaananacl` / `caaa` / `ca` と空文字列で計 8 件 | 0 件 |
| 空白 | `序号[UNK]SERIAL[UNK]NO.` | `序号 SERIAL NO.` |
| 末尾の長い行 | 欠落 | `登机口于起飞前10分钟关闭 GATES CLOSE10MINUTES BEFORE DEPARTURE TIME` |
| 誤読の例 | `03DG`, `GATe`, `Am`, `ARE` | `03DEC`, `GATE`, `NAME`, `FARE` |

詳細は `docs/devlog/ppocrv6/task-v6-005-validation.md` を参照。本タスクはクローズとする。
