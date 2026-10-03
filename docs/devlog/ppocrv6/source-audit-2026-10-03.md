# ソースコード監査・コメント整備（2026-10-03）

## 範囲

ローカルチェックアウトの OCR パイプラインを対象に、公開 API の契約、
テンソル形状、座標変換、CTC 復号、辞書、キャッシュと並列実行、関連する
単体テストおよび WASM 出力変換を確認した。依存パッケージの脆弱性調査や
全モデル・全入力に対する動作保証を目的とした監査ではない。

初回の変更では、既存のソースコメントに合わせて Rust の説明を英語で追記した。
その後、以下の 3 件について検証と修正を実施した。修正前の現象と対処は
各項目に記載する。

## 指摘事項

### 中: 認識領域の境界チェックで整数加算がオーバーフローする

対象: `src/preprocessing.rs` の `RecPreProcessor::process`。

`region.x + region.width` と `region.y + region.height` を `u32` のまま
加算している。例えば幅 10 の画像に対して `x = 1, width = u32::MAX`
を渡すと、開始座標の検査を通ったあとに加算がオーバーフローする。
オーバーフローチェック有効時は panic、無効時は折り返しにより本来の
`RegionOutOfBounds` 判定を通過し得る。公開 API を直接使う呼び出し側に影響する。

推奨: `checked_add` で失敗を領域外エラーにするか、開始座標を検査したあとに
`region.width > img_w - region.x` のように残り幅と比較する。
通常の領域外入力に加え、`u32::MAX` を含むケースを回帰テストに追加する。
検証・修正済み: 幅・高さそれぞれに `u32::MAX` を渡す回帰テストを追加し、
修正前に `attempt to add with overflow` の panic を再現した。開始座標の検査後に
残り幅・高さとの比較を行う形に変更し、両テストで `RegionOutOfBounds` が
返ることを確認。画像の右端・下端に接する正常な領域も受け入れるテストを追加した。

### 中: 検出モデル出力の空のバッチ／チャネルで panic し得る

対象: `src/detection.rs` の `DetInferenceSession::run`。

出力を四次元配列に変換したあと、最初の二軸を `index_axis(..., 0)` で
参照している。次元数は検証されるが、各軸の長さは検証されていない。
互換性のないモデルが `[0, 1, H, W]` または `[1, 0, H, W]` を返した場合、
エラーを返す API で panic する可能性がある。通常の対応 DBNet モデルでの
発生を確認したものではない。

推奨: 参照前に期待するバッチ数・チャネル数を検証し、形状を含む
`TractError` を返す。空軸の合成出力を使って検証する。

検証・修正済み: 出力変換を private ヘルパーに抽出し、合成テンソルを渡して
空バッチ・空チャネルの双方で `index_axis` の panic を再現した。
`[1, 1, H, W]` かつ `H > 0, W > 0` を参照前に検証するよう変更した。
不適合な次元数、複数バッチ／チャネル、空の空間軸は、形状を含むエラーを返す。
正常な 2×3 の出力について、軸順とスコア値を保持するテストも追加した。
以前は複数バッチ／チャネルの先頭だけを暗黙に取り出していたが、今後は対応する
単一画像・単一スコアマップ形式以外を明示的に拒否する。

### 低: CTC エラーテストがエラーの種類を検証していない

対象: `src/ctc.rs` の `error_when_blank_id_out_of_range`。

`matches!(...)` の戻り値を捨てているため、`expect_err` は失敗を確認するが、
エラー種別や `blank_id` / `class_count` が異なっていてもテストが通る。

推奨: `assert!(matches!(...))` に変更する。

検証・修正済み: `assert!(matches!(...))` に変更し、エラー種別と
`blank_id = 3, class_count = 2` を検証する既存テストが成功した。

## コメント修正内容

- CTC の繰り返し・blank の処理、有効時間長、確率／logit の判別、信頼度の算出を説明。
- 検出・認識テンソルの次元順、画像サイズの順序、padding と座標スケールを明示。
- キャッシュの LRU 順、容量ゼロの扱い、Arc による寿命、ロック外コンパイルを説明。
- パイプライン順序、設定の優先順位、幅順ソート後の結果復元、計測範囲を説明。
- 辞書の空白行の仕様を実装に合わせて訂正。空行は除外するが、空白文字のみの行は保持する。
- `config()` に付いていた実行メソッドの説明を修正し、`run_from_image()` に説明を追加。
- crate ルートの再公開コメントの対象ずれを修正。
- WASM の `polygon` は元の検出輪郭ではなく、エンジンが返す矩形の外周であることを明示。

## 初回コメント整備の検証

- `cargo fmt --all -- --check`: 成功。
- `git diff --check`: 成功。
- `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"` を設定して
  `cargo doc --offline --workspace --no-deps`: 成功。Rust/WASM 両クレートの
  API ドキュメントを生成し、ドキュメント内リンクの解決を確認。
- `cargo test --offline --lib -- --skip detection_inference_runs --skip recognition_inference_runs --skip engine_runs --skip run_pipeline`:
  67 件成功、失敗 0 件、既存の ignore 2 件、フィルターによる除外 2 件。
  実際に除外されたのは検出・認識の個別モデル推論テスト。
  エンジン経由の画像処理テストは実行され成功した。

統合テスト全体、ignored テスト、ブラウザーでの WASM 実行は今回未実施。
上記はコメント整備の検証結果であり、指摘した異常入力の安全性を保証しない。

## 不具合修正の検証

- 修正前: `cargo test --offline --lib overflow -- --nocapture` は追加した
  2 件が加算オーバーフローの panic で失敗。
- 修正前: `cargo test --offline --lib detection_output_empty -- --nocapture`
  は追加した 2 件が空軸へのアクセスの panic で失敗。
- 修正後: `cargo test --offline --lib` は 76 件成功、失敗 0 件、既存の
  ignored テスト 2 件。個別の検出・認識モデル推論テストも成功。
- `cargo test --offline --test integration_test -- --nocapture`: 3 件成功。
  PP-OCRv5 による実画像の `BOARDING` 認識と、画像・モデル欠損時のエラーを確認。
- `cargo test --offline --test ppocrv6 tiny_pipeline_reads_boarding_pass -- --nocapture`:
  1 件成功。PP-OCRv6 tiny による搭乗券の認識を確認。
- `cargo fmt --all -- --check` と `git diff --check`: 成功。

今回追加した回帰テストは 7 件。ignored テスト、PP-OCRv6 の残りの統合テスト、
PaddleOCR との全比較テストおよびブラウザー上の WASM 実行は未実施。
