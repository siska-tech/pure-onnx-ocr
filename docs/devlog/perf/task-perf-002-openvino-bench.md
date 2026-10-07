---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-07
end_date: 2026-10-07
tags: [performance, openvino, benchmark, tooling]
depends_on: perf/task-perf-001-multithread
---

# タスク概要
OpenVINO との比較に使った計測ツール `tools/openvino-bench` と報告書 [benchmark-openvino](benchmark-openvino.md) を、本リポジトリに取り込む。以降の性能改善（task-perf-003 以降）は、すべてこのツールで効果を確かめる。

現在、ツールと報告書は別のクローン（`benchmark-vino-pure/pure-onnx-ocr`）にしかない。

## 背景
- 報告書では、本クレートは OpenVINO の 1.8〜2.7 倍の時間がかかった（同じ PC、同じ ONNX、同じ前処理と後処理）。
- 同じ条件でも、本クレートの絶対値は task-perf-001 の値より 4〜29% ぶれた。改善の判定には、交互に実行した計測の倍率を使う必要がある。

## 要件
- `tools/openvino-bench/`（`Cargo.toml`、`src/`、`run_bench.py`、`README.md`）を取り込む。`target/` と `results/` の生データは含めない。`results/summary.md` は報告書の根拠として残す。
- `docs/devlog/perf/benchmark-openvino.md` を取り込み、リンク切れがないことを確かめる。
- 本体のワークスペースのビルドと CI に、`openvino-bench` を巻き込まない（OpenVINO の C ライブラリが必要なため）。
- 変更前と変更後の本クレートを交互に計測する手順を README に書く（`git worktree` で 2 つのビルドを並べるなど）。

## 判定の基準（以降のタスクで共通）
- 変更前と変更後を、交互に 3 ラウンド以上実行する。
- 指標は、ウォームアップ後の中央値と、初回の実行時間。
- 出力（領域数、文字列、box の座標）が変わらないことを、同じ計測の中で確かめる。

## 作業ログ
- 2026-10-07: `tools/openvino-bench`（ソース、`Cargo.lock`、`README.md`、`run_bench.py`、`results/summary.{md,json}`）と `benchmark-openvino.md` を取り込んだ。
  - プロセスごとの生データ（`results/raw/`、`results/extra/`、約 2.8 MB）は含めず、`.gitignore` で除外した。報告書の「生データ」の記述を、集計結果へのリンクに直した。
  - Rust のソースに `cargo fmt` をかけた（中身は変えていない）。
- 2026-10-07: 本体のビルドとパッケージに影響しないようにした。
  - ツールは独自の `[workspace]` を持つので、本体のワークスペース（`cargo metadata` のメンバーは `pure-onnx-ocr` と `pure-onnx-ocr-wasm` だけ）にも CI（`--workspace`）にも入らない。
  - 本体の `Cargo.toml` の `exclude` に `tools/` を追加した。`cargo package --list` に `tools/` が含まれないことを確かめた。
- 2026-10-07: `run_bench.py` に A/B モードを追加した。
  - `--baseline-exe PATH`: 別のビルド（変更前のコミットの worktree でビルドしたもの）で `pure` を実行する `base` を追加し、基準にする。
  - `--add-config NAME="ARGS"`: 同じビルドで設定だけを変えた比較（例: `--threads 16`）。
  - A/B モードでは OpenVINO を計測せず、`openvino` パッケージも不要。summary.md に、基準に対する倍率の表（合計、det 推論、rec 推論、初回）が出る。
  - 出力の一致（Output parity）は、OpenVINO だけでなく、すべての設定を基準と比べるようにした。
  - 手順と判定の基準を README の「変更前と変更後の比較（A/B）」に書いた。

## テスト
- `cargo build --release`（tools/openvino-bench）が成功した。
- A/B モードの動作確認: v6 tiny、1 ラウンド、`--baseline-exe`（別クローンの de05070 のビルド）と `--add-config t16="--backend pure --threads 16"`。
  - 3 つの設定とも、領域数、文字列、box が基準と完全に一致した。
  - 同じコード（`base` と `pure`）でも、1 ラウンドでは合計で最大 1.37 倍の差が出た（ビルド直後で熱の影響もある）。1 ラウンドの値では判断できないので、判定には 3 ラウンド以上が必要である。
