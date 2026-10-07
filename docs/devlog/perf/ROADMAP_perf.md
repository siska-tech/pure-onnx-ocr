# `ROADMAP_perf.md`

## 🎯 目標

CPU 推論（tract）の速度を、PaddleOCR 公式の CPU 推論（OpenVINO）に近づける。

[benchmark-openvino](benchmark-openvino.md)（2026-10-07）時点で、本クレートは同じ PC・同じ ONNX の OpenVINO の 1.8〜2.7 倍の時間がかかっていた。差の 6〜9 割は検出の推論から来ていた。

[benchmark-openvino-throughput](benchmark-openvino-throughput.md)（2026-10-08、複数画像のスループット、両方とも最速の使い方）では、OpenVINO に対して tract 0.23.8 で 64〜81%、修正版の tract（task-perf-006）で 79〜99%（medium は同等）だった。メモリのピークは OpenVINO の 18〜66%。

| 指標（i7-1360P） | 現状 | 目標 |
| :--- | ---: | ---: |
| v6 small 搭乗券（ウォームアップ後） | 1.37 s（OpenVINO の 2.35 倍） | 0.9 s 以下（1.5 倍以下） |
| v6 medium 搭乗券（ウォームアップ後） | 4.99 s（2.02 倍） | 3.5 s 以下（1.4 倍以下） |
| v6 small 初回実行 | 2.26 s | 1.4 s 以下 |

方針:

- **既定の動作では、出力を変えない**。PaddleOCR と同じ出力を保つ。出力が変わりうる最適化（検出のタイル分割など）は、オプトインの設定にする。
- 効果は [tools/openvino-bench](../../../tools/openvino-bench/README.md) で、変更前と変更後を交互に実行して確かめる（task-perf-002）。
- メモリとスレッド数が OpenVINO より少ないという利点は、大きく損なわない。

## 📊 進捗

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-perf-001`](task-perf-001-multithread.md) | 推論のマルチスレッド化（認識バッチの並列実行）と、認識バッチサイズの既定値を 1 に変更 | パイプライン全体で 2.9〜4.7 倍速くなった。v6 small は 1.3 秒 |
| `[x]` | [`task-perf-002`](task-perf-002-openvino-bench.md) | OpenVINO 比較の計測ツールと報告書を取り込み、計測の手順を決める | A/B モード（`--baseline-exe`、`--add-config`）を追加した |
| `[x]` | [`task-perf-003`](task-perf-003-default-threads.md) | 既定の推論スレッド数の上限を 8 から 16 に引き上げる | 合計 −3〜18%（全モデル）、出力は一致。初回の実行中のメモリのピークが増えたが、007 で解消 |
| `[x]` | [`task-perf-004`](task-perf-004-det-profile.md) | 検出モデルの演算子ごとのプロファイルと、画像全体を使う演算子の確認 | depthwise が 8 スレッドでも速くならず、検出の 36〜50% を占める。全モデルに画像全体の平均がある |
| `[-]` | [`task-perf-005`](task-perf-005-det-tiling.md) | 検出のタイル分割（試作と影響の測定、採用してもオプトイン） | 見送り。006 の後では、4 分割でも全体の 6〜7% 程度しか縮まず、出力も変わる |
| `[~]` | [`task-perf-006`](task-perf-006-tract-intraop.md) | tract の depthwise 畳み込みの SIMD 化と、パックの並列化（upstream への PR） | 実装と計測は完了。tract main + 修正で合計 −29〜43%、出力は一致。PR: sonos/tract#2976、#2977、#2978（レビューとリリース待ち） |
| `[x]` | [`task-perf-007`](task-perf-007-warmup.md) | 推論計画を事前にコンパイルする API（`OcrEngine::warmup`）と、同じ計画の重複コンパイルの排除 | 重複コンパイルをなくし、初回 −10〜23%、ピークメモリは上限 8 の水準に戻った。warmup で初回がさらに −16〜33% |
| `[x]` | [`task-perf-008`](task-perf-008-run-many.md) | 複数の画像をまとめて処理する API（`run_many_from_images` / `run_many_from_paths`） | スループット 1.45〜2.06 倍、結果は 1 枚ずつと一致。tiny は OpenVINO（1 枚ずつ）を上回る |
| `[x]` | [`task-perf-009`](task-perf-009-rec-profile.md) | 認識モデルの演算子ごとのプロファイル | 認識の 66〜80% は行列演算。OpenVINO との残りの差は、認識の 1 スレッドの効率（1.1〜1.34 倍）でほぼ説明できる。本クレート側の対策は見送り |

進める順番:

1. task-perf-002 → task-perf-003 → task-perf-004（完了）
2. task-perf-007（完了）→ task-perf-006（実装と計測は完了。tract への PR とリリース待ち）。task-perf-005 は見送り。
3. task-perf-008（完了）

出力を変えないタスク（003、006、007、008）だけで見込める到達点:

- ウォームアップ後の時間は、small で OpenVINO の約 1.3 倍（task-perf-006 の実測。tract のリリース待ち）。
- 初回の実行と、複数画像のスループットは、OpenVINO と同程度。

task-perf-006 の実測（修正版の tract）では、small（搭乗券）は 0.77 秒で OpenVINO の約 1.32 倍、medium（搭乗券）は 2.74 秒で約 1.11 倍になった。1 枚あたりの時間の目標（1.5 倍以下）は、タイル分割なしで届く。本クレートに反映できるのは、tract の次のリリースの後である。

## 🔭 Follow-ups

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 見送り | 重い部分を、本クレート側のグラフの書き換えで置き換える | task-perf-004 で、重いのは depthwise そのものだと分かった。別の演算子に置き換えても速くならない |
| 低 | 認識の幅を 64 刻みにまとめる、または上限を設ける | 計画のキャッシュの再利用率が上がる。ただしパディングの量が変わるので、出力への影響を測る必要がある |
| 低 | 推論計画ごとに重みのコピーを持つ問題を調べる | medium では、計画のキャッシュで約 1 GB を使っている |
| 中 | `run_many` の同時に処理する画像の数を指定できるようにする | 今は推論スレッド数（16）と同じで、medium ではメモリのピークが 1 枚ずつの 3 倍（3.2 GB）になる（[benchmark-openvino-throughput](benchmark-openvino-throughput.md)） |
| 低 | tract の行列演算: 小さい `k` と `M` の効率（CTC の全結合層）とパックの削減（upstream） | task-perf-009。tiny と small の残り 2 割程度の差の主因 |
| 低 | 行の向きの分類器も、バッチ単位で並列化する | 認識と同じ方法で並列化できる |
| 低 | ブラウザのマルチスレッド化（`wasm-bindgen-rayon`、tract の `RayonGlobal`） | COOP/COEP ヘッダが必要 |
| 低 | `ocr_smoke --benchmark` でも Windows の電力スロットリングを外す | ベンチマーク用の example では対応済み |
| 効果なし | `-C target-cpu=native` でビルドする | 実測で変化なし。tract の行列演算カーネルは、実行時に命令セットを選んでいる |
