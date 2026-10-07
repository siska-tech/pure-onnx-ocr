# `ROADMAP_perf.md`

## 🎯 目標

CPU 推論（tract）の速度を、PaddleOCR 公式の CPU 推論（OpenVINO）に近づける。

[benchmark-openvino](benchmark-openvino.md)（2026-10-07）時点で、本クレートは同じ PC・同じ ONNX の OpenVINO の 1.8〜2.7 倍の時間がかかっている。差の 6〜9 割は検出の推論から来ている。

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
| `[ ]` | [`task-perf-003`](task-perf-003-default-threads.md) | 既定の推論スレッド数の上限を 8 から 16 に引き上げる | 実測で合計 −10〜16%（v6 small）。検出には効かない |
| `[ ]` | [`task-perf-004`](task-perf-004-det-profile.md) | 検出モデルの演算子ごとのプロファイルと、画像全体を使う演算子の確認 | 005 と 006 の方針を決める |
| `[ ]` | [`task-perf-005`](task-perf-005-det-tiling.md) | 検出のタイル分割（試作と影響の測定、採用してもオプトイン） | SE ブロックがあるので出力は一致しない |
| `[ ]` | [`task-perf-006`](task-perf-006-tract-intraop.md) | tract で、行列演算以外の演算子も演算子の中で並列化する（upstream への PR） | 出力は変わらない。本命だが時間がかかる |
| `[ ]` | [`task-perf-007`](task-perf-007-warmup.md) | 推論計画を事前にコンパイルする API（初回の実行を短縮） | 初回 −0.5〜1.9 秒の見込み |
| `[ ]` | [`task-perf-008`](task-perf-008-run-many.md) | 複数の画像をまとめて処理する API（スループット向け） | 画像をまたいで検出を並列に実行する |

進める順番:

1. task-perf-002 → task-perf-003（すぐ効き、リスクが低い）
2. task-perf-004 の結果を見て、task-perf-005（試作）と task-perf-006（tract への提案）を並行して進める。task-perf-007 もこの段階で進める。
3. task-perf-008

出力を変えないタスク（003、006、007、008）だけで見込める到達点:

- ウォームアップ後の時間は、small で OpenVINO の約 1.8〜2.0 倍。
- 初回の実行と、複数画像のスループットは、OpenVINO と同程度。

1 枚あたりの時間で 1.5 倍を切るには、task-perf-006 が tract に取り込まれるか、task-perf-005 をオプトインで使う必要がある。

## 🔭 Follow-ups

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 低 | task-perf-004 で特定した重い部分を、本クレート側のグラフの書き換えで置き換える | SE ブロック、`ConvTranspose` と `Resize` の組み合わせなどが重い場合 |
| 低 | 認識の幅を 64 刻みにまとめる、または上限を設ける | 計画のキャッシュの再利用率が上がる。ただしパディングの量が変わるので、出力への影響を測る必要がある |
| 低 | 推論計画ごとに重みのコピーを持つ問題を調べる | medium では、計画のキャッシュで約 1 GB を使っている |
| 低 | 行の向きの分類器も、バッチ単位で並列化する | 認識と同じ方法で並列化できる |
| 低 | ブラウザのマルチスレッド化（`wasm-bindgen-rayon`、tract の `RayonGlobal`） | COOP/COEP ヘッダが必要 |
| 低 | `ocr_smoke --benchmark` でも Windows の電力スロットリングを外す | ベンチマーク用の example では対応済み |
| 効果なし | `-C target-cpu=native` でビルドする | 実測で変化なし。tract の行列演算カーネルは、実行時に命令セットを選んでいる |
