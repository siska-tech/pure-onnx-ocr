# `ROADMAP_perf.md`

## 🎯 目標

CPU 推論（tract）の速度を、PaddleOCR 公式の CPU 推論（OpenVINO）に近づける。

## 📊 進捗

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-perf-001`](task-perf-001-multithread.md) | 推論のマルチスレッド化（認識バッチの並列実行）と、認識バッチサイズの既定値を 1 に変更 | パイプライン全体で 2.9〜4.7 倍速くなった。v6 small は 1.3 秒 |

## 🔭 Follow-ups

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 中 | 検出の並列化（タイル分割など） | 並列化後は、v6 tiny の処理時間の 60% 以上を検出が占めている |
| 低 | 行の向きの分類器も、バッチ単位で並列化する | 認識と同じ方法で並列化できる |
| 低 | ブラウザのマルチスレッド化（`wasm-bindgen-rayon`、tract の `RayonGlobal`） | COOP/COEP ヘッダが必要 |
| 低 | `ocr_smoke --benchmark` でも Windows の電力スロットリングを外す | ベンチマーク用の example では対応済み |
