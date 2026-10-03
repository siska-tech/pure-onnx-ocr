# `ROADMAP_wasm.md`

## 🎯 目標

`pure-onnx-ocr` をブラウザ（`wasm32-unknown-unknown`）で実際に動かせる状態にする。PP-OCRv6 と、2 つの方向分類器を含む。

調査と方針の検討は [research-wasm.md](research-wasm.md) にまとめた。

## 📊 進捗

ブランチ: `feature/wasm-browser`

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-wasm-001`](task-wasm-001-build-and-clock.md) | ブラウザ向けビルドの修正（getrandom の `wasm_js`）と、`Instant` によるパニックの解消（`web-time`） | v6 対応で壊れたビルドと、develop の時点から存在した実行時の問題 |
| `[x]` | [`task-wasm-002`](task-wasm-002-in-memory-api.md) | モデル・設定・辞書・画像をメモリから読み込む API | 破壊的変更: `*_path()` が `Option<&Path>` を返すようになった |
| `[x]` | [`task-wasm-003`](task-wasm-003-bindings-simd.md) | wasm-bindgen のバインディング（`bindings/wasm`）と SIMD128 | SIMD で約 2 倍速くなる |
| `[x]` | [`task-wasm-004`](task-wasm-004-browser-demo.md) | ブラウザのデモ（Web Worker）と、ヘッドレス Chrome での確認 | tiny は 2.1 秒、medium は 30 秒 |
| `[x]` | [`task-wasm-005`](task-wasm-005-thread-safety.md) | `OcrEngine` を `Send + Sync` にする | `RefCell` を `Mutex` に置き換えた |

## 🔭 Follow-ups

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 中 | npm パッケージとして配布する（`wasm-pack` または手作業で `package.json` を用意する） | 現在は手順に沿ってビルドする必要がある |
| 中 | CI に `wasm32-unknown-unknown` のビルドを追加する | 今回のように、依存関係の更新でビルドが壊れても気付けない |
| 低 | マルチスレッドの wasm（`wasm-bindgen-rayon`、SharedArrayBuffer） | COOP/COEP ヘッダが必要。単一スレッドでも tiny と small は実用的な速度で動く |
| 低 | モデルのキャッシュ（Cache Storage や IndexedDB） | medium は 76MB あり、毎回ダウンロードすると時間がかかる |
