---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [wasm, build, dependency]
depends_on:
---

# タスク概要
`wasm32-unknown-unknown` 向けのビルドを直し、ブラウザで時刻の取得によってパニックしないようにする。

## 実装メモ
- `Cargo.toml` に、`cfg(all(target_arch = "wasm32", target_os = "unknown"))` のときだけ有効な依存を追加した。
  - `getrandom = { version = "0.4", features = ["wasm_js"] }`: tract-onnx-opl が rand 0.10 を経由して引き込む getrandom に、JS バックエンドを使わせる。
  - `web-time = "1"`: `performance.now()` を使う `Instant` の代替。
- `src/time.rs`（新規）に `crate::time::Instant` を用意し、ブラウザ向けでは `web_time::Instant`、それ以外では `std::time::Instant` を使うように切り替える。エンジン・認識・方向分類器・ダミー推論の計測は、すべてこれを経由する。
- ネイティブと WASI の挙動は変わらない。

## 検証
- `cargo build --release --lib --target wasm32-unknown-unknown` と、ネイティブのビルド・テストがすべて成功した。
- 実行時の確認（パニックしないこと）は task-wasm-003 / 004 で行った。
