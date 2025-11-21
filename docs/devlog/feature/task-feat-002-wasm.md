---
status: pending
priority: medium
assignee: Backend
start_date:
end_date:
tags: [M5, wasm, webassembly, browser]
depends_on: [task-api-002]
---

# タスク概要
WebAssembly (WASM) ターゲットへのコンパイルを可能にし、ブラウザや Node.js 環境で OCR パイプラインを実行できるようにする。

## 背景
Pure Rust 設計の主要なメリットの一つは WASM への移植性である。C/C++ FFI を排除した現行アーキテクチャにより、`wasm32-unknown-unknown` ターゲットへのコンパイルが理論上可能となっている。ブラウザ上でのクライアントサイド OCR やエッジ環境での推論を実現する。

## 要件
- `wasm32-unknown-unknown` ターゲットでのビルド成功を確認する
- ファイルシステムアクセスを抽象化し、`include_bytes!` やメモリバッファ経由のモデルロードに対応する
- `wasm-bindgen` を用いた JavaScript バインディングを実装する
- `tract` の WASM 互換性を検証し、必要に応じて代替推論エンジンを検討する
- ブラウザ上での動作デモ (HTML + JS) を作成する
- パッケージサイズ最適化 (`wasm-opt`, Tree Shaking) を実施する

## 技術調査項目
- [ ] `tract-onnx` の `wasm32` ターゲット対応状況確認
- [ ] `getrandom` クレートの WASM 対応 (js feature flag)
- [ ] 画像デコード (`image` クレート) の WASM 互換性確認
- [ ] `i_overlay` クレートの WASM 互換性確認
- [ ] スレッドプール (`rayon`) の WASM 代替検討

## 成果物
- `pkg/` ディレクトリに npm パッケージとして公開可能な WASM バンドル
- `examples/wasm-demo/` にブラウザデモを配置
- WASM ビルド手順を `README.md` に追記

## 関連リソース
- wasm-bindgen: https://github.com/rustwasm/wasm-bindgen
- wasm-pack: https://github.com/rustwasm/wasm-pack
- tract WASM: https://github.com/sonos/tract/issues (WASM 関連 Issue を参照)

## 作業ログ

