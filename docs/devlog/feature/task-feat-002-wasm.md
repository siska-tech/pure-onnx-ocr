---
status: progress
priority: medium
assignee: Backend
start_date: 2025-01-27
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

### 2025-01-27

#### 完了事項

1. **wasm-bindgen 依存関係の追加**
   - `Cargo.toml` に `wasm-bindgen` をオプション依存関係として追加
   - `console_error_panic_hook`, `serde`, `serde-wasm-bindgen`, `js-sys` を追加
   - `wasm` feature を追加して有効化

2. **メモリバッファベースのモデルロード対応**
   - `TractDetSession::load_from_bytes()` メソッドを追加
   - `TractRecSession::load_from_bytes()` メソッドを追加
   - `tract-onnx` の `model_for_read()` API を使用してメモリバッファからモデルをロード

3. **メモリバッファベースの辞書ロード対応**
   - `RecDictionary::from_bytes()` メソッドを追加
   - `from_str()` ヘルパーメソッドで共通化
   - ユニットテストを追加

4. **WASM ターゲットでのビルド確認**
   - `wasm32-unknown-unknown` ターゲットでのビルドが成功することを確認
   - `Cargo.toml` に `[lib]` セクションを追加して `crate-type = ["cdylib", "rlib"]` を設定

5. **wasm-bindgen を使った JavaScript バインディングの実装**
   - `src/wasm.rs` モジュールを追加
   - `WasmOcrEngineBuilder` 構造体とメソッドを実装
   - `WasmOcrEngine` 構造体と `run_from_bytes()` メソッドを実装
   - JavaScript から呼び出せる API を提供

6. **ブラウザデモの作成**
   - `examples/wasm-demo/` ディレクトリを作成
   - `index.html` - 参考ファイルのスタイルを踏襲した美しいUI
   - `index.js` - WASMモジュールの読み込みとOCR処理の実装
   - `README.md` - セットアップ手順と使用方法を記載

7. **README への WASM ビルド手順の追記**
   - WASM ビルド方法とメモリバッファベースの使用方法を記載

#### 未完了事項

1. **パッケージサイズ最適化**
   - `wasm-opt` の適用
   - Tree Shaking の最適化

2. **技術調査項目の確認**
   - `tract-onnx` の WASM 互換性の詳細確認
   - 画像デコードと後処理クレートの WASM 互換性確認
   - 実際のブラウザ環境での動作確認
