# ブラウザ向け WebAssembly 対応 調査・検討ログ

- 調査日: 2026-10-03
- 対象ブランチ: `feature/wasm-browser`（`feature/ppocrv6-support` から分岐）
- 関連: [ROADMAP_wasm.md](ROADMAP_wasm.md)

## 1. 背景

README には「WASM などでも動作を想定」と書かれている。PP-OCRv6 対応の後で実際に確認したところ、ブラウザ（`wasm32-unknown-unknown`）では使えない状態だった。

| ターゲット | develop | `feature/ppocrv6-support` |
| :--- | :--- | :--- |
| `wasm32-wasip1`（WASI） | 未確認 | ビルド成功。Node.js の WASI 上で OCR が正しく動作した |
| `wasm32-unknown-unknown`（ブラウザ） | ビルドは成功 | **ビルド失敗** |

## 2. 原因

### 2.1 ビルド失敗（v6 対応で発生）

`tract-onnx` 0.23 の依存関係が、getrandom 0.4 を引き込んでいる。

```
tract-onnx 0.23 -> tract-onnx-opl -> rand 0.10 -> getrandom 0.4
```

getrandom 0.4 は、`wasm32-unknown-unknown` 向けには JS バックエンド（`wasm_js` 機能）を明示的に有効にしないと、`compile_error!` で停止する。

### 2.2 実行時の問題（develop の時点から存在）

| 問題 | 内容 |
| :--- | :--- |
| `std::time::Instant::now()` | `wasm32-unknown-unknown` では呼んだ時点でパニックする。task-fix-003（ベンチマーク）以降、推論処理の中で必ず呼ばれている |
| ファイルシステム前提の API | モデル・設定・辞書はパスで受け取り、`std::fs` で読み込む。画像も `run_from_path` か、デコード済みの `DynamicImage` しか受け付けない。ブラウザにはファイルシステムがない |

### 2.3 問題にならなかったもの

- **rayon**（image と imageproc が依存している）: ブラウザにはスレッドがないが、現在のスレッドで処理するように切り替わり、問題なく動作した。
- **tract の推論**: 純粋な Rust で書かれており、そのまま動作した。

## 3. 方針

| 項目 | 検討した案 | 決定 |
| :--- | :--- | :--- |
| getrandom | (a) target 限定で `wasm_js` を有効にする。(b) tract-onnx-opl を外す | (a)。opl は ONNX の標準演算子の一部を実装しているため外せない |
| 時刻 | (a) `web-time` を使う。(b) wasm では計測しない（0 を返す） | (a)。`performance.now()` を使うので、ブラウザでも計測値が意味を持つ |
| ファイル入力 | パスを使う API と並べて、バイト列・テキストを受け取る API を追加する | 既存 API は維持し、パスとメモリ入力のうち後から指定したほうを優先する |
| JS との連携 | wasm-bindgen のバインディングを別クレートにする | 本体を wasm-bindgen に依存させないため、`bindings/wasm` をワークスペースのメンバーとして追加した |
| SIMD | `+simd128` を既定で有効にする | 主要ブラウザはすべて対応済み（Chrome/Edge 91+、Firefox 89+、Safari 16.4+）。`.cargo/config.toml` で設定し、古いブラウザ向けには削除できるようにした |

## 4. 検証方法

1. `wasm32-unknown-unknown` 向けにビルドし、wasm-bindgen 0.2.105 で JS のグルーコードを生成する。
2. Node.js の `--target nodejs` 版で、ネイティブ版と同じ画像を処理する。
3. ブラウザ版のデモ（`examples/web`）をヘッドレス Chrome 153 で開き、DevTools プロトコルで `window.ocrOutput` を読み取る。

補足: 最初の実行では、既定のポート 9333 を別のツールが起動した Edge が使っていた。そのため、テスト用のタブがその Edge に開かれた。タブはすぐに閉じ、2 回目以降は空いているポートで自前のヘッドレス Chrome を起動した。検証スクリプトは、ポートが使用中なら中止するように修正済みである。

## 5. 結果

画像は 896x528 の搭乗券。

| 環境 | モデル | 合計 | 備考 |
| :--- | :--- | ---: | :--- |
| ネイティブ（Windows、x86_64） | PP-OCRv6 tiny | 約 2.3 秒 | 比較用 |
| Node.js、wasm32 SIMD なし | tiny | 6.7 秒 | |
| Node.js、wasm32 SIMD あり | tiny | 2.4〜3.1 秒 | SIMD で約 2 倍速くなる |
| ヘッドレス Chrome 153、SIMD あり | tiny | 2.1 秒 | |
| ヘッドレス Chrome 153 | small + 行の向き + ページの向き | 7.7 秒 | 方向分類に 0.6 秒 |
| ヘッドレス Chrome 153 | medium | 30.4 秒 | |

認識結果のテキストは、ネイティブ版と完全に一致した（tiny で 37 領域）。

## 6. あわせて修正したこと

- `OcrEngine` のドキュメントには「スレッド間で共有できる」と書かれていた。しかし実際は、推論計画のキャッシュに使っている `RefCell` のせいで `Send` でも `Sync` でもなかった。キャッシュを `Mutex` に置き換えて `Send + Sync` にした。ロックはキャッシュの参照中だけ保持し、推論中は保持しない。
