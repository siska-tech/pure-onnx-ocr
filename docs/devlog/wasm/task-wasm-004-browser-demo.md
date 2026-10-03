---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [wasm, browser, demo]
depends_on: task-wasm-003
---

# タスク概要
ブラウザで動くデモを用意し、実際のブラウザエンジンで動作を確認する。

## 実装メモ（`examples/web`）
- `worker.js`:
  - モデルを `fetch` で読み込み、Web Worker の中で wasm のエンジンを動かす。
  - 推論は同期処理なので、Worker で実行しないとメインスレッドが止まる。
- `index.html`:
  - ファイルを選ぶと、OCR の結果をキャンバスに描画する（`box` を枠として描く）。
  - `?image=URL&tier=…&textline=1&doc=1` を付けると、ページを開いたときに自動で実行し、結果を `window.ocrOutput` に保存する（自動テスト用）。
- `serve.mjs`:
  - 依存のない静的ファイルサーバー。
  - `.wasm` を `application/wasm` として配信する。
  - モジュール Worker は `file://` では動かないので、HTTP で配信する必要がある。
- `README.md`: ビルド手順、モデルの配置、計測値を記載した。
- `examples/web/pkg/`（生成物）と `examples/web/models/` は gitignore に追加した。

## 検証（ヘッドレス Chrome 153、DevTools プロトコル）

| 構成 | 合計 | 検出 | 認識 | 方向分類 |
| :--- | ---: | ---: | ---: | ---: |
| tiny | 2.1 秒 | 0.7 秒 | 1.4 秒 | – |
| small + 行の向き + ページの向き | 7.7 秒 | 1.1 秒 | 5.9 秒 | 0.6 秒 |
| medium | 30.4 秒 | 5.5 秒 | 24.9 秒 | – |

- tiny の結果（37 領域）は、ネイティブ版・Node.js 版と完全に一致した。
- 検証スクリプトはリポジトリに含めていない（スクラッチ領域に置いた）。手順は [research-wasm.md](research-wasm.md) の 4 章に記載した。
