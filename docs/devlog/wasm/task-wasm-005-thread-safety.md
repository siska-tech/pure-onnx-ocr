---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [concurrency, api]
depends_on: task-wasm-002
---

# タスク概要
`OcrEngine` のドキュメントコメントには、「`Arc` で包んでいるのでスレッド間で共有できる」と書かれていた。実際にはそうなっていなかったので、修正する。

## 調査
- `DetInferenceSession` / `RecInferenceSession` / `OrientationClassifier` は、推論計画のキャッシュを `RefCell` で保持している。そのため `!Sync` になる。
- エンジンはセッションを `Arc<Session>` として持っている。`Arc<T>` が `Send` になるには `T: Send + Sync` が必要なので、**エンジンは `Send` でも `Sync` でもなかった**。
- clippy の `arc_with_non_send_sync` 警告も、この問題を指摘していた。

## 実装メモ
- キャッシュを `std::sync::Mutex<PlanCache>` に置き換えた。
  - ロックは `onnx_model::lock_cache` で取得する。panic によるロックの汚染（poison）は無視する。
  - ロックを持つのは、キャッシュから計画を探す間と、計画を追加する間だけにした。推論は、取り出した `Arc<plan>` を使ってロックの外で実行する。
- wasm のように単一スレッドの環境でも、`Mutex` はそのまま動作する。

## 検証
- 型レベルのテストで、`OcrEngine: Send + Sync` をコンパイル時に確認する。
- 統合テスト `engine_can_be_shared_between_threads` を追加した。1 つのエンジンを `Arc` で 3 スレッドから同時に使い、結果が単一スレッドの場合と一致することを確認する。
- clippy の `arc_with_non_send_sync` 警告がなくなった。
