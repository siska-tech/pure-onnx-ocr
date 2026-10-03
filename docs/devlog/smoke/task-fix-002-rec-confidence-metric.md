---
status: completed
priority: medium
assignee: Backend
start_date: 2025-11-10
end_date: 2025-11-10
tags: [quality, metrics, recognition]
depends_on: task-fix-001
---

# タスク概要
CTC デコーダーの信頼度（confidence）計算を暫定仕様から脱却させ、PaddleOCR 準拠の Softmax 正規化に基づいた定量的スコアへ置き換える。

## 要件
- `ctc::DecodedSequence` が保持する `confidence` を Softmax ベースの平均対数尤度（もしくは同等の安定化手法）で算出し直すこと。
- 既存の `ocr_smoke` CLI で 0.000 固定になっている表示を、実測値に近いレンジへ更新する。
- ロジットのオーバーフローを避けるために log-sum-exp 等の数値安定化を必ず組み込む。
- 新アルゴリズムを検証するユニットテスト／統合テストを追加し、旧仕様との比較（最低限ベンチマークで 0 にならないこと）を確認する。
- タスク完了後、`README.md` や関連ドキュメントの「現在の制約」セクションを更新する。

## メモ
- `CtcGreedyDecoder` が time-step ごとに確率分布を推定し、既に正規化済みの予測であればそのまま最大値を使用、ロジットの場合は log-sum-exp で Softmax を適用した上で算出した確率の**算術平均**を `DecodedSequence::confidence` として返すよう更新。
- 空文字列の場合は PaddleOCR と同じく信頼度 1.0 を返すように変更。
- `ctc` モジュールに確率分布検出／Softmax 計算・算術平均を確認するユニットテストを追加。
- `cargo test` 全系統成功、`cargo fmt` 適用済み。

