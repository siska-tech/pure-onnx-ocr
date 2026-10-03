# `ROADMAP_ppocrv6.md`

## 🎯 目標

PaddleOCR PP-OCRv6（2026-06 公開）の ONNX モデル（tiny / small / medium）を、Pure Rust のまま `OcrEngine` で実行できるようにする。PP-OCRv5 との互換性も維持する。

調査と方針の検討は [research-ppocrv6.md](research-ppocrv6.md) にまとめた。

## 📊 進捗

ブランチ: `feature/ppocrv6-support`

### P1: PP-OCRv6 対応

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-v6-001`](task-v6-001-tract-upgrade.md) | tract 0.23 への移行と、value_info のシンボリック次元の破棄 | MSRV を 1.91 に変更。medium_rec を動かすために必須 |
| `[x]` | [`task-v6-002`](task-v6-002-inference-yml.md) | `inference.yml` の読み込みと、辞書の YAML 対応・space クラス | 依存を増やさない最小 YAML パーサ |
| `[x]` | [`task-v6-003`](task-v6-003-pre-post-alignment.md) | 前処理と後処理を PaddleOCR 3.x に合わせる | BGR / ImageNet 正規化 / box_thresh / 可変幅認識 |
| `[x]` | [`task-v6-004`](task-v6-004-builder-cli.md) | モデルディレクトリ指定 API・縦横比ソートのバッチ化・CLI | `det_model_dir` / `rec_model_dir`、`--det-model-dir` など |
| `[x]` | [`task-v6-005`](task-v6-005-validation.md) | テスト・実測・ドキュメント | v5 と v6 の 3 階層を 2 枚の画像で比較 |

### P2: P1 の残課題

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-v6-006`](task-v6-006-runtime-plan-cache-logging.md) | 推論計画のキャッシュに上限を設ける。`println!` を `log` クレートへ移す | シンボリック形状の計画は 1.4〜2.6 倍遅いため不採用 |
| `[x]` | [`task-v6-007`](task-v6-007-det-resize-and-thresholds.md) | 検出の原寸モード（`DetLimitType::Min`）と、YAML のしきい値を適用するオプション | 既定は従来どおり長辺 960 |
| `[x]` | [`task-v6-008`](task-v6-008-rotated-crop.md) | 検出領域を回転補正して切り出す。縦長の領域は 90° 回転する | 10° 傾けた画像で、一致した行が 21 から 24 に増加 |
| `[x]` | [`task-v6-009`](task-v6-009-orientation-classifiers.md) | ページの向き（0/90/180/270）と行の向き（0/180）の分類器 | 行の向きの分類器は x0_25 を推奨 |
| `[x]` | `smoke/task-fix-001` | OCR 結果の乱れの調査をクローズする | [task-fix-001](../smoke/task-fix-001-ocr-smoke-quality.md) に再評価結果を記録 |
| `[x]` | [`benchmark-v5-vs-v6`](benchmark-v5-vs-v6.md) | PP-OCRv5 と v6 の速度・精度の比較（tract / CPU） | v6 medium は v5 server より約 2.4 倍速い。v6 small は v5 mobile とほぼ同じ速さで精度が高い |
| `[x]` | [`perf/task-perf-001`](../perf/task-perf-001-multithread.md) | 推論のマルチスレッド化 | パイプライン全体で 2.9〜4.7 倍速くなった |
| `[x]` | [`task-v6-010`](task-v6-010-paddle-parity.md) | PaddleOCR 本体（Python）と出力を突き合わせ、DB 後処理・検出リサイズ・認識前処理を本家と同じにする | v6 medium は本家と完全一致。他も検出 F1 0.95 以上 |

## 🔭 Follow-ups（未着手）

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 低 | 文書の歪み補正（UVDoc）とレイアウト解析（PP-DocLayout など） | OCR パイプラインの外側にある文書解析の機能。出力形式（領域、表、読み順）を含めた API 設計が必要なので、別プロジェクトとして判断する |
| 低 | 短い大文字だけの行で、上下の判定を取りこぼす問題 | PP-LCNet の 0/180 分類の限界。認識結果の信頼度を使った再判定などが考えられる |
| 低 | `OcrResult` で、回転補正に使った四角形（`Quad`）を返す | 現在は検出ポリゴンだけを返している。公開構造体へのフィールド追加になる |
| 低 | `ocr_smoke --benchmark` でも Windows の電力スロットリングを外す | ベンチマーク用の example では対応済み。CLI はそのまま計測すると、値が数倍ぶれることがある |
