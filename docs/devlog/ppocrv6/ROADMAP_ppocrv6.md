# `ROADMAP_ppocrv6.md`

## 🎯 目標

PaddleOCR PP-OCRv6（2026-06 公開）の ONNX モデル（tiny / small / medium）を、Pure Rust のまま `OcrEngine` で実行できるようにする。PP-OCRv5 との互換性も維持する。

調査と方針の検討は [research-ppocrv6.md](research-ppocrv6.md) にまとめた。

## 📊 進捗

ブランチ: `feature/ppocrv6-support`

| ステータス | タスクID | 概要 | 備考 |
| :--- | :--- | :--- | :--- |
| `[x]` | [`task-v6-001`](task-v6-001-tract-upgrade.md) | tract 0.23 への移行と、value_info のシンボリック次元の破棄 | MSRV を 1.91 に変更。medium_rec を動かすために必須 |
| `[x]` | [`task-v6-002`](task-v6-002-inference-yml.md) | `inference.yml` の読み込みと、辞書の YAML 対応・space クラス | 依存を増やさない最小 YAML パーサ |
| `[x]` | [`task-v6-003`](task-v6-003-pre-post-alignment.md) | 前処理と後処理を PaddleOCR 3.x に合わせる | BGR / ImageNet 正規化 / box_thresh / 可変幅認識 |
| `[x]` | [`task-v6-004`](task-v6-004-builder-cli.md) | モデルディレクトリ指定 API・縦横比ソートのバッチ化・CLI | `det_model_dir` / `rec_model_dir`、`--det-model-dir` など |
| `[x]` | [`task-v6-005`](task-v6-005-validation.md) | テスト・実測・ドキュメント | v5 と v6 の 3 階層を 2 枚の画像で比較 |

## 🔭 Follow-ups

| 優先度 | 概要 | 背景 |
| :--- | :--- | :--- |
| 高 | 検出領域を回転補正して切り出す（PaddleOCR の `get_rotate_crop_image` 相当）。縦長の領域は 90° 回転する | 現状は外接矩形で切り出しているので、傾いた行や縦書きに弱い |
| 高 | `smoke/task-fix-001`（OCR 結果の乱れの調査）を、本ブランチの成果で再評価して閉じる | 乱れの主因（検出の正規化漏れ、space クラスの欠落、認識幅の固定）は本ブランチで解消済み |
| 中 | 検出の原寸モード（PaddleOCR 3.x の `limit_type=min, limit_side_len=64, max_side_limit=4000`）を選べるようにする | PaddleOCR と同じ条件で比較するため。速度とのトレードオフがある |
| 中 | 認識の推論計画をシンボリックな幅で 1 回だけ最適化し、入力形状ごとの再コンパイル（1 回あたり 100〜400 ms）をなくす | 可変幅にしたことで、コンパイルされる形状の種類が増えた |
| 中 | `inference.yml` の `PostProcess`（thresh / box_thresh / unclip_ratio）を明示的に適用するオプションを追加する | 現在は PaddleOCR パイプラインの既定値（0.3 / 0.6 / 1.5）を使っている |
| 低 | `DetInferenceSession` / `RecInferenceSession` の `println!` ログを `log` クレートなどへ移す | ライブラリが標準出力に書き込むのは利用者にとって扱いにくい |
| 低 | PP-OCRv6 の検出・認識以外のモデル（文字行の向き分類、レイアウト解析）に対応する | 本タスクの対象外 |
