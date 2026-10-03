# PP-OCRv6 対応 調査・検討ログ

- 調査日: 2026-10-03
- 対象ブランチ: `feature/ppocrv6-support`
- 関連タスク: [ROADMAP_ppocrv6.md](ROADMAP_ppocrv6.md)

## 1. PP-OCRv6 の概要

| 項目 | 内容 |
| :--- | :--- |
| 公開日 | 2026-06-11（PaddleOCR v3.7.0 と同時） |
| 構成 | 検出 (DBNet 系) + 認識 (CTC デコード) の 2 段構成。PP-OCRv5 と同じ |
| モデル階層 | `tiny` / `small` / `medium` の 3 段階（総パラメータ 1.5M〜34.5M） |
| 対応言語 | small / medium は 50 言語（簡体字・繁体字・英語・日本語・ラテン系 46 言語）。tiny は小さい辞書 |
| 配布形態 | Hugging Face `PaddlePaddle/PP-OCRv6_{tier}_{det,rec}_onnx` に ONNX 版あり |
| PaddleOCR 既定 | 3.x の OCR パイプライン既定モデルは `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` |

参考:
- https://huggingface.co/blog/PaddlePaddle/pp-ocrv6
- https://arxiv.org/html/2606.13108v1
- https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx
- https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx

### 1.1 配布ファイル

各リポジトリには次のファイルだけが含まれる。**辞書テキストファイルは同梱されていない。**

```
inference.onnx   ONNX グラフ
inference.yml    前処理・後処理設定（認識モデルは辞書を内包）
inference.json   Paddle 形式のグラフ定義（本クレートでは不使用）
README.md
```

| モデル | inference.onnx | inference.yml |
| :--- | ---: | ---: |
| tiny_det | 1.8 MB | 0.9 KB |
| small_det | 9.9 MB | 0.9 KB |
| medium_det | 62.0 MB | 0.9 KB |
| tiny_rec | 4.5 MB | 56 KB |
| small_rec | 21.2 MB | 151 KB |
| medium_rec | 76.6 MB | 151 KB |

### 1.2 `inference.yml` の内容

検出モデル（3 階層とも同形式。値は tiny / small / medium の順）:

```yaml
PostProcess:
  name: DBPostProcess
  thresh: 0.2
  box_thresh: 0.4 / 0.4 / 0.45
  unclip_ratio: 1.4
  max_candidates: 3000
PreProcess:
  transform_ops:
  - DecodeImage: {img_mode: BGR}
  - DetResizeForTest: null
  - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], scale: 1./255.}
```

認識モデル:

```yaml
PreProcess:
  transform_ops:
  - DecodeImage: {img_mode: BGR}
  - RecResizeImg: {image_shape: [3, 48, 320]}
PostProcess:
  name: CTCLabelDecode
  character_dict: [...]   # 1 行 1 文字の YAML シーケンス
```

| モデル | `character_dict` 件数 | ONNX 出力クラス数 | 差分 |
| :--- | ---: | ---: | :--- |
| PP-OCRv6 tiny_rec | 6,904 | 6,906 | blank + space |
| PP-OCRv6 small_rec | 18,708 | 18,710 | blank + space |
| PP-OCRv6 medium_rec | 18,708 | 18,710 | blank + space |
| PP-OCRv5 mobile_rec（参考） | 18,383 | 18,385 | blank + space |

small と medium の YAML は `model_name` 以外は同一（辞書も同一）。

tiny の辞書（6,904 文字）には漢字とラテン文字が含まれるが、**ひらがなとカタカナが 1 文字も含まれない**。モデルカードには「49 languages」とあるが、日本語は対象外とみなすべきである。実際に日本語画像を tiny で認識すると、かなが類似した漢字や記号に置き換わる（5 章を参照）。日本語を扱う場合は small 以上を使う。

## 2. 現行実装との差分調査

### 2.1 tract での実行可否（最大のリスク）

検証用の小さなプログラムで、6 モデルを読み込み、ゼロ入力で推論した結果。入力は検出が `1x3x320x320`、認識が `1x3x48x320`。

| モデル | tract 0.20.7（現行） | tract 0.20.7 + value_info 破棄 | tract 0.23.8 + value_info 破棄 |
| :--- | :--- | :--- | :--- |
| tiny_det | 失敗 (shape unify) | OK 101 ms | OK 45 ms |
| tiny_rec | 失敗 | OK 18 ms | OK 9 ms |
| small_det | 失敗 | OK 191 ms | OK 95 ms |
| small_rec | 失敗 | OK 65 ms | OK 39 ms |
| medium_det | 失敗 | OK 672 ms | OK 470 ms |
| medium_rec | 失敗 | **失敗**（ReduceMean） | OK 162 ms |

補足: develop ブランチ（tract 0.20、value_info 破棄なし）でビルドした `ocr_smoke` は、Hugging Face 配布の **PP-OCRv5** mobile（`PaddlePaddle/PP-OCRv5_mobile_det_onnx`）でも同じエラーで失敗した。

```
error: detection inference failed: Failed analyse for node #247 "Conv.0" ConvHir
```

つまり、この問題は v6 に固有のものではない。PaddleOCR 3.x の paddle2onnx で出力されたモデルすべてに共通する。

発見した問題は 2 つ。

1. **`value_info` のシンボリック次元**
   - PaddleOCR 3.x の ONNX は、中間テンソルの形状を `DynamicDimension.0` などのシンボルで `value_info` に記録している。
   - tract はこれを推論ルールとしてそのまま採用する。入力に具体的な形状 `1x3xHxW` を与えると、`Impossible to unify Sym(DynamicDimension.0) with Val(1)` で失敗する。
   - tract-onnx には `value_info` を無視するオプションがない（0.20 / 0.23 とも）。
   - 対策として、読み込み後に「入力でも定数でもない outlet」の fact を `InferenceFact::default()` へリセットし、tract に形状を再推論させる。PP-OCRv5 のエクスポートは `value_info` を持たないので、この処理は v5 には影響しない。
2. **tract-core 0.20.7 の不具合**
   - `ops/nn/reduce.rs:289` に、次元 768 を含む ReduceMean を拒否するデバッグ用の `ensure!` が残っている。
   - medium_rec の隠れ次元はちょうど 768 なので、0.20 系では回避できない。
   - tract 0.21 は `DynamicDimension.0` という次元名自体をパースできなかった。0.23 はすべて成功し、推論速度も 0.20 の約 1.5〜2 倍だった。

### 2.2 前処理・後処理（PaddleOCR 3.x / PaddleX の参照実装との比較）

参照: `paddlex/inference/models/text_recognition/processors.py`、`paddlex/configs/pipelines/OCR.yaml`（develop ブランチ）

| 項目 | PaddleOCR 3.x | 現行実装 | 影響 |
| :--- | :--- | :--- | :--- |
| 検出の色順 | BGR | RGB | 検出精度の低下 |
| 検出の正規化 | ImageNet mean/std | `x/255` のみ | **確率マップが大きく劣化**。ノイズ領域が出る |
| 検出のリサイズ | `limit_type=min, limit_side_len=64, max_side_limit=4000`（縮小はほぼしない） | 長辺 960 に縮小 | 小さい文字の検出漏れ。ただし速度は有利 |
| `thresh` / `box_thresh` / `unclip_ratio` | パイプライン既定 0.3 / 0.6 / 1.5（YAML の値を上書き） | 0.3 / なし / 1.5 | **box_thresh が無く**、低スコア領域が残る |
| 輪郭 | 外側輪郭のみ | 穴の輪郭も候補にする | 重複・反転した候補が出る |
| 認識の色順 | BGR | RGB | 認識精度の低下 |
| 認識の幅 | `max(320, ceil(48 x 縦横比))`、上限 3200 | 320 に固定して押し潰す | **長い行が読めない** |
| 認識のパディング値 | 正規化後に 0 | 正規化後に -1（黒） | 軽微 |
| 認識のバッチ | 縦横比でソートしてから分割 | 全領域を 1 バッチ（`rec_batch_size` は未使用） | 無駄なパディングが増える |
| 辞書 | `blank` + 辞書 + `" "`（`use_space_char=True`） | `blank` + 辞書 | **空白が `[UNK]` になる** |
| 辞書の供給 | `inference.yml` に内包 | テキストファイルのみ | **v6 は辞書を読めない** |

太字の項目は PP-OCRv5 でも同じ問題が起きている。`docs/devlog/smoke/task-fix-001-ocr-smoke-quality.md` で調査中の「OCR 結果の乱れ」の主因と考えられる（後述の実測で確認）。

### 2.3 v5 からの継続点（変更不要）

- 検出の出力は `[N, 1, H, W]` の確率マップで、DB 後処理をそのまま使える。
- 認識の出力は `[N, T, C]` で、softmax 済みの確率値。`T = W / 8`。既存の CTC greedy デコーダをそのまま使える。
- 認識の入力高さは 48 で変わらない。

## 3. 検討した方針と決定

### 3.1 tract のバージョン

| 案 | 長所 | 短所 |
| :--- | :--- | :--- |
| A. 0.20 のまま、value_info 破棄だけ行う | 依存関係も MSRV も変わらない | **medium_rec が動かない**。既定モデルに対応できない |
| B. 0.23 へ更新する（採用） | 6 モデルすべて動作。推論は約 1.5〜2 倍速い | MSRV が 1.70 から 1.91 に上がる。`ndarray` が 0.17 になる。API の小修正が必要 |

PaddleOCR 3.x の既定モデルは medium である。「PP-OCRv6 に対応する」と言うには medium が動くことが必須なので、B を採用した。

### 3.2 `inference.yml` の読み込み方法

| 案 | 内容 | 判断 |
| :--- | :--- | :--- |
| `serde_yaml` などを追加する | 汎用の YAML パーサを使う | 依存が増える。`serde_yaml` はメンテナンス終了済み |
| 辞書だけをテキストに書き出すツールを用意する | 実行時は従来どおりテキスト辞書を読む | 利用者に変換作業が残る |
| **最小限の YAML パーサを自前で実装する（採用）** | PyYAML が出力するブロック形式だけに対応する | 純 Rust・依存ゼロ。読める範囲はテストで固定する |

対応する構文は以下。
- ブロックマップ
- ブロックシーケンス。親キーと同じインデントの `- item` 形式と、入れ子の `- - x` 形式を含む
- 引用符なしのスカラー、シングルクオートとダブルクオートのスカラー（エスケープを含む）
- アンカーとエイリアス（不透明な値として保持する）

フロー形式 `[a, b]` や複数行スカラーは、エラーで明示的に拒否する。

注意点として、トリミングの対象は YAML の空白（半角スペースとタブ）だけにする。Rust の `trim()` は U+3000（全角スペース）も除去してしまうので使わない。

### 3.3 しきい値の既定値

YAML の `PostProcess` にある値（thresh 0.2、box_thresh 0.4、unclip 1.4）は、モジュール単体で使うときの既定値である。PaddleOCR の OCR パイプラインは、これを `OCR.yaml` の 0.3 / 0.6 / 1.5 で上書きする。本クレートはパイプラインとして使われるため、既定値はパイプライン側の値に合わせ、YAML の値は自動では適用しない。利用者はビルダーの `det_threshold` / `det_box_threshold` / `det_unclip_ratio` で変更できる。

### 3.4 検出のリサイズ方針

PaddleOCR 3.x は実質的に原寸で検出する（`limit_type=min`）。tract の CPU 推論では大きな画像ほど時間がかかるため、今回は従来の「長辺 960」を維持した。原寸モードは後続の [task-v6-007](task-v6-007-det-resize-and-thresholds.md) で、オプション（`DetLimitType::Min`）として追加した。

### 3.5 互換性への影響（破壊的変更）

- `DetPreProcessorConfig` に `mean` / `std` / `color_order` を追加した。構造体リテラルで初期化しているコードは `..Default::default()` が必要になる。
- `DetPostProcessorConfig` に `box_threshold` / `max_candidates` を追加した。
- `RecPreProcessorConfig` に `max_dynamic_width` / `width_alignment` / `color_order` を追加した。`pad_value` の既定値は 0.0 から 0.5 に変わった。
- 既定の前処理が PaddleOCR 準拠に変わるため、v5 を使っている場合も出力が変わる（改善方向）。
- 辞書の末尾に `" "` を追加することが既定になった。`rec_use_space_char(false)` で従来の挙動に戻せる。
- MSRV が 1.91 になった。

## 4. 実装の概要

| 変更 | ファイル | タスク |
| :--- | :--- | :--- |
| tract 0.23 への移行と、value_info の破棄 | `Cargo.toml`, `src/onnx_model.rs`, `src/detection.rs`, `src/recognition.rs` | [task-v6-001](task-v6-001-tract-upgrade.md) |
| `inference.yml` の読み込みと、辞書の YAML 対応・space クラス | `src/paddle_config.rs`, `src/dictionary.rs` | [task-v6-002](task-v6-002-inference-yml.md) |
| 前処理と後処理を PaddleOCR 3.x に合わせる | `src/preprocessing.rs`, `src/postprocessing.rs` | [task-v6-003](task-v6-003-pre-post-alignment.md) |
| モデルディレクトリ指定・バッチ化・CLI | `src/engine.rs`, `src/bin/ocr_smoke.rs` | [task-v6-004](task-v6-004-builder-cli.md) |
| テスト・実測・ドキュメント | `tests/ppocrv6.rs`, `README*.md` | [task-v6-005](task-v6-005-validation.md) |

## 5. 実測結果

詳細は [task-v6-005-validation.md](task-v6-005-validation.md) を参照。要点は次のとおり。

- PP-OCRv6 の tiny / small / medium は、すべて検出から認識まで end-to-end で動作する。
- PP-OCRv5 も前処理の修正で大きく改善した。ノイズ領域が消え、空白が正しく出力され、長い行も読めるようになった。
- 日本語画像では small / medium がほぼ正しく読める。tiny は辞書にかながないため読めない。
- 速度は tiny が約 2.3 秒、small が約 7 秒、medium が約 23 秒（896x528 の画像 1 枚）。

## 6. 後続課題

[ROADMAP_ppocrv6.md](ROADMAP_ppocrv6.md) の「Follow-ups」を参照。
