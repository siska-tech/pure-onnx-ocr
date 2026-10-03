---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [ppocrv6, test, benchmark, docs]
depends_on: task-v6-004
---

# タスク概要
PP-OCRv6 の 3 階層と PP-OCRv5 を実画像で動かし、認識結果と処理時間を記録する。あわせて、テストとドキュメントを整備する。

## 測定条件
- 画像 1: `general_ocr_002.jpg`。896x528 の搭乗券で、中国語と英語が混在する。PaddleX の公式デモ画像。
- 画像 2: `ja.jpg`。1536x839 の日本語の単語群。PaddleOCR リポジトリの `doc/imgs/japan_2.jpg`。
- 実行方法: `ocr_smoke --benchmark`（release ビルド）。各構成を 1 回ずつ、ほかの処理を止めて実行した。推論計画のコンパイル時間も含む。
- 環境: Windows 11、16 スレッド、tract 0.23.8。
- モデル:
  - PP-OCRv5 は Hugging Face の `PP-OCRv5_mobile_{det,rec}_onnx` と `ppocrv5_dict.txt`。
  - PP-OCRv6 は `PP-OCRv6_{tier}_{det,rec}_onnx` で、`--det-model-dir` / `--rec-model-dir` で指定した。

## 結果: 処理時間（秒）

| 構成 | 画像 | 検出領域数 | 全体 | 検出推論 | 認識推論 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| v5 mobile | 搭乗券 | 34 | 7.15 | 0.86 | 6.21 |
| v6 tiny | 搭乗券 | 37 | 2.29 | 0.68 | 1.56 |
| v6 small | 搭乗券 | 33 | 6.46 | 1.18 | 5.21 |
| v6 medium | 搭乗券 | 33 | 20.93 | 4.13 | 16.73 |
| v5 mobile | 日本語 | 51 | 8.49 | 0.91 | 7.45 |
| v6 tiny | 日本語 | 56 | 2.40 | 0.69 | 1.60 |
| v6 small | 日本語 | 55 | 8.15 | 1.17 | 6.83 |
| v6 medium | 日本語 | 56 | 25.77 | 4.55 | 21.11 |

参考値: 前処理を修正する前（tract 0.23、旧前処理、1 バッチで幅 320 固定）の v5 は、搭乗券で全体 4.48 秒だった。認識幅を可変にした分だけ遅くなっているが、長い行が読めるようになった（下記）。

## 結果: 認識テキスト（搭乗券）

**v5 mobile、旧前処理**（BGR なし・正規化なし・space なし・幅 320 固定）:
```
ccaa | cYaaaananacl | (空) | caaa | BOARDING | 登机牌 | 座位号SEATNO | 序号[UNK]SERIAL[UNK]NO. | ca | 舱位 | 日期DATE | 航班FlIGHT | 035 | (空) | W | 03DG | MU2379 | ca | 登机时间BDT | 登机口GATe | ...
| 姓名Am | 票号aaka | 票价ARE | ETKT[UNK]7813699238489/1 | (末尾の長い行が欠落)
```

**v5 mobile、新前処理**:
```
www.997788.com | BOARDING PASS | 登机牌 | SEAT NO | 座位号 | 序号 SERIAL NO. | CLASS | 舱位 | 日期DATE | 航班FLIGHT | 035 | W | 03DEC | MU2379 | 登机时间BDT | 登机口 GATE | 始发地FROM | 目的地TO | 福州 | G11 | TAIYUAN | FUZHOU | 身份识别IDNO | 姓名NAME | ZHANGQIWEI | 票号TKT NO. | 张祺伟 | 票价FARE | ETKT7813699238489/1
| 登机口于起飞前10分钟关闭 GATES CLOSE10MINUTES BEFORE DEPARTURE TIME
```

**v6 small**:
```
中国收藏热线 | www.997788.com | PASS | BOARDING | 登机牌 | SEAT NO | 座位号 | 序号SERIAL NO. | 舱位 CLASS | 日期 DATE | 航班 FLIGHT | 035 | W | 03DEC | MU 2379 | ... | 身份识别ID NO. | 姓名NAME | ZHANGQIWEI | 票号 TKT NO. | 张祺伟 | 票价 FARE | ETKT 7813699238489/1
| 登机口于起飞前10分钟关闭 GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME
```

**v6 medium**: small とほぼ同じ。`MU 2379 03DEC` を 1 領域として読み、`www.997788.com-中国收藏：` まで読めた。

**v6 tiny**: 本文はおおむね読める。ただし左上の透かし（`中Ey | wwrsg.gg/88` など）と、`漏州`（正しくは福州）、`O3DEC` などの誤りがある。

旧前処理で出ていたノイズ（`ccaa` など）、空白の `[UNK]`、末尾の長い行の欠落は、新前処理ではすべて解消した。

## 結果: 認識テキスト（日本語）
- **v6 small / medium**: 56 語のほぼすべてを正しく読めた。誤りは `不わ不わ`（正しくは ふわふわ）、`スパイシ`（スパイシー）、`ロどけ`（small のみ。正しくは 口どけ）など少数。
- **v5 mobile**: おおむね読めるが、近くの語が連結されたもの（`とろっと後味のよい` など）や、`ふんわ`（ふんわり）、`食べろ`（食べごろ）などの誤りが v6 より多い。
- **v6 tiny**: かなが全く読めない（`毛古毛古` = もちもち）。辞書にひらがなとカタカナが無いことによる仕様上の制約で、[research](research-ppocrv6.md) に記録した。

## テスト
`tests/ppocrv6.rs` を追加した。フィクスチャが無い場合はスキップする。

| テスト | 内容 | 既定で実行 |
| :--- | :--- | :--- |
| `detection_configs_use_bgr_imagenet_normalisation` | 検出 YAML 3 種の model_name / BGR / mean / std | ○ |
| `recognition_configs_embed_dictionaries` | 認識 YAML 3 種の辞書件数（6,904 / 18,708）と image_shape | ○ |
| `recognition_class_count_matches_dictionary` | tiny_rec の出力クラス数と、blank と space を含む辞書長が一致すること | ○ |
| `tiny_pipeline_reads_boarding_pass` | end-to-end で `BOARDING`、`张祺伟`、空白を含む末尾の行などが読めること | ○ |
| `small_pipeline_reads_boarding_pass` | 同上（small） | `--ignored` |
| `medium_pipeline_reads_boarding_pass` | 同上（medium） | `--ignored` |
| `ppocrv5_yaml_dictionary_matches_text_dictionary` | v5 の YAML 辞書とテキスト辞書が一致すること | ○ |

実行結果（release ビルド）:
- `cargo test --release`: すべて成功した。内訳は lib 53、ppocrv6 5、ほか 6 で、ignored は 4。
- `cargo test --release --test ppocrv6 -- --ignored`: small と medium の 2 件が成功した（31.6 秒）。
- `cargo test --release --lib -- --ignored`: ダミー推論 2 件が成功した。`models/ppocrv5/` に v5 mobile を一時的に置いて実行した。
- `cargo clippy --all-targets`: 今回の変更による警告は無い。変更前から存在する警告が 8 件残っている。

注意: `tests/integration_test.rs` は `images/demo.png` を前提にしている。この環境には `demo.png` が無いため、3 件ともスキップ扱いで成功になっている。

## ドキュメント
- `README.md` / `README_en.md` を更新した。PP-OCRv6 の取得手順、階層ごとの比較、モデルディレクトリ API の例、CLI の例、既知の制約、MSRV 1.91 を記載した。
- `tests/fixtures/README.md` に、PP-OCRv6 のディレクトリ構成とダウンロード手順を追記した。
