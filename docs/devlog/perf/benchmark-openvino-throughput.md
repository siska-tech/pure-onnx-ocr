---
status: completed
date: 2026-10-08
tags: [performance, openvino, benchmark, throughput, ppocrv6]
depends_on: perf/task-perf-008-run-many
---

# pure-onnx-ocr と OpenVINO の複数画像スループット比較（同一 PC・同一 ONNX）

[benchmark-openvino](benchmark-openvino.md) では、1 枚ずつ処理したときの処理時間を比べた。本書では、**複数の画像を処理するときのスループット（枚/秒）**を、両方のバックエンドで最も速い使い方にそろえて比べた。OpenVINO 側は、ストリーム数を変えて最速の設定を選んだ。

計測ツールは [tools/openvino-bench](../../../tools/openvino-bench/README.md) のスループットモード（`run_bench.py --throughput`）、集計結果は [tools/openvino-bench/results/summary-throughput.md](../../../tools/openvino-bench/results/summary-throughput.md) にある。

## 要点

16 枚を処理したときのスループット（枚/秒、3 ラウンドの中央値）。

| Model | pure-onnx-ocr（tract 0.23.8） | pure-onnx-ocr（修正版 tract） | OpenVINO（最速の設定） | 対 OpenVINO（0.23.8 / 修正版） |
|---|---:|---:|---:|---:|
| v6 tiny | 5.27 | 6.52 | 8.26 | 0.64 / **0.79** |
| v6 small | 1.51 | 1.82 | 2.25 | 0.67 / **0.81** |
| v6 medium | 0.34 | 0.42 | 0.42 | 0.81 / **0.99** |

pure-onnx-ocr はどちらも `run_many_from_images`、OpenVINO は複数画像を並列に処理する設定である。

- **今リリースされている tract（0.23.8）では、OpenVINO の 64〜81% のスループット**である。
- **tract の修正（[task-perf-006](task-perf-006-tract-intraop.md)）が入ると、79〜99%** になる。medium では OpenVINO と並ぶ。
- **1 枚ずつ処理する場合**（どちらも最速の 1 枚ずつの設定）は、OpenVINO の 45〜58%（0.23.8）、70〜85%（修正版）である。
- **メモリのピークは、OpenVINO の 18〜66%**（複数画像の場合）。ただし `run_many` のピークは、1 枚ずつのときの 2.5〜3.7 倍になる（後述）。
- **出力は、全設定・全モデル・全画像で完全に一致した**（領域数、文字列、box）。

## 計測条件

| 項目 | 条件 |
| :--- | :--- |
| PC | Intel Core i7-1360P（P コア 4 + E コア 8、16 スレッド）、Windows 11 Home 26200、AC 電源、電源プラン「バランス」 |
| pure-onnx-ocr | `442d13a`（`feature/openvino-bench`）、rustc 1.99.0、release ビルド。推論スレッド数は既定の 16 |
| tract | リリース版 0.23.8 と、修正版（upstream main `46056f93f` + task-perf-006 の 3 つの修正、ローカルのビルド） |
| OpenVINO | 2026.4.1（pip 版の C API）、CPU プラグイン、f32 |
| モデル | `PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx` の `inference.onnx`。両方のバックエンドが同じファイルを読む |
| 画像 | `general_ocr_002.jpg`（搭乗券）と `ja.jpg`（日本語）を 8 回ずつ繰り返した 16 枚を 1 パスとする |
| 回数 | 1 プロセスで 1 つの設定を測る。ウォームアップの 1 パス（推論計画のコンパイルを含む）の後、2 パスを計測する。設定の順番を入れ替えながら 3 ラウンド実行し、ラウンドの中央値をとる |
| プロセス | 両方とも EcoQoS を無効にし、優先度を HIGH にした |
| 出力の確認 | 各プロセスで、計測した全パスの結果がウォームアップのパスと一致することを確かめた。設定どうしの一致は、`pure-seq` を基準に比べた |

### 比べた設定

| 設定 | 内容 |
| :--- | :--- |
| `pure-seq` | `OcrEngine::run_from_image` を 1 枚ずつ（tract 0.23.8） |
| `pure-many` | `OcrEngine::run_many_from_images`（tract 0.23.8） |
| `pure-seq-tfix` / `pure-many-tfix` | 同じ処理を修正版の tract で |
| `ov-seq` | OpenVINO を 1 枚ずつ。[benchmark-openvino](benchmark-openvino.md) の `ov` と同じ設定（検出は LATENCY、認識は THROUGHPUT の request を並列に使う） |
| `ov-many` | OpenVINO で複数画像を並列に処理する。検出は THROUGHPUT で、推奨数の request を並列の画像で分け合う。認識は、全画像のバッチを THROUGHPUT の request で処理する。ストリーム数は OpenVINO の既定（検出 4、認識 4） |
| `ov-many-d4r8` / `ov-many-d8r16` | `ov-many` のストリーム数を、検出 4・認識 8 / 検出 8・認識 16 にしたもの |

- OpenVINO の複数画像の処理は、`run_many_from_images` と同じ構造にした。画像を推論スレッド数（16 枚）ずつのグループに分け、グループ内の検出を並列に実行し、その後グループ内の全画像の認識バッチをまとめて処理する。前処理と後処理は、本クレートのコードそのものである。
- OpenVINO のストリーム数は、事前のスイープ（v6 small と medium、1 ラウンド）で選んだ。候補は、検出・認識とも 2、4（既定）、8、16 と、その組み合わせ、検出だけ request を増やす、検出を LATENCY のままにする、などである。表には、上位の 2 つと既定を載せた。**モデルごとに最も速かった OpenVINO の設定**（tiny と small は `d8r16`、medium は `d4r8`）を「OpenVINO（最速の設定）」とした。

## 結果

### スループット（枚/秒、3 ラウンドの中央値）

| Model | 設定 | 枚/秒 | 最小〜最大 | 対 pure-seq | 対 OpenVINO 最速 | 平均使用コア数 | スレッド数 | WS のピーク MB | Private のピーク MB | 実行後の Private MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | pure-seq | 2.20 | 2.10〜2.52 | 1.00 | 0.27 | 4.3 | 36 | 187 | 181 | 104 |
| tiny | pure-many | 5.27 | 5.06〜5.32 | 2.40 | 0.64 | 14.4 | 20 | 597 | 664 | 101 |
| tiny | pure-seq-tfix | 3.45 | 3.02〜3.69 | 1.57 | 0.42 | 6.5 | 36 | 179 | 171 | 95 |
| tiny | **pure-many-tfix** | **6.52** | 6.30〜6.68 | 2.97 | **0.79** | 14.2 | 20 | 476 | 524 | 95 |
| tiny | ov-seq | 4.90 | 3.74〜5.63 | 2.23 | 0.59 | 7.2 | 58 | 369 | 589 | 519 |
| tiny | ov-many | 6.53 | 5.44〜7.49 | 2.97 | 0.79 | 10.9 | 45 | 1,517 | 1,765 | 821 |
| tiny | ov-many-d4r8 | 7.45 | 6.85〜8.26 | 3.39 | 0.90 | 11.4 | 49 | 1,589 | 1,846 | 909 |
| tiny | **ov-many-d8r16** | **8.26** | 8.04〜8.83 | 3.76 | 1.00 | 12.0 | 55 | 2,076 | 2,378 | 1,436 |
| small | pure-seq | 0.92 | 0.79〜0.94 | 1.00 | 0.41 | 6.6 | 33 | 443 | 439 | 253 |
| small | pure-many | 1.51 | 1.50〜1.54 | 1.64 | 0.67 | 14.7 | 17 | 985 | 1,106 | 249 |
| small | pure-seq-tfix | 1.18 | 1.08〜1.30 | 1.28 | 0.53 | 8.5 | 33 | 445 | 440 | 244 |
| small | **pure-many-tfix** | **1.82** | 1.80〜1.84 | 1.98 | **0.81** | 14.4 | 20 | 798 | 867 | 245 |
| small | ov-seq | 1.62 | 1.43〜1.80 | 1.75 | 0.72 | 9.1 | 58 | 705 | 1,035 | 865 |
| small | ov-many | 1.89 | 1.69〜2.08 | 2.05 | 0.84 | 11.7 | 45 | 3,168 | 3,559 | 1,307 |
| small | ov-many-d4r8 | 2.12 | 1.91〜2.24 | 2.30 | 0.94 | 12.6 | 49 | 3,340 | 3,804 | 1,574 |
| small | **ov-many-d8r16** | **2.25** | 2.06〜2.29 | 2.44 | 1.00 | 13.4 | 54 | 4,213 | 4,845 | 2,609 |
| medium | pure-seq | 0.23 | 0.22〜0.23 | 1.00 | 0.55 | 7.9 | 33 | 1,078 | 1,072 | 866 |
| medium | pure-many | 0.34 | 0.34〜0.35 | 1.48 | 0.81 | 14.9 | 17 | 2,939 | 3,227 | 873 |
| medium | pure-seq-tfix | 0.34 | 0.34〜0.34 | 1.47 | 0.81 | 11.0 | 33 | 1,073 | 1,069 | 870 |
| medium | **pure-many-tfix** | **0.42** | 0.42〜0.42 | 1.80 | **0.99** | 15.1 | 17 | 2,826 | 3,256 | 869 |
| medium | ov-seq | 0.40 | 0.40〜0.41 | 1.74 | 0.96 | 11.8 | 55 | 1,097 | 1,808 | 1,575 |
| medium | ov-many | 0.41 | 0.39〜0.41 | 1.77 | 0.97 | 12.9 | 42 | 3,926 | 4,692 | 2,423 |
| medium | **ov-many-d4r8** | **0.42** | 0.38〜0.42 | 1.82 | 1.00 | 13.6 | 46 | 4,114 | 4,970 | 2,730 |
| medium | ov-many-d8r16 | 0.42 | 0.39〜0.43 | 1.81 | 1.00 | 14.2 | 52 | 5,361 | 6,310 | 4,016 |

- 平均使用コア数は、計測した 2 パスの CPU 時間 ÷ 実時間。スレッド数とメモリは、全パスの後の値（ピークはプロセス全体での最大）。
- 実行後の Private は、計測を終えた時点のメモリで、推論計画のキャッシュなどの常駐分にあたる。

### 1 枚ずつと複数画像の比較（対 OpenVINO）

| Model | 1 枚ずつ（0.23.8 / 修正版、対 `ov-seq`） | 複数画像（0.23.8 / 修正版、対 OpenVINO 最速） |
| :--- | :--- | :--- |
| tiny | 0.45 / 0.70 | 0.64 / 0.79 |
| small | 0.57 / 0.73 | 0.67 / 0.81 |
| medium | 0.58 / 0.85 | 0.81 / 0.99 |

## 考察

### 1. どれだけ戦えているか

- **medium は、修正版の tract で OpenVINO と並んだ**（0.42 対 0.42 枚/秒）。計算量の大きいモデルほど、CPU を使い切った状態での差が小さい。
- **tiny と small は、修正版でも OpenVINO の約 8 割**である。モデルが小さいほど 1 回の推論が短く、1 推論あたりのオーバーヘッド（前処理、計画の実行、テンソルのコピー）と、カーネルの効率の差が目立つ。
- 1 枚ずつ処理する場合の差（0.70〜0.85、修正版）は、複数画像の場合より大きい。OpenVINO は 1 回の推論の中での並列化が上手く、本クレートは画像やバッチをまたいだ並列化に頼っているためである。

### 2. CPU の使い方

- `pure-many` の平均使用コア数は 14.2〜15.1 で、16 スレッドをほぼ使い切っている。OpenVINO は最速の設定で 12.0〜13.6 である。
- そのため、本クレートのスループットをこれ以上上げるには、**同じ計算をより少ない CPU 時間で行う**（カーネルの効率を上げる）必要がある。複数画像の場合、並列化の工夫で縮められる余地はほとんど残っていない。
- 修正版の tract でも認識が処理時間の大半を占める（[task-perf-006](task-perf-006-tract-intraop.md)）。残りの差を詰めるには、認識モデルの演算子の効率を調べる必要がある。

### 3. メモリ

- 複数画像の場合、本クレートのメモリのピークは OpenVINO（最速の設定）の **18〜66%** である（修正版の tract で、tiny 524 対 2,378 MB、small 867 対 4,845 MB、medium 3,256 対 4,970 MB）。
- 実行後の常駐分（Private）は、本クレートが OpenVINO の 7〜32% にとどまる。OpenVINO はストリームごとに作業領域を持つので、ストリーム数を増やすほどメモリが増える。
- ただし、**`run_many` のピークは、1 枚ずつのときの 2.5〜3.7 倍**になる（tiny 181 → 664 MB、small 439 → 1,106 MB、medium 1,072 → 3,227 MB）。16 枚を同時に検出するので、画像、検出の入出力、切り出しを 16 枚分同時に持つためである。実行後のメモリは 1 枚ずつのときと同じなので、増えるのは処理中だけである。
  - medium で 3.2 GB は、メモリの小さい環境では大きい。同時に処理する画像の数（今は推論スレッド数と同じ 16）を指定できるようにすれば、ピークを抑えられる（Follow-up）。

### 4. 公平性について

- 両方とも、同じ ONNX ファイル、同じ画像、同じ前処理と後処理のコードを使い、出力が完全に一致することを確かめた。
- 両方とも、複数画像を並列に処理する、それぞれの最速の使い方で比べた。OpenVINO は、ストリーム数のスイープで最も速かった設定を、モデルごとに選んだ。
- OpenVINO の非同期 API（`start_async` とコールバック）や、検出と認識のパイプライン化（画像 N の認識と画像 N+1 の検出を重ねる）は試していない。これらでさらに速くなる可能性はある。本クレートも同じ構造の処理なので、両方に同じ余地がある。
- 修正版の tract は、まだリリースされていない（upstream main と、tract に提案中の 3 つの修正 [sonos/tract#2976](https://github.com/sonos/tract/pull/2976)、[#2977](https://github.com/sonos/tract/pull/2977)、[#2978](https://github.com/sonos/tract/pull/2978)）。数値は、tract のリリース後に計測し直す。

## 再現手順

```powershell
bash scripts/fetch_fixtures.sh --all
cd tools\openvino-bench; cargo build --release
$T = "--backend ov --many --ov-det-hint THROUGHPUT --ov-det-requests 0 --ov-rec-hint THROUGHPUT --ov-rec-requests 0"
uv run --no-project --python 3.12 --with openvino==2026.4.1 run_bench.py `
  --models v6-tiny,v6-small,v6-medium --rounds 3 --runs 2 --throughput 8 `
  --add-config "ov-many-d4r8=$T --ov-det-streams 4 --ov-rec-streams 8" `
  --add-config "ov-many-d8r16=$T --ov-det-streams 8 --ov-rec-streams 16"
```

修正版の tract の行は、tract の依存を手元の clone に向けたビルドを `--add-config "pure-many-tfix=--backend pure --many" --config-exe "pure-many-tfix=<そのビルドの openvino-bench.exe>"` で加えて計測した（[task-perf-006](task-perf-006-tract-intraop.md) の「作業環境」）。

## 注意点

- この PC で、この時点に計測した値である。同じ設定でも、ラウンドによって最大 20〜30% ぶれた（表の最小〜最大）。比べるときは中央値を使う。
- バックグラウンドで VS Code などが動いている。すべての設定に同じように影響する。
- 画像は 2 種類を繰り返したものなので、検出の推論計画は 2 つしか使われない。画像サイズが毎回違う用途では、計画のコンパイルが増え、スループットは下がる（`OcrEngine::warmup` や計画のキャッシュの容量で対処する）。
