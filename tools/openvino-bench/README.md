# openvino-bench

pure-onnx-ocr（tract）と OpenVINO Runtime を、同じ PC・同じ画像・同じ ONNX ファイル・同じ前処理と後処理で比べるベンチマークです。

結果と考察は [docs/devlog/perf/benchmark-openvino.md](../../docs/devlog/perf/benchmark-openvino.md) にあります。

## 公平性の担保

| 項目 | 方法 |
| :--- | :--- |
| モデル | 両方とも `tests/fixtures/models/ppocrv6/*/inference.onnx` を直接読みます。OpenVINO は `ov_core_read_model` で ONNX を読むので、IR への変換はしていません。`run_bench.py` は ONNX の SHA-256 を記録します。 |
| 前処理・後処理 | OpenVINO 側は Rust で書き、pure-onnx-ocr の公開 API をそのまま呼びます（`DetPreProcessor`、`DetPostProcessor::db_boxes`、`min_area_quad` + `crop_quad`、`RecPreProcessor`、`RecPostProcessor`）。推論の 2 か所だけを OpenVINO の infer request に置き換えています。 |
| 設定 | `inference.yml` から `OcrEngineBuilder::build` と同じ手順で設定を作ります。pure 側は実行のたびに、`engine.config()` と完全に一致することを確かめます（`config_matches_engine`）。 |
| 出力 | 両方のバックエンドで、認識した文字列と box が一致することを確かめます（`summary.md` の Output parity）。 |
| 実行条件 | どちらのプロセスも EcoQoS（電力スロットリング）を無効にし、優先度を HIGH にします。前者は `examples/ocr_bench.rs` と同じ処理です。 |
| 熱・クロックの変動 | 1 プロセスで 1 つのバックエンドと 1 つのモデルを測ります。ラウンドごとにバックエンドの順番をずらしながら交互に実行します。 |

## 計測項目

1 プロセスの流れは次のとおりです。

1. **読み込み**
   - pure: `OcrEngineBuilder::build`（ONNX の解析まで。推論計画はまだ作らない）。
   - OV: `read_model` と `compile_model`（det と rec の両方）。
2. **初回実行**: 画像ごとに 1 回実行します。pure では推論計画のコンパイルを含みます。
3. **ウォームアップ後**: `--runs` 回 × 画像の数だけ実行します。段階ごとの時間、CPU 時間と実時間の比（平均で何コア使ったか）、スループットを記録します。
4. その時点のメモリのピーク（Working Set と Private）とスレッド数を記録します。
5. **A. モデル単体**: 同一テンソルで計測します（`examples/ocr_bench.rs` と同じ入力）。
   - 検出: 搭乗券を前処理した `[1,3,H,W]`
   - 認識: 合成した `[8,3,48,320]` と `[1,3,48,320]`

段階の区切りは `OcrTimings` と同じです。認識の前処理には、切り出し（crop）の時間を含みます。

## バックエンドの設定（`run_bench.py` の `CONFIGS`）

| 名前 | 内容 |
| :--- | :--- |
| `pure` | pure-onnx-ocr の既定。推論スレッド数は min(論理 CPU 数, 8) = 8。認識はバッチ 1 で、複数のバッチを並列に実行します。 |
| `ov` | OpenVINO の主な比較対象。検出は `LATENCY`。認識は `THROUGHPUT` で、`OPTIMAL_NUMBER_OF_INFER_REQUESTS` 個の request を使い、バッチ 1 を並列に実行します。pure と同じ構造で、事前のスイープでは最速でした。 |
| `ov-latency` | OpenVINO の素の使い方。`LATENCY` で request は 1 つ。切り出した画像を 1 枚ずつ順番に推論します。 |

## 実行手順（Windows 11）

```powershell
# 1. フィクスチャ（モデルと画像）
cd pure-onnx-ocr
bash scripts/fetch_fixtures.sh --all

# 2. OpenVINO Runtime（pip 版。libs/ に openvino_c.dll が入っている）
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe openvino==2026.4.1

# 3. ビルド（release、既定の profile。ocr_bench と同じ）
cd tools\openvino-bench
cargo build --release

# 4. 計測（AC 電源で、ほかのアプリを閉じてから）
..\..\.venv\Scripts\python.exe run_bench.py --models v6-small,v6-medium,v6-tiny --rounds 5 --runs 5
#   -> results/summary.md, results/summary.json, results/raw/*.json
```

`results/` のうち、リポジトリに含めるのは `summary.md` と `summary.json`（[benchmark-openvino](../../docs/devlog/perf/benchmark-openvino.md) の根拠）だけです。`raw/` などは `.gitignore` で除外しています。

## 変更前と変更後の比較（A/B）

本クレートの性能改善は、変更前のビルドと変更後のビルドを**交互に**実行して比べます。同じコードでも、ラウンドによって絶対値が数十 % ぶれることがあるため、別々の時間に計測した値どうしは比べません。

```powershell
# 1. 変更前のコミットを worktree に展開し、そこで openvino-bench をビルドする
git worktree add ..\pure-onnx-ocr-base develop
cd ..\pure-onnx-ocr-base\tools\openvino-bench
cargo build --release
#    基準のコミットに tools/openvino-bench がない場合は、このディレクトリを
#    worktree の同じ場所にコピーしてからビルドする。

# 2. 変更後（この作業ツリー）でビルドし、両方を交互に計測する
cd <この作業ツリー>\tools\openvino-bench
cargo build --release
python run_bench.py --models v6-small,v6-medium,v6-tiny --rounds 3 --runs 5 `
  --baseline-exe ..\..\..\pure-onnx-ocr-base\tools\openvino-bench\target\release\openvino-bench.exe `
  --out results\ab-<タスク名>
#   -> summary.md の「A/B」表に、base（変更前）に対する倍率が出る

# 3. 片付け
git worktree remove ..\pure-onnx-ocr-base
```

- `--baseline-exe PATH` は、`pure` と同じ設定を別のビルドで実行する `base` を追加します。倍率は `base` を基準にします。
- 同じビルドで設定だけを比べる場合は、`--add-config` を使います（基準は `pure`）。

  ```powershell
  python run_bench.py --models v6-small --rounds 3 `
    --add-config t12="--backend pure --threads 12" --add-config t16="--backend pure --threads 16"
  ```

- `--baseline-exe` か `--add-config` を指定すると、既定では OpenVINO を計測しません（`openvino` パッケージも不要です）。OpenVINO も含める場合は `--configs base,pure,ov` のように指定します。
- 出力（領域数、文字列、box の座標）が基準と一致するかどうかは、summary.md の Output parity に出ます。

判定の基準:

| 項目 | 基準 |
| :--- | :--- |
| ラウンド数 | 3 以上（`--rounds 3`）。ラウンドごとに実行順を入れ替える |
| 指標 | ウォームアップ後の中央値と、初回の実行時間。A/B 表の倍率で判断する |
| 出力 | 既定の設定では、全モデル・全画像で基準と完全に一致すること |

## 単体での実行

単体で実行する場合は、OpenVINO の `libs` ディレクトリを `PATH` に入れておきます。

```powershell
$env:PATH = "<venv>\Lib\site-packages\openvino\libs;$env:PATH"
.\target\release\openvino-bench.exe --backend ov --model v6-small --ov-rec-hint THROUGHPUT --ov-rec-requests 0
.\target\release\openvino-bench.exe --backend pure --model v6-small
```

主なオプション:

| オプション | 対象 | 内容 |
| :--- | :--- | :--- |
| `--threads N` | pure | 推論スレッド数 |
| `--rec-batch-size N` | 両方 | 認識のバッチサイズ |
| `--ov-threads N` | OV | `INFERENCE_NUM_THREADS`（0 は自動） |
| `--ov-det-hint` / `--ov-rec-hint` | OV | `PERFORMANCE_HINT` |
| `--ov-rec-requests N` | OV | 並列に使う request の数（0 は最適値） |
| `--priority` | 両方 | プロセスの優先度 |
| `--json OUT` | 両方 | 結果の出力先 |

`run_bench.py` の主なオプション:

| オプション | 内容 |
| :--- | :--- |
| `--models` | 計測するモデル（`v6-small,v6-medium,v6-tiny` など） |
| `--configs` | 計測する設定（既定: `pure,ov,ov-latency`。A/B モードでは基準と追加した設定） |
| `--rounds` / `--runs` | ラウンド数と、1 プロセスあたりのウォームアップ後の回数 |
| `--baseline-exe PATH` | 別のビルドで `pure` を実行する `base` を追加し、基準にする |
| `--add-config NAME="ARGS"` | 任意の引数で設定を追加する（複数指定できる） |
| `--no-single-thread` | 1 スレッドでのモデル単体の計測を省く（OpenVINO を含む場合だけ実行される） |
| `--out DIR` | 結果の出力先（既定: `results`） |
