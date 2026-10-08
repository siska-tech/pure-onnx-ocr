---
status: completed
priority: low
assignee: Backend
start_date: 2026-10-07
end_date: 2026-10-08
tags: [performance, wasm, multithread, browser]
depends_on: perf/task-perf-001-multithread
---

# タスク概要
ブラウザ（WebAssembly）でも、cross-origin isolated なページでは推論を複数スレッドで実行する。isolated でない環境では、これまでどおりシングルスレッド版で動かす。

## 背景
- task-perf-001 / 003 / 007 の高速化（認識バッチの並列実行、スレッド数の上限 16、重複コンパイルの排除）は、ネイティブのマルチスレッドでしか効いていなかった。`src/threading.rs` の `MULTITHREAD_SUPPORTED` は、wasm32 では常に `false` だった。
- 実測でも、ブラウザ向けの wasm（`bindings/wasm`）は v0.2.0 と v0.3.0 で速度に差がなかった（Node 22、PP-OCRv5、10 行・1000×700 の画像、2 回目以降で約 3.3〜3.6 秒）。
- デモのサイト（GitHub Pages）では、coi-serviceworker による PoC で、`crossOriginIsolated`、Worker 間の共有 `WebAssembly.Memory`、Hugging Face からの CORS 取得を確認済みだった（4 Worker で約 2.5〜2.8 倍）。

## 要件
- `wasm-bindgen-rayon` と tract のマルチスレッド実行器で、認識バッチの並列実行と行列積のマルチスレッド実行の両方を効かせる。
- stable でのビルド、`.cargo/config.toml` の simd128、ネイティブ版の動作と性能は変えない。
- バインディングに `initThreadPool(n)` を公開し、スレッド数の指定を `OcrEngineBuilder` と揃える。
- デモは `crossOriginIsolated` ならスレッド版、そうでなければシングルスレッド版を読み込む。
- 認識結果（テキスト・座標）がネイティブ版と一致すること。

## 設計

### 実行器: tract の `Executor::RayonGlobal`
- `wasm32-unknown-unknown` では `std::thread::spawn` が使えないので、エンジンごとの rayon プール（`Executor::MultiThread`）は作れない。
- tract-linalg 0.23.8 の `multithread-mm` には `Executor::RayonGlobal` がある。rayon のグローバルプール（wasm-bindgen-rayon の `initThreadPool` が作る）で、行列積（`chunked_dispatch_rayon`）と要素ごとの演算（`par_bin`、`par_chunks_mut`）を並列に実行する。
- `src/threading.rs`:
  - `MULTITHREAD_SUPPORTED` を `multithread && (!wasm32 || target_feature = "atomics")` にした。新しい feature は作らず、atomics 付きでビルドしたときだけ wasm でもスレッドを使う。
  - wasm + atomics では、`executor_for(n > 1)` が `Executor::RayonGlobal` を返す。`parallel_map`（認識バッチ、`run_many` の検出）は `RayonGlobal` のとき `par_iter` を使う。
  - スレッド数の既定値と上限は、グローバルプールのサイズ（`rayon::current_num_threads()`）にした。`RayonGlobal` はプールのサイズを変えられないので、それより大きい値はプールのサイズに抑える。
  - ネイティブ向けのコードは変えていない（`cfg` で分けた。共通部分の `.min(max_inference_threads())` はネイティブでは `usize::MAX`）。
- **呼ぶ順番の制約**: プールを作る前に rayon のグローバルプールに触れると、rayon は呼び出したスレッドだけの 1 スレッドのプールを作る（スレッドを作れない環境向けのフォールバック）。その後の `initThreadPool` は失敗する。そのため `initThreadPool` はエンジンの `build` より前に呼ぶ。呼ばずに build したエンジンは 1 スレッドで動く（確認済み）。

### ビルド
- 共有メモリには atomics 付きでコンパイルした標準ライブラリが必要で、`-Z build-std=panic_abort,std`（nightly のみ）を使う。
- `bindings/wasm/threads/` に `rust-toolchain.toml`（`nightly-2026-10-06`、rust-src）と `.cargo/config.toml` を置いた。そのディレクトリで cargo を実行したときだけ使われる。
  - リポジトリ直下には rust-toolchain を置かないので、ほかのビルドは stable のまま。
  - Cargo はリポジトリ直下の `.cargo/config.toml`（simd128）とこのディレクトリの設定を合わせて使う。
  - `target-dir` を `target/wasm-threads` に分け、stable のビルドの成果物と混ざらないようにした。
- **リンク引数は明示が必要だった**。`+atomics,+bulk-memory,+mutable-globals,+simd128` だけでは、メモリが共有にならなかった（nightly-2026-10-06）。その状態では `initThreadPool` が「#<Memory> could not be cloned」で失敗する。次の引数をすべて付けた。
  - `--shared-memory --import-memory --max-memory=2147483648`
  - `--export=__wasm_init_tls --export=__tls_size --export=__tls_align --export=__tls_base --export=__heap_base`（`__heap_base` がないと wasm-bindgen が「failed to find `__heap_base` for injecting thread id」で失敗する）
- **共有メモリの上限は 2 GiB**（32768 ページ）にした。
  - 共有メモリは作るときに最大サイズを宣言する必要があり、ブラウザはその分のアドレス空間を予約する。
  - 4 GiB の予約は、iOS Safari などメモリの少ないモバイル端末で失敗する例がある。Emscripten の既定値（`MAXIMUM_MEMORY`）も 2 GiB。
  - PP-OCRv6 medium はネイティブで約 1 GB を使う。スレッド数を増やすと、並列にコンパイルする推論計画や中間テンソルの分だけ増える（下の表）。
- `scripts/build_wasm.sh [--threads]` で、cargo と wasm-bindgen を実行して `examples/web/pkg` と `pkg-threads` を作る。
- CI に、固定した nightly でスレッド版をビルドし、wasm-bindgen の出力が共有メモリ（`shared:true`）になっていることを確かめるジョブ（`wasm-threads`）を追加した。

### バインディング
- `pure-onnx-ocr-wasm` に `threads` feature を追加した（`wasm-bindgen-rayon` 1.3.0 を optional の依存にする）。
  - wasm-bindgen-rayon 1.3.0 の要求は `wasm-bindgen >= 0.2.99` で、Cargo.lock の 0.2.105 のまま使えた。wasm-bindgen-cli 0.2.105 で生成できる。
  - 既定の方式（バンドラ向け）は `import('../../..')` でメインのモジュールを探すので、`--target web` では動かない。`no-bundler` feature（`import.meta.url` を使う）にした。
- 公開する関数:
  - `initThreadPool(n)`: wasm-bindgen-rayon のもの。シングルスレッド版では、何もせずに resolve する同じ名前の関数を公開する。同じ Worker のコードで両方のビルドを読み込めるようにするため。
  - `threadsSupported()`: スレッド版なら `true`。
  - `OcrEngineBuilder.inferenceThreads(n)`、`OcrEngine.inferenceThreads`: Rust の `inference_threads` と同じ。既定値はプールのサイズ。
- 生成物（`pkg-threads/`）:
  - `pure_onnx_ocr_wasm.js`、`pure_onnx_ocr_wasm_bg.wasm`、`*.d.ts`
  - `snippets/wasm-bindgen-rayon-38edf6e439f6d70d/src/workerHelpers.no-bundler.js`（rayon のスレッドになる Worker のスクリプト）

### デモ
- `worker.js`: `self.crossOriginIsolated` なら `pkg-threads` を読み込み、`initThreadPool(navigator.hardwareConcurrency)`（またはページで選んだ数）を呼ぶ。失敗した場合や isolated でない場合は `pkg` を読み込む。
- `serve.mjs --coi`: COOP `same-origin` と COEP `require-corp` を付けて配信する。
- GitHub Pages 向けの coi-serviceworker の手順を `examples/web/README.md` に書いた。ヘッダを付けないサーバー（`serve.mjs` の `--coi` なし）に coi-serviceworker を置き、ヘッドレス Chromium で次を確認した。
  - 1 回目の読み込みで Service Worker が登録され、再読み込みの後に `crossOriginIsolated === true` になる。
  - スレッド版が 4 スレッドで動き、rayon の入れ子 Worker（`workerHelpers.no-bundler.js`）にも COEP が付く。
- `bench.html` / `bench.mjs`: ヘッドレス Chrome（Playwright）で、両方のビルドとスレッド数ごとに、初回、2 回目以降、wasm のメモリを測る。出力をシングルスレッド版（完全一致）とネイティブ版（`examples/ocr_json.rs`、テキスト一致、座標 0.01 px 以内）に対して確かめる。

## 計測

### 条件
- ヘッドレス Chromium 141（Playwright 1.56）、`serve.mjs --coi`（COOP/COEP 付き）で配信した。
- 4 vCPU（Intel Xeon 2.10 GHz）、メモリ 15 GB のクラウドのコンテナ。**8 スレッドは論理 CPU 数を超えるので参考値**。
- 構成ごとに新しいページと Worker を作り、複数回実行した。初回（推論計画のコンパイルを含む）と、2 回目以降の中央値を記録した。
- メモリは、最後の実行の後の `WebAssembly.Memory` のサイズである。wasm のメモリは縮まないので、これがピークになる（JS のヒープや Worker のスタックは含まない）。
- モデル:
  - 結果 1: PP-OCRv5 mobile（デモのサイトに置いている単一ファイル版の `det.onnx` / `rec.onnx`、`ppocrv5_dict.txt`）。v0.3.0 のリリース版との比較に使った。5 回実行、同じ計測を 2 回繰り返して範囲で示す。
  - 結果 2: Hugging Face の PP-OCRv6 tiny / small / medium と PP-OCRv5 mobile（`inference.onnx` + `inference.yml`）。4 回実行、1 回ずつ。
- 既存の表で使っている搭乗券の画像（`general_ocr_002.jpg`）は、取得元（bcebos.com）にこの環境から接続できなかったので使っていない。
- 画像:
  - `sample.png`: 1000×700、英語 10 行（合成画像。依頼時の条件に合わせた）
  - `ja.jpg`: 1536×839、日本語 50 領域（`scripts/fetch_fixtures.sh --all` の `japan_2.jpg`）

### 結果 1: PP-OCRv5 mobile（単一ファイル版）と v0.3.0

`sample.png`（1000×700、10 行）

| ビルド | スレッド | 初回 | 2 回目以降 | 検出 | 認識 | wasm メモリ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.3.0（リリース版の pkg） | 1 | 8.16 s | 5.27 s | 1.27 s | 4.03 s | 189 MiB |
| シングルスレッド版 | 1 | 6.58〜6.67 s | 5.15〜5.24 s | 1.11〜1.15 s | 4.03〜4.11 s | 189 MiB |
| スレッド版 | 1 | 6.06〜6.24 s | 4.70〜5.38 s | 1.18〜1.24 s | 3.51〜4.08 s | 200 MiB |
| スレッド版 | 2 | 4.72〜4.82 s | 2.93〜3.24 s | 1.04〜1.14 s | 1.92〜2.09 s | 210〜213 MiB |
| スレッド版 | 4 | 3.29〜3.37 s | **1.79〜1.99 s** | 0.60〜0.65 s | 1.23〜1.36 s | 225〜232 MiB |
| スレッド版 | 8（参考） | 3.32〜3.37 s | 1.79〜2.04 s | 0.62〜0.73 s | 1.17〜1.30 s | 242〜259 MiB |

`ja.jpg`（1536×839、50 領域）

| ビルド | スレッド | 初回 | 2 回目以降 | 検出 | 認識 | wasm メモリ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.3.0（リリース版の pkg） | 1 | 9.74 s | 7.83 s | 0.93 s | 6.88 s | 269 MiB |
| シングルスレッド版 | 1 | 9.57〜10.87 s | 8.19〜8.59 s | 0.97〜1.01 s | 7.21〜7.49 s | 269 MiB |
| スレッド版 | 1 | 9.46〜10.79 s | 7.70〜8.03 s | 0.97〜0.99 s | 6.75〜7.01 s | 274〜276 MiB |
| スレッド版 | 2 | 6.42〜6.55 s | 4.14〜4.48 s | 0.88〜0.96 s | 3.24〜3.49 s | 283〜286 MiB |
| スレッド版 | 4 | 3.90〜4.15 s | **2.43〜2.65 s** | 0.59〜0.60 s | 1.89〜2.02 s | 294〜295 MiB |
| スレッド版 | 8（参考） | 4.27〜4.31 s | 2.56〜2.60 s | 0.57〜0.58 s | 1.99〜2.06 s | 312〜313 MiB |

- v0.3.0 の行は、デモのサイトの `v0.3.0/pkg` を `examples/web/pkg` に置き換えて測った（同じ実行の中で、スレッド版 4 スレッドは 2.20 s / 2.60 s だった）。
- **4 スレッドで、v0.3.0 のシングルスレッド版より 2 回目以降が 2.4 倍（sample）〜3.0 倍（ja）速くなった**。初回は 2.1〜2.3 倍。
- **出力は、全構成でネイティブ版と一致した**（テキストが完全一致、座標の差 0.01 px 以内。スレッド数を変えてもシングルスレッド版とビット単位で一致した）。
- 認識はバッチごとに並列に動くので、スレッド数にほぼ比例して縮む（4 スレッドで 3.0〜3.9 倍）。検出（tract の行列積の分割）は 2 スレッドではほとんど縮まず、4 スレッドで 1.7〜1.8 倍になった。2 スレッドで縮まない原因は調べていない。
- スレッド版を 1 スレッドで動かした場合は、シングルスレッド版と同等か少し速い（−10〜+3%）。atomics を有効にしたことによる遅れは見られなかった。
- メモリは、1 スレッド増えるごとに約 5〜10 MiB 増える。4 スレッドでシングルスレッド版の 1.1〜1.2 倍（+25〜43 MiB）だった。
- 8 スレッドは 4 vCPU のマシンでは 4 スレッドと同じか少し遅く、メモリだけが増える。

### 結果 2: PP-OCRv6 tiny / small / medium と PP-OCRv5 mobile

各セルは「初回 / 2 回目以降（中央値） / wasm メモリ」。

`sample.png`（1000×700、10 行）

| モデル | シングルスレッド版 | スレッド版 1 | 2 | 4 | 8（参考） |
| :--- | ---: | ---: | ---: | ---: | ---: |
| v6 tiny | 3.06 / 1.95 s / 160 MiB | 3.47 / 1.83 s / 162 MiB | 2.46 / 1.33 s / 154 MiB | 2.20 / **0.95 s** / 178 MiB | 2.40 / 1.04 s / 224 MiB |
| v6 small | 8.83 / 7.14 s / 352 MiB | 9.11 / 6.99 s / 324 MiB | 6.24 / 4.44 s / 359 MiB | 5.24 / **2.92 s** / 341 MiB | 4.99 / 2.55 s / 402 MiB |
| v6 medium | 36.99 / 32.71 s / 1,095 MiB | 34.97 / 31.82 s / 1,096 MiB | 25.97 / 22.79 s / 1,061 MiB | 16.20 / **13.33 s** / 1,070 MiB | 15.77 / 12.07 s / 1,206 MiB |
| v5 mobile | 7.79 / 6.01 s / 198 MiB | 7.43 / 5.73 s / 192 MiB | 5.41 / 3.45 s / 219 MiB | 3.89 / **1.97 s** / 231 MiB | 3.68 / 2.12 s / 290 MiB |

`ja.jpg`（1536×839、v6 は 55 領域、v5 は 50 領域）

| モデル | シングルスレッド版 | スレッド版 1 | 2 | 4 | 8（参考） |
| :--- | ---: | ---: | ---: | ---: | ---: |
| v6 tiny | 2.73 / 2.07 s / 128 MiB | 2.95 / 1.92 s / 130 MiB | 2.20 / 1.39 s / 133 MiB | 1.83 / **0.87 s** / 154 MiB | 1.88 / 0.92 s / 173 MiB |
| v6 small | 10.50 / 8.67 s / 277 MiB | 9.48 / 8.11 s / 279 MiB | 6.41 / 5.08 s / 289 MiB | 4.40 / **2.86 s** / 302 MiB | 4.27 / 2.89 s / 345 MiB |
| v6 medium | 43.29 / 40.66 s / 613 MiB | 43.92 / 41.34 s / 676 MiB | 28.04 / 25.69 s / 685 MiB | 16.57 / **14.18 s** / 847 MiB | 15.16 / 12.67 s / 833 MiB |
| v5 mobile | 10.75 / 8.63 s / 273 MiB | 9.88 / 8.12 s / 272 MiB | 6.45 / 4.68 s / 284 MiB | 4.70 / **2.61 s** / 291 MiB | 4.38 / 2.60 s / 318 MiB |

- **4 スレッドで、2 回目以降がシングルスレッド版の 2.0〜3.3 倍速くなった**（tiny 2.0〜2.4 倍、small 2.4〜3.0 倍、medium 2.5〜2.9 倍、v5 mobile 3.0〜3.3 倍）。初回は 1.4〜2.6 倍。
- tiny は検出の割合が大きく（4 スレッドで全体の約半分）、検出が認識ほど縮まないので、倍率が最も小さい。
- **出力は全構成でネイティブ版と一致した**（4 モデル × 2 画像 × 5 構成）。
- **medium のメモリのピークは最大 1,206 MiB（8 スレッド）で、共有メモリの上限 2 GiB に収まった**。4 スレッドでは 847〜1,070 MiB で、シングルスレッド版と比べて −2〜+38% だった。medium のメモリは、スレッド数よりも、どの幅の推論計画を何個キャッシュするかで変わる（sample の 1 スレッドで 1,095 MiB、ja で 613 MiB）。
- v5 mobile は、結果 1 の単一ファイル版より少し遅い（sample のシングルスレッド版で 6.0 s 対 5.2 s）。`inference.yml` の検出の設定（入力サイズの上限など）が違うためで、スレッドの効果は同じ傾向だった。

### 再計測の手順

```bash
scripts/fetch_fixtures.sh --all
mkdir -p examples/web/models && cp -r tests/fixtures/models/ppocrv6 tests/fixtures/models/ppocrv5 examples/web/models/
cp tests/fixtures/images/ja.jpg examples/web/models/ja.jpg
scripts/build_wasm.sh && scripts/build_wasm.sh --threads
mkdir -p native
for m in v6-tiny v6-small v6-medium v5-mobile; do
  dir=$(case $m in v6-*) echo ppocrv6/${m#v6-};; v5-mobile) echo ppocrv5/mobile;; esac)
  cargo run -q --release --example ocr_json -- --det tests/fixtures/models/${dir}_det \
    --rec tests/fixtures/models/${dir}_rec examples/web/models/ja.jpg > native/$m.json
done
node examples/web/bench.mjs --image models/ja.jpg --native-dir "$PWD/native" --threads 1,2,4,8
```

## 結果（要点）
- ブラウザで cross-origin isolated なページなら、推論が複数スレッドで動くようになった。4 スレッドで、2 回目以降がシングルスレッド版の **2.0〜3.3 倍**速い（PP-OCRv6 tiny / small / medium、PP-OCRv5 mobile）。v0.3.0 のリリース版と比べても 2.4〜3.0 倍（PP-OCRv5 mobile）。
- 出力は、全構成でネイティブ版と一致した。
- PP-OCRv6 medium のメモリのピークは最大 1,206 MiB で、共有メモリの上限（2 GiB）に収まった。
- isolated でないページ、または `pkg-threads` を読み込めない場合は、シングルスレッド版で動く（デモで確認済み）。
- 残り: 実機のモバイル端末（特に iOS Safari）で、2 GiB の共有メモリを確保できるか、入れ子の Worker が動くかの確認。
