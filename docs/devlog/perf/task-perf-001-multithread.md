---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-03
end_date: 2026-10-03
tags: [performance, multithread, tract, rayon]
depends_on: ppocrv6/benchmark-v5-vs-v6
---

# タスク概要
[benchmark-v5-vs-v6](../ppocrv6/benchmark-v5-vs-v6.md) で、本クレートの推論が 1 コアしか使っていないことが分かった。CPU 時間と実時間の比で 0.98 コア分である。そこで推論をマルチスレッド化し、CPU 推論を高速化する。

ブランチ: `feature/multithread`（`feature/wasm-browser` から分岐）

## 結果（要点）

PP-OCRv6 を 8 スレッドで動かした場合の処理時間（i7-1360P、ミリ秒、5 回の中央値）。

| モデル | 画像 | 変更前（1 スレッド・バッチ 8） | 変更後（8 スレッド・バッチ 1） | 倍率 |
| :--- | :--- | ---: | ---: | ---: |
| v6 tiny | 搭乗券 | 1,406 | 454 | 3.1 倍 |
| v6 tiny | 日本語 | 1,497 | 513 | 2.9 倍 |
| v6 small | 搭乗券 | 4,218 | 1,319 | 3.2 倍 |
| v6 small | 日本語 | 5,270 | 1,236 | 4.3 倍 |
| v6 medium | 搭乗券 | 14,892 | 4,558 | 3.3 倍 |
| v6 medium | 日本語 | 17,346 | 4,501 | 3.9 倍 |
| v5 mobile | 搭乗券 | 4,700 | 1,306 | 3.6 倍 |
| v5 mobile | 日本語 | 5,647 | 1,209 | 4.7 倍 |
| v5 server | 搭乗券 | 39,694 | 11,744 | 3.4 倍 |
| v5 server | 日本語 | 44,628 | 11,079 | 4.0 倍 |

- 認識結果は変わらない。日本語 57 語での一致数も、搭乗券での出力も同じだった。
- メモリのピークは、medium で約 1.0GB だった。

## 調査と検討

### 1. tract の並列行列演算（`multithread-mm`）だけでは効果が小さい

tract-linalg には、rayon で行列演算を分割する `multithread-mm` 機能がある。既定の実行器は `SingleThread` で、本クレートではこの機能が無効だった。有効にしてスレッド数を変えたときの、パイプライン全体の処理時間は次のとおり（搭乗券、ミリ秒）。

| スレッド数 | v6 tiny | v6 small | v6 medium |
| ---: | ---: | ---: | ---: |
| 1 | 1,563 | 4,683 | 17,007 |
| 2 | 1,342 | 3,944 | 12,635 |
| 4 | 1,464 | 4,158 | 13,828 |
| 8 | 1,329 | 3,739 | 12,037 |
| 16 | 1,284 | 3,571 | 11,594 |

- 16 スレッドでも 1.15〜1.45 倍にしかならなかった。
- 理由:
  - PaddleOCR のモデルは小さな CNN で、1 回あたりの行列演算が小さい。
  - depthwise 畳み込みや要素ごとの演算は、行列演算ではないので並列化されない。
- 並列化の下限となる閾値（`set_threading_panel_threshold`、既定 64 パネル）を 16 や 0 に下げても、差は計測のばらつきの範囲だった。

### 2. 認識をバッチ単位で並列実行する（採用）

パイプライン全体の 80〜90% は認識が占めている。そして認識のバッチは互いに独立している。そこで、行列演算の中ではなく、**複数のバッチを同時に推論する**ことにした。

- 前処理・推論・後処理の各ステージで、全バッチを rayon で並列に処理する。ステージを分けているので、計測値は実時間のまま意味を持つ。
- 並列に動くバッチの推論では、tract をシングルスレッドで実行する（`RecInferenceSession::run_single_threaded`）。
  - 最初は入れ子のまま実行していた。すると tract の並列行列演算の待ち時間に、rayon のワーカーが別のバッチを奪って（work stealing）同じスレッドで実行した。tract が行列演算の作業領域を thread-local の `RefCell` で持っているため、`RefCell already borrowed` で panic した。
- 検出は 1 枚の画像につき 1 回の推論なので、従来どおり tract の並列行列演算を使う。

### 3. 認識のバッチサイズは 1 が最速

バッチサイズとスレッド数を変えて計測した（v6 small、搭乗券 / 日本語、ミリ秒）。

| スレッド数 | バッチ 8 | バッチ 4 | バッチ 2 | バッチ 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4,536 / 5,898 | 4,007 / 5,440 | 3,588 / 4,753 | **2,268 / 2,671** |
| 8 | 2,328 / 2,203 | 1,752 / 1,977 | 1,595 / 1,874 | **1,240 / 1,280** |

- シングルスレッドでも、バッチ 1 は バッチ 8 より約 2 倍速かった。
- 理由として考えられること:
  - バッチ内の切り出しは、最も幅の広いものに合わせてパディングされる。縦横比でソートしていても無駄が残る。
  - tract はバッチの次元を効率よく扱えていない可能性がある。
- バッチ 1 では、幅ごとに推論計画が作られる（幅は 32 の倍数に揃えている）。計画のキャッシュは 16 個に制限しているので、メモリは medium でも約 1.0GB に収まった。
- 日本語 57 語の精度は、バッチ 1 と 8 で同じだった（v5 mobile 39、v6 small 51、v6 medium 53）。
- PaddleOCR の既定（`batch_size: 6`）は GPU 向けの値と考えられる。tract の CPU 推論ではバッチ 1 が最速なので、**既定値を 8 から 1 に変更した**。

### 4. スレッド数の既定値

8 スレッドと 16 スレッドでは差がなかった。i7-1360P は 4 つの P コアと 8 つの E コアで 16 スレッドを持つ。そこで既定値は「論理 CPU 数、最大 8」とした。

## 実装メモ
- `Cargo.toml`:
  - 機能 `multithread`（既定で有効）を追加した。中身は `tract-linalg/multithread-mm` と `rayon` である。
  - tract-linalg を直接の依存に追加した。tract-onnx と同じ 0.23 系に統一される。
- `src/threading.rs`（新規）:
  - `executor_for(threads)`: スレッドプールを作る。作れない場合は警告を出してシングルスレッドに切り替える。
  - `run_with`: tract の `multithread_tract_scope` を使い、呼び出しの範囲だけ実行器を差し替える。tract のプロセス全体の既定値は変更しない。
  - `parallel_map`: 実行器のプールの上で `par_iter` を実行する。
  - `default_inference_threads()` と `MULTITHREAD_SUPPORTED` を公開した。
- セッション（検出・認識・方向分類器）に `executor` を持たせ、`set_inference_threads(n)` で変更できるようにした。
- `OcrEngineBuilder::inference_threads(n)` と `OcrEngineConfig::inference_threads` を追加した。エンジンは 1 つのプールを全セッションで共有する。
- 認識パイプラインで、バッチを並列に処理するようにした（上記 2）。
- `rec_batch_size` の既定値を 8 から 1 に変更した。
- `ocr_smoke` に `--threads` を追加し、使ったスレッド数を表示するようにした。
- `examples/ocr_bench` に、`--threads`、`--rec-batch-size`、`--panel-threshold` を追加した。
- WebAssembly（`target_arch = "wasm32"`）では、常にシングルスレッドで動く。`--no-default-features` でマルチスレッドを無効にした場合も同じ。

## 検証
- 全テストが成功した（lib 67、ppocrv6 11、ignored の small / medium / 傾き補正も含む）。
- 結果が変わらないことのテスト `thread_count_does_not_change_results` を追加した。tiny で 1 スレッドと 4 スレッドの結果（テキストと信頼度）が完全に一致することを確認する。
- 既存の `engine_can_be_shared_between_threads` は、既定の 8 スレッドのプールを持つエンジンを 3 スレッドから同時に使う形になり、これも成功した。
- `wasm32-unknown-unknown`（バインディングを含む）、`wasm32-wasip1`、`--no-default-features` のいずれもビルドできた。Node.js 上のブラウザ向け wasm でも同じ 37 領域を認識した。

## 残課題
- 検出は 1 回の推論なので、tract の並列行列演算の効果しかない（v6 medium で 2.4 秒から 2.0 秒）。tiny では処理時間の 60% 以上を検出が占めるようになった。タイル分割などで並列化できないか検討の余地がある。
- 行の向きの分類器も、バッチ単位で並列化できる。
- ブラウザのマルチスレッド化（`wasm-bindgen-rayon` と、tract の `RayonGlobal` 実行器）。
