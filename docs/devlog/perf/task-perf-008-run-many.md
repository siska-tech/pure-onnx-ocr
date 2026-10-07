---
status: completed
priority: medium
assignee: Backend
start_date: 2026-10-07
end_date: 2026-10-07
tags: [performance, throughput, multithread, api]
depends_on: perf/task-perf-003-default-threads
---

# タスク概要
複数の画像をまとめて処理する API（例: `OcrEngine::run_many`）を追加し、スループットを上げる。

## 背景
[benchmark-openvino](benchmark-openvino.md) より:

- 本クレートの平均使用コア数は 3.2〜4.8 で、OpenVINO は 7.1〜10.4 だった。
- 1 枚あたりの CPU 時間の総量は、OpenVINO とほぼ同じだった（small 約 6.2 対 5.9 コア秒）。
- 実時間の差の多くは「同時に何コア使えているか」の差で、特に検出は 1 回の推論が並列化されない。

検出は画像ごとに独立しているので、画像をまたいで並列に実行すればよく並列化できる。結果も変わらない。

## 要件
- 複数の画像の検出を、それぞれシングルスレッドの推論で並列に実行する。
- 認識は、全画像の切り出しをまとめて、既存のバッチ並列の仕組みで実行する。
- 結果は入力の順に返す。1 枚ずつ `run` した結果と完全に一致させる。
- 画像の数に比例してメモリが増えすぎないように、同時に処理する画像の数に上限を設ける。

## 計測
- 10 枚以上の画像で、1 枚ずつ `run` した場合と、`run_many` の場合の枚/秒を比べる。可能なら OpenVINO の `ov` 設定とも比べる。

## 結果（要点）

`OcrEngine::run_many_from_images(&[DynamicImage])` と `run_many_from_paths(&[P])` を追加した。

- 画像ごとの `Result<Vec<OcrResult>, OcrError>` を入力の順に返す。**結果は 1 枚ずつ `run_*` を呼んだ場合と完全に一致する**（文字列、信頼度のビット列、box）。
- 1 枚の失敗（読み込めない画像など）は、その画像の `Err` になるだけで、ほかの画像の処理は続く。
- 16 枚（2 枚 × 8）の処理で、**スループットは 1.45〜2.06 倍**になった（tract 0.23.8、i7-1360P、16 スレッド、3 ラウンドの中央値）。

| Model | 1 枚ずつ `run_from_image` | `run_many_from_images` | 倍率 | 参考: OpenVINO `ov`（1 枚ずつ） |
| :--- | ---: | ---: | ---: | ---: |
| v6 tiny | 2.45 枚/秒 | 5.04 枚/秒 | 2.06 | 4.45 枚/秒 |
| v6 small | 0.81 枚/秒 | 1.28 枚/秒 | 1.59 | 1.45 枚/秒 |
| v6 medium | 0.21 枚/秒 | 0.31 枚/秒 | 1.45 | 0.34 枚/秒 |

- tiny は、1 枚ずつ処理する OpenVINO を上回った。small と medium は、OpenVINO の 0.88〜0.91 倍である。
- OpenVINO の値は [benchmark-openvino](benchmark-openvino.md) のもの（1 枚ずつ、別の時間帯）。OpenVINO も複数画像を並列に処理すれば速くなるので、公平な比較ではない。

### 修正版の tract（task-perf-006）との組み合わせ

task-perf-006 の worktree（tract main + 3 つの修正）で、同じ計測をした。

| Model | 1 枚ずつ | `run_many_from_images` | 倍率 | OpenVINO `ov`（1 枚ずつ）に対して |
| :--- | ---: | ---: | ---: | ---: |
| v6 tiny | 3.37 枚/秒 | 6.41 枚/秒 | 1.90 | 1.44 倍 |
| v6 small | 1.21 枚/秒 | 1.73 枚/秒 | 1.43 | 1.19 倍 |
| v6 medium | 0.29 枚/秒 | 0.36 枚/秒 | 1.24 | 1.06 倍 |

- 修正版の tract では検出自体が並列化されるので、`run_many` の上乗せは小さくなる（1.24〜1.90 倍）。
- それでも、**3 モデルとも、1 枚ずつ処理する OpenVINO のスループットを上回った**。tract 0.23.8 で 1 枚ずつ処理した場合と比べると、2.6 倍（tiny）、2.1 倍（small）、1.7 倍（medium）である。
- 結果は、すべてのラウンドで 1 枚ずつの場合と一致した。

## 設計

1 枚分の処理を 2 段階に分けた（`src/engine.rs`）。

- **前段 `detect_regions`**: ページの向き → 検出 → 切り出し → 行の向き。画像ごとに独立している。
- **後段**: 認識 → `assemble`（領域と文字列の組み立て）。

`run_many` の処理:

1. 画像を、`inference_threads` 枚ずつのグループに分ける（同時に持つ画像と中間データの量を抑えるため）。
2. グループ内の画像の前段を、エンジンのスレッドプールで並列に実行する。画像の読み込み（`run_many_from_paths`）も並列になる。
   - 各画像の推論はシングルスレッドで実行する（`DetInferenceSession::run_single_threaded`、`OrientationClassifier::classify_single_threaded` を追加）。tract の並列行列演算を入れ子にすると、task-perf-001 で見た thread-local の `RefCell` の panic が起きるためである。
   - グループに画像が 1 枚しかない場合は、従来どおりエンジンのスレッドプールで推論する。
3. 全画像の認識バッチを、まとめて並列に実行する（`RecognitionPipeline::run_many`）。
   - バッチは画像ごとに、1 枚ずつ処理する場合と同じ分け方（縦横比で並べ替えてから `rec_batch_size` ずつ）で作る。画像をまたいで切り出しを混ぜないので、`rec_batch_size` が 2 以上でも結果は 1 枚ずつの場合と一致する。
   - バッチのエラーは、そのバッチを含む画像だけを失敗させる。

1 枚ずつの経路（`run_with_metrics_from_image_impl`）も、同じ `detect_regions` と `assemble` を使うように書き換えた。動作と計測値の意味は変わらない（切り出しの時間は、今までどおり認識の前処理に含める）。

## 作業ログ
- 2026-10-07: エンジンの 1 枚分の処理を `detect_regions` と `assemble` に分け、`run_many_from_images` と `run_many_from_paths` を追加した。
- 2026-10-07: 検出と向きの分類器に、シングルスレッドで実行する経路（crate 内部）を追加した。
- 2026-10-07: 結合テスト `run_many_matches_run_per_image` を追加した。v6 tiny で、向きの分類器（ページと行）ありの設定で、`rec_batch_size` 1 と 6 のそれぞれについて、4 枚（読み込めないパスを 1 つ含む）を処理し、1 枚ずつの結果との一致と、失敗した画像が `ImageDecode` になることを確かめる。
- 2026-10-07: `examples/throughput_bench.rs` を追加した。Windows の電力スロットリングを外す処理は `examples/common/power.rs` に切り出し、`ocr_bench` と共有した。
- 2026-10-07: CHANGELOG（Added）、README / README_en、docs/interface_design(_en) を更新した。

## テスト
- `cargo fmt --all --check`、`cargo clippy --workspace --all-targets -- -D warnings` が成功した。
- `cargo test --release --workspace` が成功した（lib 78、ppocrv6 13 ほか）。
- `throughput_bench` の各ラウンドで、`run_many_from_images` と `run_from_image` の結果が一致した。

## 残課題
- メモリのピークは [benchmark-openvino-throughput](benchmark-openvino-throughput.md) で計測した。`run_many` のピークは 1 枚ずつのときの 2.5〜3.7 倍（medium 1,072 → 3,227 MB）で、実行後のメモリは変わらない。それでも OpenVINO（複数画像の最速の設定）の 18〜66% である。同時に処理する画像の数を指定できるようにする（ROADMAP の Follow-up）。
