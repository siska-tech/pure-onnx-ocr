---
status: progress
priority: high
assignee: Backend
start_date: 2026-10-07
end_date:
tags: [performance, detection, tract, upstream, depthwise]
depends_on: perf/task-perf-004-det-profile
---

# タスク概要
tract の depthwise 畳み込みを中心に、検出で並列化されていない演算子を速くする（並列化と SIMD 化）。tract 本体への PR として提案する。

## 背景
- tract の並列化は行列演算の中だけで、検出は 8 スレッドでも 1.23〜1.29 倍にしかならない（[benchmark-openvino](benchmark-openvino.md)、task-perf-001）。OpenVINO は演算子の中でも空間方向に分割して、2〜2.5 倍に伸びる。
- この方法なら出力は変わらない。task-perf-005 のタイル分割と違い、既定の動作に入れられる。本クレート以外の tract の利用者にも効く。
- [task-perf-004](task-perf-004-det-profile.md) で、検出の最大のボトルネックは depthwise 畳み込みだと分かった。x86_64 では SIMD のカーネルがなく、チャネル方向の並列化もない。
- tract 側のレビューに時間がかかるので、fork での計測を先に進め、PR は効果を確かめたものから出す。

## 要件

[task-perf-004](task-perf-004-det-profile.md) のプロファイルにもとづき、効果の大きい順に進める。8 スレッドでは、次の 4 つの並列化されない処理が検出時間の約 70% を占める。

| 順 | 対象 | 内容 | 出力 | 見込める効果（small / medium、det 推論） |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `DepthWiseConv` | チャネル方向に並列化する（tract-core の `ops/cnn/conv/depth_wise.rs` のチャネルのループ） | 完全に一致する | −130 ms / −700 ms |
| 2 | `DepthWiseConv` | x86_64 の SIMD（AVX2 / FMA）のカーネルを追加する。タップ数の多い内側の領域（3x3、7x7、9x9）に効く形にする | FMA で丸めが変わる可能性。変化を測る | 1 スレッドで数倍 |
| 3 | `MultiBroadcastTo`（ConvTranspose のバイアス） | バイアスを出力全体に展開せず、DeconvSum の初期値などで足す | 足す順序が同じなら一致する | −34 ms / −100 ms |
| 4 | `OptMatMulPack`、`Pad` | 並列化する。またはパディングを畳み込みに取り込む | 一致する | −50 ms / −200 ms 程度 |

- 1 と 3 は小さい変更で、出力も変わらない。まずこの 2 つを tract の fork で実装し、本クレートの検出の時間と出力の一致を確かめてから、tract に PR を出す。
- 2 は tract-linalg への追加になる。arm64 には `depthwise_w_f32` があるので、同じ枠組み（`routines.rs` の `Func::DepthwiseW`）に x86_64 版を追加する形で提案する。ただし、今の枠組みはタップ数 4 以下の領域にしか使われていないので、呼び出し側も変える必要がある。
- tract の作業領域が thread-local の `RefCell` であることに注意する（task-perf-001 で、入れ子の rayon 実行による panic があった）。depthwise は行列演算の作業領域を使わないが、並列化には `multithread_tract_scope` で渡される実行器を使う。
- 計測には [tools/tract-profile](../../../tools/tract-profile/README.md)（演算子ごと）と [tools/openvino-bench](../../../tools/openvino-bench/README.md)（パイプライン全体の A/B）を使う。tract の fork は、計測用に `[patch.crates-io]` で差し替える。

## 完了の条件
- tract に PR を出し、取り込まれたら本クレートの tract を更新して効果を計測する。
- 取り込まれるまでの間、fork の tract に依存して公開することはしない。

## 結果（要点）

- **upstream の tract main（未リリースの 0.23.9-pre）では、計画した 4 つのうち 3 つがすでに解決されていた**。
  - depthwise のチャネル方向の並列化（対象 1）
  - ConvTranspose のバイアスを出力全体に展開する処理の削除（対象 3）
  - 畳み込みの前の `Pad` の大部分の削除（対象 4 の一部）
- 残りを、tract の手元の clone（ブランチ `x86-depthwise`）で実装した。**どれも出力をビット単位で変えない**。
  1. **depthwise の x86_64 カーネル**（`linalg/src/x86_64/depthwise.rs`、新規）。AVX の乗算と加算で、FMA は使わない。スカラーの経路と同じ演算を同じ順序で行うので、結果がビット単位で一致する。arm64 と wasm にある `depthwise_w_f32` と同じ枠組みに登録した。
  2. **カーネルに渡すタップ数の上限を 64 から 128 に上げた**（`core/src/ops/cnn/conv/depth_wise.rs` の `MAX_VECTORISED_TAPS`）。上限が 7x7（49 タップ）向けだったため、medium の 9x9（81 タップ）はカーネルを使えていなかった。
  3. **行列演算の入力のパックをパネル単位で並列化した**（`linalg/src/frame/pack.rs` の `pack_tensor_view`）。パックは値のコピーなので、結果は同じになる。
- パイプライン全体では、tract 0.23.8 に対して**合計が 29〜43% 縮み、出力は完全に一致した**。
  - v6 small（搭乗券）は 1,066 → 770 ms で、OpenVINO（585 ms）の 1.32 倍になった。**ROADMAP の目標（1.5 倍以下）に届く**。
  - v6 medium（搭乗券）は 4,055 → 2,737 ms で、OpenVINO（2,466 ms）の 1.11 倍になった。
- ただし、本クレートで使えるのは、tract の次のリリースが出てからになる。1〜3 は tract への PR として提案する。

## 検出の推論（tools/tract-profile、`[1,3,512,896]`、ミリ秒）

| Model | スレッド | tract 0.23.8 | tract main | main + 1 | main + 1 + 2 + 3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 | 297 | 251 | 180 | 188 |
| tiny | 8 | 226 | 135 | 120 | 119 |
| small | 1 | 678 | 534 | 376 | 374 |
| small | 8 | 434 | 258 | 219 | 202 |
| medium | 1 | 2,524 | 2,522 | 2,236 | 1,711 |
| medium | 8 | 1,916 | 1,117 | 1,131 | 903 |

depthwise の時間（1 スレッド）は、tiny 82 → 15 ms、small 189 → 30 ms、medium 953 → 124 ms（2 も含めた場合）になった。行列演算の入力のパック（8 スレッド）は、small 60 → 33 ms、medium 197 → 96 ms になった。

### 出力の一致（ビット単位）

tract-profile に、出力テンソルのビット列のハッシュ（`output_hash`）を表示する機能を追加して確かめた。

- tract main と「main + 1 + 2 + 3」のハッシュは、全モデルで一致した。1 スレッドと 8 スレッドでも一致した。カーネルの登録を外した main とも一致した。
- tract 0.23.8 と main のハッシュは一致しなかった。main のほかの変更（バイアスやパディングの扱い）で、浮動小数点の演算順序が変わったためと考えられる。ただし、OCR の結果（領域数、文字列、box の座標）は、全モデル・全画像で一致した（下の A/B）。

## パイプライン全体（openvino-bench の A/B、tract 0.23.8 が基準、交互に 3 ラウンド、ウォームアップ後の中央値）

| Model | 画像 | 合計（0.23.8 → 修正版） | 倍率 | det 推論 | rec 推論 | 初回 | 参考: tract main のみの合計 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| v6 tiny | 搭乗券 | 402 → 267 | 0.66 | 0.51 | 0.88 | 0.73 | 0.77 |
| v6 tiny | 日本語 | 423 → 298 | 0.70 | 0.52 | 0.90 | 0.59 | 0.77 |
| v6 small | 搭乗券 | 1,066 → 770 | 0.72 | 0.48 | 0.88 | 0.72 | 0.83 |
| v6 small | 日本語 | 1,145 → 804 | 0.70 | 0.44 | 0.84 | 0.70 | 0.81 |
| v6 medium | 搭乗券 | 4,055 → 2,737 | 0.68 | 0.39 | 0.90 | 0.69 | 0.78 |
| v6 medium | 日本語 | 4,533 → 3,214 | 0.71 | 0.39 | 0.93 | 0.68 | 0.80 |
| v5 mobile | 搭乗券 | 1,221 → 772 | 0.63 | 0.57 | 0.62 | 0.69 | 0.95 |
| v5 mobile | 日本語 | 1,176 → 736 | 0.63 | 0.57 | 0.59 | 0.65 | 0.93 |
| v5 server | 搭乗券 | 9,527 → 5,478 | 0.57 | 0.50 | 0.61 | 0.64 | 0.88 |
| v5 server | 日本語 | 9,911 → 5,775 | 0.58 | 0.50 | 0.62 | 0.59 | 0.86 |

- PP-OCRv5 は、認識モデルにも depthwise 畳み込みがあるので、認識も約 40% 縮んだ。
- メモリのピークは変わらない（v6 medium 1,044 → 1,039 MB）。平均使用コア数は増えた（v6 small 6.7 → 8.4）。
- 出力（領域数、文字列、box）は、全モデル・全画像で基準と完全に一致した。

### OpenVINO との比較（[benchmark-openvino](benchmark-openvino.md) の値と比べた目安）

| Model | 画像 | 報告書の時点の倍率 | 修正版 tract の倍率 |
| :--- | :--- | ---: | ---: |
| v6 small | 搭乗券 | 2.35 | 約 1.32 |
| v6 small | 日本語 | 1.94 | 約 1.02 |
| v6 medium | 搭乗券 | 2.02 | 約 1.11 |
| v6 medium | 日本語 | 1.80 | 約 1.00 |
| v6 tiny | 搭乗券 | 2.70 | 約 1.48 |

OpenVINO の値は報告書のもので、同じ時間帯に交互に計測したものではない。正式な比較は、tract のリリース後に openvino-bench で計測し直す。

## 進め方

1. tract への PR は、独立した 3 つに分ける（1: x86_64 カーネル、2: タップ数の上限、3: パックの並列化）。どれも出力を変えず、単体テストを付けている。
   - tract の AGENTS.md のルール（コメントは最小限、コミットメッセージは短い 1 段落、PR の要約は 1〜2 文、レビューへの返答は人間が行う）に従う。
   - 2026-10-08 に PR を作成した（作業ログを参照）。
2. tract の次のリリースが出たら、本クレートの tract を更新し、openvino-bench で計測し直して OpenVINO との比較を更新する。
3. 取り込まれるまでの間、fork の tract に依存して公開することはしない。

## 作業環境（手元）

- tract の clone: `C:\Users\Shion\Documents\Projects\tract`。upstream main `46056f93f` から、PR ごとのブランチを切った。
  - `x86-depthwise-kernel`（1）、`depthwise-81-taps`（2）、`parallel-pack-tensor-view`（3）
  - `x86-depthwise`: 3 つをマージした計測用のブランチ。worktree はこれを使う。
- 本クレートの worktree: `C:\Users\Shion\Documents\Projects\pure-onnx-ocr-tractmain`（`Cargo.toml` と tools の tract の依存を、上の clone へのパス指定に書き換えたもの。コミットしない）。
  - tract main は `0.23.9-pre` なので、`[patch.crates-io]` では差し替えられない（pre-release は `0.23` の指定と一致しない）。そのため、パス指定にした。

## 作業ログ
- 2026-10-07: tract の upstream を clone し、v0.23.8 以降の変更を確認した。depthwise の並列化とバイアスの展開の削除が、main にすでに入っていた。
- 2026-10-07: 本クレートの worktree で tract main をパス指定で使い、tract-profile と openvino-bench で計測した（検出 −36〜51%、合計 −5〜23%、出力は一致）。
- 2026-10-07: x86_64 の depthwise カーネル、タップ数の上限、パックの並列化を実装し、それぞれ単体テストを付けた（`matches_scalar_bit_for_bit`、`multithreaded_pack_matches_serial`）。
- 2026-10-07: tract-profile に `output_hash` を追加し、ビット単位の一致を確かめた。
- 2026-10-07: openvino-bench で、tract 0.23.8 と修正版を交互に 3 ラウンド計測した（上の表）。
- 2026-10-08: 3 つのブランチを upstream の最新の main（`252521b62`）に載せ直した（衝突なし）。tract-linalg 3,861 件、tract-core 355 件のテストが成功し、検出の出力のハッシュは載せ直す前と同じだった。
- 2026-10-08: `siska-tech/tract` に fork し、PR を作成した。
  - [sonos/tract#2976](https://github.com/sonos/tract/pull/2976): linalg: add an x86_64 depthwise_w kernel
  - [sonos/tract#2977](https://github.com/sonos/tract/pull/2977): core: hand 9x9 depthwise zones to the vectorised kernel
  - [sonos/tract#2978](https://github.com/sonos/tract/pull/2978): linalg: pack activation panels on the executor
  - レビューへの対応は、tract のルールに従い作者が行う。

## テスト
- tract: `cargo test --release -p tract-linalg --features multithread-mm`（3,861 件）と `cargo test --release -p tract-core`（354 件）が成功した。`cargo fmt --all --check` も通った。clippy の警告は 126 件で、変更前と同じ（すべて既存のもの）。
- 本クレート: 修正版 tract でのパイプライン全体の出力が、tract 0.23.8 と全モデル・全画像で一致した。
