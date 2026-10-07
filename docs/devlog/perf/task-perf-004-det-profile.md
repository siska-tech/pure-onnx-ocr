---
status: completed
priority: high
assignee: Backend
start_date: 2026-10-07
end_date: 2026-10-07
tags: [performance, detection, tract, profiling]
depends_on: perf/task-perf-002-openvino-bench
---

# タスク概要
検出モデル（DBNet）のどの演算子が遅いのかを特定する。検出を速くする方法（task-perf-005 のタイル分割、task-perf-006 の tract の演算子内の並列化、グラフの書き換え）のどれに注力するかを、この結果で決める。

## 背景
[benchmark-openvino](benchmark-openvino.md) より:

- パイプライン全体の差の 6〜9 割は、検出の推論から来ている。
- 検出モデル単体では、本クレートは OpenVINO の 5〜8 倍遅い。
  - 1 スレッド同士でも 2.6〜4.8 倍遅い（カーネルの効率の差）。
  - 1 → 8 スレッドで 1.23〜1.29 倍にしかならない（並列化の差）。OpenVINO は 2〜2.5 倍になる。

## 調べること
1. **演算子ごとの時間**
   - tract CLI の `--profile --cost` で、v6 tiny / small / medium の検出を 1 スレッドと 8 スレッドで測る。入力は搭乗券を前処理した `[1,3,512,896]`。
   - depthwise 畳み込み、通常の畳み込み、`ConvTranspose`、`Resize`、要素ごとの演算（`HardSigmoid` など）、SE ブロック（`GlobalAveragePool`）に分けて集計する。
   - 8 スレッドにしたときに速くなる演算子と、速くならない演算子を分ける。
2. **画像全体を使う演算子の確認（タイル分割で結果が同じになるか）**
   - v6 tiny と small の検出には `GlobalAveragePool`（SE ブロック）が含まれる。画像全体の平均を使うので、タイルに分けると確率マップは元と一致しない。
   - 全モデルに含まれる `ReduceMean` の軸を調べる。チャネル方向（LayerNorm 系）なら、タイル分割しても結果は変わらない。空間方向なら変わる。
   - medium で `GlobalAveragePool` が見当たらないことも確かめる。

## 結果（要点）

- **検出の推論のボトルネックは depthwise 畳み込み（tract の `DepthWiseConv`）である**。
  - 1 スレッドで検出の 28〜36% を占め、**8 スレッドにしても速くならない**（0.95〜1.17 倍）。
  - 8 スレッドでは、行列演算が速くなる分、比率が 36〜50% に上がる。
  - 原因: tract 0.23.8 の depthwise は、x86_64 では SIMD のカーネルがなく（`tract_linalg::routines::depthwise_w_f32()` は arm64 にしかない）、タップ数が 5 以上の内側の領域（3x3、7x7、9x9）は汎用のスカラーの経路を通る。チャネルのループも並列化されていない。
  - 最も重いノードは、medium の 9x9（256 チャネル、128x224）1 つで 433 ms（1 スレッド）。計算量は約 1.19 GFLOP なので、約 2.7 GFLOP/s しか出ていない。small の 7x7（96 チャネル、128x224）も 1 つで 88 ms（約 3.0 GFLOP/s）。
- 並列化されない処理は、depthwise のほかに 3 つある。8 スレッドでは、これらと depthwise の合計が**検出時間の約 70%** になる。
  - 行列演算の入力のパック（`OptMatMulPack`）: 1 スレッドで 7〜12%
  - **ConvTranspose のバイアスを出力全体に展開する処理**（`MultiBroadcastTo`、1 ノード）: 1 スレッドで 4〜9%。バイアスを足すためだけに、`[1,24,256,448]` のテンソルを作っている。
  - 畳み込みの前の明示的なパディング（`Pad`）: 1 スレッドで 3〜6%
- 1x1 畳み込み（`OptMatMul`）は 2.5〜3.1 倍に並列化される。tract の並列化は、ここにしか効いていない。
- **画像全体の平均を取る演算子が、v6 の検出モデルすべてにある**。
  - `ReduceMean` の軸は全モデルで空間方向（axes=(2, 3)）だった（各 5 個）。tiny と small には、さらに `GlobalAveragePool` が 8 個ある。
  - そのため、**どのモデルもタイル分割すると確率マップが元と一致しない**。medium も例外ではない。

## 計測条件

- ツール: [tools/tract-profile](../../../tools/tract-profile/README.md)（今回作成）。本クレートと同じ手順でコンパイルし、`SimpleState::run_plan_with_eval` でノードごとの時間を測る。tract CLI（`tract-cli`）も使えるが、本クレートと同じ実行器で 1 スレッドと 8 スレッドを比べるために、自前で書いた。
- 入力: `[1,3,512,896]`（搭乗券を前処理した検出の入力と同じ形）、合成値。2 回のウォームアップの後、5 回の平均。
- PC: i7-1360P。tract 0.23.8。
- 合計（中央値）は、tiny 297 → 226 ms、small 678 → 434 ms、medium 2,524 → 1,916 ms（1 → 8 スレッド）。[benchmark-openvino](benchmark-openvino.md) のモデル単体の値（small 675 → 525 ms など）と同程度である。

## 演算子ごとの時間（ミリ秒、1 スレッド → 8 スレッド）

| tract の op（元の ONNX の op） | tiny | small | medium | 並列化 |
| :--- | ---: | ---: | ---: | :--- |
| `DepthWiseConv`（depthwise の Conv） | 83 → 81 | 220 → 188 | 914 → 960 | **されない** |
| `OptMatMul`（1x1 の Conv など） | 92 → 35 | 241 → 78 | 1,067 → 430 | される（2.5〜3.1 倍） |
| `OptMatMul`（その他の Conv） | 11 → 5 | 21 → 9 | 58 → 26 | される |
| `OptMatMulPack`（行列演算の入力のパック） | 35 → 33 | 64 → 59 | 172 → 185 | **されない** |
| `MultiBroadcastTo`（ConvTranspose のバイアス） | 27 → 24 | 38 → 34 | 92 → 100 | **されない** |
| `Pad`（Conv の前のパディング） | 18 → 18 | 28 → 27 | 86 → 92 | **されない** |
| `GeluExact`（Erf を使う GELU） | 4 → 2 | 10 → 4 | 29 → 11 | される |
| `Concat`、`NearestUpsample`（Resize）、`OptMaxPool` など | 各 1〜20 | | | ほぼされない |

8 スレッドでの、並列化されない 4 つの処理（depthwise、パック、バイアスの展開、パディング）の合計:

| Model | 合計 | うち depthwise |
| :--- | ---: | ---: |
| tiny | 157 / 226 ms（69%） | 81 ms（36%） |
| small | 308 / 434 ms（71%） | 188 ms（43%） |
| medium | 1,337 / 1,916 ms（70%） | 960 ms（50%） |

## OpenVINO との比較

- OpenVINO の検出モデル単体（1 スレッド）は、small 174 ms、medium 1,096 ms だった（[benchmark-openvino](benchmark-openvino.md)）。
- 本クレートは、**depthwise だけで** small 220 ms、medium 914 ms かかっている。1 スレッド同士で 2.6〜4.8 倍という「カーネルの効率」の差も、大部分は depthwise で説明できる。

## 判断（次にやること）

1. **task-perf-006 を最優先にする**。対象は次のとおり。どれも出力を変えない（または、変化を測ったうえで判断できる）。
   - (a) depthwise を**チャネル方向に並列化**する。チャネルごとの計算は独立しているので、出力は完全に一致する。8 スレッドで 3〜4 倍になれば、det 推論は small で約 130 ms、medium で約 700 ms 縮む見込み。
   - (b) depthwise に **x86_64 の SIMD（AVX2 / FMA）のカーネル**を追加する。今は約 3 GFLOP/s なので、数倍の余地がある。FMA で丸めの順序が変わるので、出力への影響を確かめる必要がある。
   - (c) ConvTranspose のバイアスを、出力全体に展開せずに足す（例: DeconvSum の出力の初期値にする）。足す順序が同じなら、出力は一致する。
   - (d) パックとパディングの並列化、またはパディングを畳み込みに取り込む。
   - (a)〜(c) がそろえば、small の det 推論は約 470 ms（16 スレッド、パイプライン内）から 200 ms 前後になる見込み。報告書がタイル分割で見込んだ値と同程度を、出力を変えずに達成できる。
2. **task-perf-005（タイル分割）は優先度を下げる**。v6 の検出モデルはすべて画像全体の平均を使うので、出力が一致しない。task-perf-006 で同程度の効果が見込めるので、006 の結果を見てから必要かどうかを判断する。
3. グラフの書き換え（ROADMAP の Follow-up）は見送る。重いのは depthwise そのもので、tract の別の演算子で置き換えても速くなる見込みがない。

## 作業ログ
- 2026-10-07: ONNX のグラフを `onnx` パッケージで調べた（演算子の種類、`ReduceMean` の軸、畳み込みの種類）。
  - tiny / small: Conv 83（1x1 が 61、depthwise が 17）、`GlobalAveragePool` 8、`ReduceMean` 5、ConvTranspose 2（2x2、stride 2）、Resize 6。
  - medium: Conv 122（1x1 が 57、depthwise が 21。9x9 の depthwise 8 と、7x7、1x7、7x1 などの大きいカーネル）、`ReduceMean` 5、`GlobalAveragePool` はない。
- 2026-10-07: `tools/tract-profile` を作成し、tiny / small / medium を 1 スレッドと 8 スレッドで計測した（上の表）。
- 2026-10-07: tract 0.23.8 の `ops/cnn/conv/depth_wise.rs` と `tract-linalg` の `routines.rs` を読み、depthwise に x86_64 の SIMD カーネルとチャネル方向の並列化がないことを確かめた。

## テスト
- コードの変更なし（調査のみ）。`tools/tract-profile` は独自の workspace で、本体のビルドと CI には入らない。
