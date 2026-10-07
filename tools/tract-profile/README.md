# tract-profile

PaddleOCR の検出モデル（DBNet）を、pure-onnx-ocr と同じ手順で tract にコンパイルし、ノードごとの実行時間を測るツールです。tract の実行器（`multithread_tract_scope`）も本クレートと同じ方法で渡すので、1 スレッドと複数スレッドで、どの演算子が並列化の恩恵を受けているかを比べられます。

結果と考察は [task-perf-004](../../docs/devlog/perf/task-perf-004-det-profile.md) にあります。

## 使い方

```powershell
cd tools\tract-profile
cargo build --release

# <model.onnx> <height> <width> <threads> <runs>
.\target\release\tract-profile.exe ..\..\tests\fixtures\models\ppocrv6\small_det\inference.onnx 512 896 1 5
.\target\release\tract-profile.exe ..\..\tests\fixtures\models\ppocrv6\small_det\inference.onnx 512 896 8 5

# 遅いノードの一覧（op 名に DepthWise を含むもの）
$env:DETAIL = "DepthWise"; .\target\release\tract-profile.exe ..\..\tests\fixtures\models\ppocrv6\small_det\inference.onnx 512 896 1 2
```

- 出力は、tract の最適化後の op と、元の ONNX の op の組ごとの時間（ミリ秒、2 回のウォームアップの後の `runs` 回の平均）です。元の ONNX の op は、ノード名から分かる場合だけ表示します（分からない場合は `?`）。
- 入力は合成した値です。これらの演算子の時間は画素値に依存しません。
- 搭乗券（`general_ocr_002.jpg`）を前処理した検出の入力は `[1,3,512,896]` です。
- 本体と同じく、tract は Cargo.lock で 0.23.8 に固定しています。本体の tract を更新したら、ここも合わせます。
