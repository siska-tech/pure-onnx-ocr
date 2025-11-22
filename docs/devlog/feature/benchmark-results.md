# GPU バックエンド ベンチマーク結果

実行日: 2025-11-22  
テスト画像: `tests/fixtures/images/11.jpg`  
モデル: PP-OCRv5 (det.onnx, rec.onnx)

## ベンチマーク結果

### CPU バックエンド (tract-onnx)

```
[INFO] benchmark.total_seconds=22.919080
[INFO] benchmark.image_decode_seconds=22.919079
[INFO] benchmark.det.preprocess_seconds=0.050885
[INFO] benchmark.det.inference_seconds=5.486581
[INFO] benchmark.det.postprocess_seconds=0.026487
[INFO] benchmark.rec.preprocess_seconds=0.305357
[INFO] benchmark.rec.inference_seconds=16.942886
[INFO] benchmark.rec.postprocess_seconds=0.065097
```

**合計時間**: 22.92秒
- 検出推論: 5.49秒
- 認識推論: 16.94秒

### GPU バックエンド (wonnx)

**状態**: ❌ 実行不可

**エラー**: 
```
GPU initialization failed: Failed to load wonnx session: IR error: issue with data types: 
encountered parametrized dimensions 'DynamicDimension.3'; this is not currently supported 
(this may be solved by running onnx-simplifier on the model first)
```

**原因**: wonnx 0.5.1 は動的次元（DynamicDimension）をサポートしていない。PP-OCRv5 モデルは動的次元を使用しているため、現状では wonnx で直接実行できない。

### Auto モード（フォールバック動作確認）

```
[OcrEngineBuilder] GPU initialization failed: ... Falling back to CPU.
[INFO] benchmark.total_seconds=22.232604
[INFO] benchmark.det.inference_seconds=5.480657
[INFO] benchmark.rec.inference_seconds=16.258992
```

**合計時間**: 22.23秒（CPU フォールバック）

✅ **フォールバック機構は正常に動作**: GPU 初期化失敗時に CPU へ自動切り替えが確認された。

## 結論

1. **CPU バックエンド**: 正常に動作（22.92秒）
2. **GPU バックエンド**: 現状では実行不可（wonnx の動的次元制限）
3. **フォールバック機構**: 正常に動作確認済み

## 今後の対応

- wonnx の代替案検討（onnx-simplifier でモデル簡略化、または他の GPU ランタイム）
- または、wonnx の動的次元サポート待ち

