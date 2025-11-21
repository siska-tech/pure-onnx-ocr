# WASM環境でのモデルファイルの扱い方

WASM環境では、ファイルシステムに直接アクセスできません。モデルファイルを扱う方法は3つあります。

## 1. ユーザーがファイルをアップロード（現在の実装）

デモ（`examples/wasm-demo/index.js`）で実装されている方法です。

- ✅ ユーザーが任意のモデルを使用可能
- ✅ WASMモジュールサイズが小さい
- ❌ 毎回ファイルのアップロードが必要
- ❌ 初回読み込みに時間がかかる

```javascript
const detBytes = await readFileAsArrayBuffer(detFile);
const recBytes = await readFileAsArrayBuffer(recFile);
const dictBytes = await readFileAsArrayBuffer(dictFile);

const builder = new WasmOcrEngineBuilder()
    .det_model_bytes(new Uint8Array(detBytes))
    .rec_model_bytes(new Uint8Array(recBytes))
    .dictionary_bytes(new Uint8Array(dictBytes));

const engine = builder.build();
```

## 2. `include_bytes!` でビルド時に埋め込む

モデルファイルをWASMモジュールに直接埋め込みます。

- ✅ ファイルアップロードが不要
- ✅ 初回読み込みが高速
- ❌ WASMモジュールサイズが大きくなる（~20MB+）
- ❌ モデル変更時に再ビルドが必要

**使用方法：**

```rust
// src/wasm.rs または別のモジュールで
#[cfg(feature = "wasm")]
pub fn create_default_engine() -> Result<WasmOcrEngine, JsValue> {
    let det_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/det.onnx");
    let rec_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/rec.onnx");
    let dict_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/ppocrv5_dict.txt");
    
    WasmOcrEngineBuilder::new()
        .det_model_bytes(det_bytes)
        .rec_model_bytes(rec_bytes)
        .dictionary_bytes(dict_bytes)
        .build()
}
```

## 3. JavaScript側でfetchして読み込む

サーバーからモデルファイルをダウンロードします。

- ✅ WASMモジュールサイズが小さい
- ✅ モデルファイルの更新が容易
- ❌ 初回読み込みに時間がかかる（ネットワーク依存）
- ❌ CORS制約に注意が必要

**使用方法：**

```javascript
async function loadModels() {
    const [detBytes, recBytes, dictBytes] = await Promise.all([
        fetch('/models/det.onnx').then(r => r.arrayBuffer()),
        fetch('/models/rec.onnx').then(r => r.arrayBuffer()),
        fetch('/models/ppocrv5_dict.txt').then(r => r.arrayBuffer()),
    ]);
    
    const builder = new WasmOcrEngineBuilder()
        .det_model_bytes(new Uint8Array(detBytes))
        .rec_model_bytes(new Uint8Array(recBytes))
        .dictionary_bytes(new Uint8Array(dictBytes));
    
    return builder.build();
}
```

## 推奨される使い分け

- **開発・テスト環境**: 方法1（ファイルアップロード）または方法3（fetch）
- **本番環境（モデル固定）**: 方法2（`include_bytes!`で埋め込み）
- **本番環境（モデル更新あり）**: 方法3（fetch）

## 注意事項

- モデルファイル（`det.onnx` 4.5MB、`rec.onnx` 16MB）は大きいため、埋め込むとWASMモジュールが重くなります
- ブラウザのメモリ制限を考慮してください
- CDNを使用する場合、適切なCORS設定が必要です

