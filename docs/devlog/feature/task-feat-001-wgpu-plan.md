# wgpu (GPU バックエンド) 実装プラン

作成日: 2025-11-22
関連タスク: `task-feat-001-wgpu.md`

---

## 1. 現状分析

### 1.1 現行アーキテクチャ

```
OcrEngine
├── DetInferenceSession (tract-onnx, CPU)
│   └── キャッシュ: HashMap<(width, height), TypedRunnableModel>
├── RecInferenceSession (tract-onnx, CPU)
│   └── キャッシュ: HashMap<(batch_size, width), TypedRunnableModel>
└── 前処理・後処理 (バックエンド非依存)
```

### 1.2 既存の GPU 対応準備

- `Cargo.toml` に `wonnx = "0.5.1"` (optional) が定義済み
- `Backend` enum (`Cpu`, `Gpu`, `Auto`) が `engine.rs` に存在
- GPU 選択時は `"GPU backend is not yet implemented"` エラーを返す状態

### 1.3 課題

1. **抽象化層の欠如**: `DetInferenceSession` / `RecInferenceSession` が tract-onnx に直接依存
2. **型の結合**: `TypedRunnableModel<TypedModel>` が具象型としてハードコード
3. **非同期未対応**: wgpu/wonnx は async-friendly だが現行は同期 API

---

## 2. 実装方針

### 2.1 アプローチ選択

| 選択肢 | 概要 | 評価 |
|--------|------|------|
| A. wonnx 直接利用 | WebGPU ベースの ONNX ランタイム | ✅ 推奨: Cargo.toml に既存、ONNX 互換 |
| B. burn + wgpu | burn フレームワークの wgpu バックエンド | ❌ モデル再定義が必要 |
| C. カスタム wgpu シェーダー | 演算子ごとに WGSL 実装 | ❌ 工数過大 |

**結論**: `wonnx` を採用し、tract-onnx との共存を実現する。

### 2.2 設計原則

1. **Trait ベース抽象化**: 推論セッションを trait で抽象化し、バックエンド切り替えを可能にする
2. **Feature Flag 分離**: GPU 関連コードは `#[cfg(feature = "gpu")]` で分離
3. **後方互換性維持**: 既存の CPU API は変更しない
4. **フォールバック機構**: GPU 初期化失敗時は CPU へ自動フォールバック

---

## 3. 実装フェーズ

### Phase 1: 抽象化層の導入 (推定: 中規模)

#### 1.1 推論セッション Trait の定義

```rust
// src/inference/mod.rs (新規)

pub trait DetInference: Send + Sync {
    fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError>;
}

pub trait RecInference: Send + Sync {
    fn run(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput, OcrError>;
}
```

#### 1.2 既存実装のリファクタリング

```rust
// src/inference/tract_cpu.rs (既存コードを移動)

pub struct TractDetSession { /* 現行 DetInferenceSession */ }
pub struct TractRecSession { /* 現行 RecInferenceSession */ }

impl DetInference for TractDetSession { ... }
impl RecInference for TractRecSession { ... }
```

#### 1.3 ファイル構成

```
src/
├── inference/
│   ├── mod.rs           # Trait 定義、Backend enum
│   ├── tract_cpu.rs     # CPU 実装 (既存コード移動)
│   └── wonnx_gpu.rs     # GPU 実装 (Phase 2)
├── detection.rs         # 前処理・後処理のみ残す
├── recognition.rs       # 前処理・後処理のみ残す
└── engine.rs            # Trait オブジェクト経由で呼び出し
```

#### 1.4 OcrEngine の変更

```rust
pub struct OcrEngine {
    det_session: Arc<dyn DetInference>,
    rec_session: Arc<dyn RecInference>,
    // ... 他フィールド
}
```

---

### Phase 2: wonnx GPU バックエンド実装 (推定: 大規模)

#### 2.1 wonnx セッションの実装

```rust
// src/inference/wonnx_gpu.rs

#[cfg(feature = "gpu")]
pub struct WonnxDetSession {
    session: wonnx::Session,
    // キャッシュは不要 (wonnx は動的形状対応)
}

#[cfg(feature = "gpu")]
impl WonnxDetSession {
    pub async fn load(model_path: impl AsRef<Path>) -> Result<Self, OcrError> {
        let session = wonnx::Session::from_path(model_path).await?;
        Ok(Self { session })
    }
}

#[cfg(feature = "gpu")]
impl DetInference for WonnxDetSession {
    fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError> {
        // wonnx は async だが、ここでは block_on でラップ
        pollster::block_on(self.run_async(input))
    }
}
```

#### 2.2 テンソル変換ユーティリティ

```rust
// src/inference/tensor_convert.rs

/// tract Tensor → wonnx 入力形式
pub fn tensor_to_wonnx_input(tensor: &Tensor) -> HashMap<String, InputTensor> { ... }

/// wonnx 出力 → ndarray
pub fn wonnx_output_to_ndarray(output: &OutputTensor) -> Array3<f32> { ... }
```

#### 2.3 非同期 API の追加 (オプション)

```rust
impl OcrEngine {
    /// 非同期版 (GPU バックエンド向け最適化)
    #[cfg(feature = "gpu")]
    pub async fn run_from_path_async(&self, path: impl AsRef<Path>) -> Result<Vec<OcrResult>, OcrError> {
        // ...
    }
}
```

---

### Phase 3: ビルダー統合とフォールバック (推定: 小規模)

#### 3.1 OcrEngineBuilder の更新

```rust
impl OcrEngineBuilder {
    pub fn build(self) -> Result<OcrEngine, OcrError> {
        let effective_backend = match self.backend {
            Backend::Auto => self.detect_best_backend(),
            other => other,
        };

        match effective_backend {
            Backend::Cpu => self.build_cpu_engine(),
            #[cfg(feature = "gpu")]
            Backend::Gpu => self.build_gpu_engine(),
            #[cfg(not(feature = "gpu"))]
            Backend::Gpu => Err(OcrError::Config("GPU feature not enabled".into())),
        }
    }

    fn detect_best_backend(&self) -> Backend {
        #[cfg(feature = "gpu")]
        if self.is_gpu_available() {
            return Backend::Gpu;
        }
        Backend::Cpu
    }

    #[cfg(feature = "gpu")]
    fn is_gpu_available(&self) -> bool {
        // wgpu アダプタの列挙を試行
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            instance.request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .is_some()
        })
    }
}
```

#### 3.2 エラー型の拡張

```rust
pub enum OcrError {
    // ... 既存
    #[cfg(feature = "gpu")]
    GpuInit(String),
    #[cfg(feature = "gpu")]
    GpuInference(String),
}
```

---

### Phase 4: テストとベンチマーク (推定: 中規模)

#### 4.1 ユニットテスト

```rust
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    #[test]
    fn test_wonnx_det_session_load() { ... }

    #[test]
    fn test_wonnx_rec_session_load() { ... }

    #[test]
    fn test_gpu_cpu_output_equivalence() {
        // 同一入力に対して CPU/GPU 出力が許容誤差内で一致
    }
}
```

#### 4.2 ベンチマーク拡張

```bash
# ocr_smoke に --backend オプション追加
cargo run --bin ocr_smoke --features gpu -- image.jpg --benchmark --backend gpu
cargo run --bin ocr_smoke -- image.jpg --benchmark --backend cpu
```

#### 4.3 比較項目

| メトリクス | CPU (tract) | GPU (wonnx) |
|-----------|-------------|-------------|
| DBNet 推論時間 | baseline | 目標: 2-5x 高速化 |
| SVTR 推論時間 | baseline | 目標: 2-5x 高速化 |
| メモリ使用量 | baseline | 測定 |
| 初期化時間 | baseline | 測定 (GPU は初回遅延あり) |

---

## 4. 依存関係の更新

### 4.1 Cargo.toml 変更

```toml
[features]
default = []
gpu = ["wonnx", "pollster"]

[dependencies]
# 既存
tract-onnx = "0.20"

# GPU バックエンド (optional)
wonnx = { version = "0.5.1", optional = true }
pollster = { version = "0.3", optional = true }  # async → sync ブリッジ
```

### 4.2 対応プラットフォーム

| プラットフォーム | wgpu バックエンド | 対応状況 |
|-----------------|------------------|---------|
| Windows | DirectX 12 / Vulkan | ✅ |
| macOS | Metal | ✅ |
| Linux | Vulkan | ✅ |
| WASM | WebGPU | ⚠️ 別タスク (task-feat-002-wasm) |

---

## 5. リスクと対策

### 5.1 技術リスク

| リスク | 影響 | 対策 |
|--------|------|------|
| wonnx が一部 ONNX オペレータ未対応 | GPU 推論失敗 | 事前に DBNet/SVTR のオペレータ互換性検証 |
| GPU メモリ不足 | 大画像で OOM | バッチサイズ自動調整、CPU フォールバック |
| wonnx の非同期 API | 既存同期 API との不整合 | pollster で sync ラップ + async API 追加 |

### 5.2 検証項目 (Phase 0)

```bash
# wonnx で DBNet/SVTR が実行可能か事前検証
cargo run --example wonnx_compat_check --features gpu
```

```rust
// examples/wonnx_compat_check.rs
#[tokio::main]
async fn main() {
    let det_session = wonnx::Session::from_path("models/ppocrv5/det.onnx").await;
    match det_session {
        Ok(_) => println!("✅ DBNet: wonnx compatible"),
        Err(e) => println!("❌ DBNet: {}", e),
    }
    // SVTR も同様
}
```

---

## 6. マイルストーン

| フェーズ | 成果物 | 完了条件 |
|---------|--------|---------|
| Phase 0 | wonnx 互換性検証 | DBNet/SVTR が wonnx でロード・実行可能 |
| Phase 1 | 抽象化層 | 既存テストが全パス、API 変更なし |
| Phase 2 | GPU セッション実装 | `--features gpu` ビルド成功、基本推論動作 |
| Phase 3 | ビルダー統合 | `Backend::Gpu` 選択で E2E OCR 成功 |
| Phase 4 | テスト・ベンチマーク | CPU/GPU 出力一致、性能測定完了 |

---

## 7. 代替案検討

### 7.1 wonnx 以外の選択肢

| ライブラリ | 長所 | 短所 |
|-----------|------|------|
| **ort (onnxruntime-rs)** | 高い互換性、最適化済み | C++ FFI 依存 (Pure Rust 方針に反する) |
| **candle** | Hugging Face 製、活発な開発 | ONNX 直接サポートなし |
| **burn** | モダンな設計、複数バックエンド | ONNX インポート機能が限定的 |

### 7.2 判断

Pure Rust 方針を維持するため、`wonnx` を第一選択とする。wonnx で互換性問題が発生した場合は、`ort` への移行を検討する (FFI 許容の判断が必要)。

---

## 8. 作業順序 (推奨)

```
1. [Phase 0] wonnx 互換性検証スクリプト作成・実行
   ↓ 成功した場合のみ続行
2. [Phase 1] Trait 抽象化層の導入
   - src/inference/mod.rs 作成
   - 既存コードを tract_cpu.rs へ移動
   - OcrEngine を Trait オブジェクト化
   - 既存テスト全パス確認
3. [Phase 2] wonnx GPU セッション実装
   - WonnxDetSession / WonnxRecSession 実装
   - テンソル変換ユーティリティ作成
4. [Phase 3] OcrEngineBuilder 統合
   - Backend 切り替えロジック実装
   - Auto モード実装
5. [Phase 4] テスト・ベンチマーク
   - GPU ユニットテスト追加
   - ocr_smoke --backend オプション追加
   - 性能比較レポート作成
```

---

## 9. 参考リンク

- wonnx: https://github.com/webonnx/wonnx
- wgpu: https://github.com/gfx-rs/wgpu
- tract: https://github.com/sonos/tract
- ONNX オペレータセット: https://onnx.ai/onnx/operators/
