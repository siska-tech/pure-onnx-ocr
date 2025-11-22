//! WebAssembly bindings for pure-onnx-ocr.
//!
//! This module provides JavaScript-friendly APIs for running OCR in the browser.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::dictionary::RecDictionary;
#[cfg(feature = "wasm")]
use crate::engine::OcrEngine;
#[cfg(feature = "wasm")]
use crate::inference::{TractDetSession, TractRecSession};
#[cfg(feature = "wasm")]
use std::sync::Arc;

/// WASM-compatible OCR engine that loads models from memory.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmOcrEngine {
    engine: OcrEngine,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmOcrEngine {
    /// Runs OCR on an image from bytes (JPEG, PNG, etc.).
    ///
    /// # Arguments
    /// * `image_bytes` - Image file as bytes
    ///
    /// # Returns
    /// JSON string containing OCR results
    #[wasm_bindgen]
    pub fn run_from_bytes(&self, image_bytes: &[u8]) -> Result<String, JsValue> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode image: {}", e)))?;

        let results = self
            .engine
            .run_from_image(&img)
            .map_err(|e| JsValue::from_str(&format!("OCR failed: {}", e)))?;

        let json_results: Vec<JsOcrResult> = results
            .into_iter()
            .map(|r| {
                // Convert Polygon to coordinate array format
                let mut shape = Vec::with_capacity(1 + r.bounding_box.interiors().len());
                // Outer ring
                shape.push(
                    r.bounding_box
                        .exterior()
                        .points()
                        .map(|p| Point2D { x: p.x(), y: p.y() })
                        .collect(),
                );
                // Inner rings
                for interior in r.bounding_box.interiors() {
                    shape.push(
                        interior
                            .points()
                            .map(|p| Point2D { x: p.x(), y: p.y() })
                            .collect(),
                    );
                }
                JsOcrResult {
                    text: r.text,
                    confidence: r.confidence,
                    bounding_box: shape,
                }
            })
            .collect();

        let json_value = serde_wasm_bindgen::to_value(&json_results)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize results: {}", e)))?;

        js_sys::JSON::stringify(&json_value)
            .map(|s| s.as_string().unwrap_or_default())
            .map_err(|_| JsValue::from_str("Failed to stringify JSON"))
    }
}

/// WASM-compatible OCR engine builder.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmOcrEngineBuilder {
    det_model_bytes: Option<Vec<u8>>,
    rec_model_bytes: Option<Vec<u8>>,
    dictionary_bytes: Option<Vec<u8>>,
    det_limit_side_len: Option<u32>,
    det_unclip_ratio: Option<f64>,
    rec_batch_size: Option<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmOcrEngineBuilder {
    /// Creates a new builder instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmOcrEngineBuilder {
        console_error_panic_hook::set_once();
        WasmOcrEngineBuilder {
            det_model_bytes: None,
            rec_model_bytes: None,
            dictionary_bytes: None,
            det_limit_side_len: None,
            det_unclip_ratio: None,
            rec_batch_size: None,
        }
    }

    /// Sets the detection model bytes.
    #[wasm_bindgen]
    pub fn det_model_bytes(mut self, bytes: &[u8]) -> WasmOcrEngineBuilder {
        self.det_model_bytes = Some(bytes.to_vec());
        self
    }

    /// Sets the recognition model bytes.
    #[wasm_bindgen]
    pub fn rec_model_bytes(mut self, bytes: &[u8]) -> WasmOcrEngineBuilder {
        self.rec_model_bytes = Some(bytes.to_vec());
        self
    }

    /// Sets the dictionary bytes.
    #[wasm_bindgen]
    pub fn dictionary_bytes(mut self, bytes: &[u8]) -> WasmOcrEngineBuilder {
        self.dictionary_bytes = Some(bytes.to_vec());
        self
    }

    /// Sets the detection limit side length.
    #[wasm_bindgen]
    pub fn det_limit_side_len(mut self, len: u32) -> WasmOcrEngineBuilder {
        self.det_limit_side_len = Some(len);
        self
    }

    /// Sets the detection unclip ratio.
    #[wasm_bindgen]
    pub fn det_unclip_ratio(mut self, ratio: f64) -> WasmOcrEngineBuilder {
        self.det_unclip_ratio = Some(ratio);
        self
    }

    /// Sets the recognition batch size.
    #[wasm_bindgen]
    pub fn rec_batch_size(mut self, size: usize) -> WasmOcrEngineBuilder {
        self.rec_batch_size = Some(size);
        self
    }

    /// Builds the OCR engine.
    #[wasm_bindgen]
    pub fn build(self) -> Result<WasmOcrEngine, JsValue> {
        let det_bytes = self.det_model_bytes.ok_or_else(|| {
            JsValue::from_str("det_model_bytes is required. Call det_model_bytes() first.")
        })?;
        let rec_bytes = self.rec_model_bytes.ok_or_else(|| {
            JsValue::from_str("rec_model_bytes is required. Call rec_model_bytes() first.")
        })?;
        let dict_bytes = self.dictionary_bytes.ok_or_else(|| {
            JsValue::from_str("dictionary_bytes is required. Call dictionary_bytes() first.")
        })?;

        let det_session: Arc<dyn crate::inference::DetInference> =
            Arc::new(TractDetSession::load_from_bytes(&det_bytes).map_err(|e| {
                JsValue::from_str(&format!("Failed to load detection model: {}", e))
            })?);

        let rec_session: Arc<dyn crate::inference::RecInference> =
            Arc::new(TractRecSession::load_from_bytes(&rec_bytes).map_err(|e| {
                JsValue::from_str(&format!("Failed to load recognition model: {}", e))
            })?);

        let dictionary = RecDictionary::from_bytes(&dict_bytes)
            .map_err(|e| JsValue::from_str(&format!("Failed to load dictionary: {}", e)))?;

        // Build engine using internal APIs
        // We need to construct OcrEngine directly since OcrEngineBuilder uses file paths
        let mut config = crate::engine::OcrEngineConfig::default();
        if let Some(len) = self.det_limit_side_len {
            config.det_preprocessor.limit_side_len = len;
        }
        if let Some(ratio) = self.det_unclip_ratio {
            config.det_unclipper.unclip_ratio = ratio as f32;
        }
        if let Some(size) = self.rec_batch_size {
            config.rec_batch_size = size;
        }
        config.rec_postprocessor.blank_id = dictionary.blank_id();

        let dictionary = std::sync::Arc::new(dictionary);

        let engine = OcrEngine::new(
            std::path::PathBuf::from("<wasm-det>"),
            std::path::PathBuf::from("<wasm-rec>"),
            std::path::PathBuf::from("<wasm-dict>"),
            det_session,
            rec_session,
            (*dictionary).clone(),
            config,
        );

        Ok(WasmOcrEngine { engine })
    }
}

#[cfg(feature = "wasm")]
#[derive(serde::Serialize)]
struct Point2D {
    x: f64,
    y: f64,
}

#[cfg(feature = "wasm")]
#[derive(serde::Serialize)]
struct JsOcrResult {
    text: String,
    confidence: f32,
    bounding_box: Vec<Vec<Point2D>>, // Outer ring + inner rings as coordinate arrays
}

/// Creates a WASM OCR engine with embedded models.
///
/// This function uses `include_bytes!` to embed model files at compile time.
/// Note: This will significantly increase the WASM module size (~20MB+).
///
/// # Example
///
/// ```rust,no_run
/// // In your Rust code (when building WASM)
/// use pure_onnx_ocr::wasm::WasmOcrEngineBuilder;
///
/// let det_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/det.onnx");
/// let rec_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/rec.onnx");
/// let dict_bytes = include_bytes!("../../tests/fixtures/models/ppocrv5/ppocrv5_dict.txt");
///
/// let builder = WasmOcrEngineBuilder::new()
///     .det_model_bytes(det_bytes)
///     .rec_model_bytes(rec_bytes)
///     .dictionary_bytes(dict_bytes)
///     .build()?;
/// ```
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_engine_with_embedded_models(
    det_model_bytes: &[u8],
    rec_model_bytes: &[u8],
    dictionary_bytes: &[u8],
) -> Result<WasmOcrEngine, JsValue> {
    let builder = WasmOcrEngineBuilder::new()
        .det_model_bytes(det_model_bytes)
        .rec_model_bytes(rec_model_bytes)
        .dictionary_bytes(dictionary_bytes);

    builder.build()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
