//! WebAssembly bindings for `pure-onnx-ocr`.
//!
//! Build for browsers with:
//!
//! ```text
//! cargo build --release -p pure-onnx-ocr-wasm --target wasm32-unknown-unknown
//! wasm-bindgen --target web --out-dir pkg \
//!     target/wasm32-unknown-unknown/release/pure_onnx_ocr_wasm.wasm
//! ```
//!
//! JavaScript usage (models are fetched by the page; nothing touches a file
//! system):
//!
//! ```js
//! import init, { OcrEngineBuilder } from "./pkg/pure_onnx_ocr_wasm.js";
//! await init();
//! const bytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
//! const text = async (url) => (await fetch(url)).text();
//! const engine = new OcrEngineBuilder()
//!   .detModel(await bytes("models/det/inference.onnx"), await text("models/det/inference.yml"))
//!   .recModel(await bytes("models/rec/inference.onnx"), await text("models/rec/inference.yml"))
//!   .build();
//! const results = engine.run(await bytes("image.jpg"));
//! // [{ text: "BOARDING", confidence: 0.99, box: [[x, y] x 4], polygon: [[x, y], ...] }, ...]
//! ```

use js_sys::{Array, Object, Reflect};
use pure_onnx_ocr::{min_area_quad, DetLimitType, OcrResult, RecCropMode};
use wasm_bindgen::prelude::*;

fn to_js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

/// Builder mirroring `pure_onnx_ocr::OcrEngineBuilder` with in-memory inputs.
#[wasm_bindgen]
pub struct OcrEngineBuilder {
    inner: Option<pure_onnx_ocr::OcrEngineBuilder>,
}

impl OcrEngineBuilder {
    fn map(
        mut self,
        f: impl FnOnce(pure_onnx_ocr::OcrEngineBuilder) -> pure_onnx_ocr::OcrEngineBuilder,
    ) -> Self {
        let inner = self.inner.take().unwrap_or_default();
        self.inner = Some(f(inner));
        self
    }
}

impl Default for OcrEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl OcrEngineBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Some(pure_onnx_ocr::OcrEngineBuilder::new()),
        }
    }

    /// Detection model: `inference.onnx` bytes and `inference.yml` text.
    #[wasm_bindgen(js_name = detModel)]
    pub fn det_model(self, model: Vec<u8>, config_yaml: String) -> Self {
        self.map(|b| b.det_model_bytes(model).det_config_yaml(config_yaml))
    }

    /// Recognition model: `inference.onnx` bytes and `inference.yml` text
    /// (the dictionary embedded in the YAML is used).
    #[wasm_bindgen(js_name = recModel)]
    pub fn rec_model(self, model: Vec<u8>, config_yaml: String) -> Self {
        self.map(|b| b.rec_model_bytes(model).rec_config_yaml(config_yaml))
    }

    /// Plain text dictionary (one character per line), e.g. `ppocrv5_dict.txt`.
    #[wasm_bindgen(js_name = dictionaryText)]
    pub fn dictionary_text(self, text: String) -> Self {
        self.map(|b| b.dictionary_text(text))
    }

    /// Optional document orientation classifier (`PP-LCNet_x1_0_doc_ori`).
    #[wasm_bindgen(js_name = docOrientationModel)]
    pub fn doc_orientation_model(self, model: Vec<u8>, config_yaml: String) -> Self {
        self.map(|b| b.doc_orientation_model_bytes(model, config_yaml))
    }

    /// Optional text-line orientation classifier (`PP-LCNet_x0_25_textline_ori`).
    #[wasm_bindgen(js_name = textlineOrientationModel)]
    pub fn textline_orientation_model(self, model: Vec<u8>, config_yaml: String) -> Self {
        self.map(|b| b.textline_orientation_model_bytes(model, config_yaml))
    }

    /// Detection size limit (default 960, longest side).
    #[wasm_bindgen(js_name = detLimitSideLen)]
    pub fn det_limit_side_len(self, len: u32) -> Self {
        self.map(|b| b.det_limit_side_len(len))
    }

    /// `"max"` (default) bounds the longest side, `"min"` the shortest side.
    #[wasm_bindgen(js_name = detLimitType)]
    pub fn det_limit_type(self, limit_type: &str) -> Result<OcrEngineBuilder, JsError> {
        let limit_type = match limit_type {
            "max" => DetLimitType::Max,
            "min" => DetLimitType::Min,
            other => return Err(JsError::new(&format!("unknown limit type `{}`", other))),
        };
        Ok(self.map(|b| b.det_limit_type(limit_type)))
    }

    /// `"rotated"` (default) or `"axis"`.
    #[wasm_bindgen(js_name = cropMode)]
    pub fn crop_mode(self, mode: &str) -> Result<OcrEngineBuilder, JsError> {
        let mode = match mode {
            "rotated" => RecCropMode::Rotated,
            "axis" => RecCropMode::AxisAligned,
            other => return Err(JsError::new(&format!("unknown crop mode `{}`", other))),
        };
        Ok(self.map(|b| b.rec_crop_mode(mode)))
    }

    /// Recognition batch size (default 1; with tract, one crop per batch is
    /// fastest because it avoids padding crops to a common width).
    #[wasm_bindgen(js_name = recBatchSize)]
    pub fn rec_batch_size(self, size: usize) -> Self {
        self.map(|b| b.rec_batch_size(size))
    }

    /// Loads the models and returns a ready engine.
    pub fn build(mut self) -> Result<OcrEngine, JsError> {
        let inner = self.inner.take().unwrap_or_default();
        Ok(OcrEngine {
            inner: inner.build().map_err(to_js_error)?,
        })
    }
}

/// OCR engine. `run` is synchronous; call it from a Web Worker to keep the
/// page responsive.
#[wasm_bindgen]
pub struct OcrEngine {
    inner: pure_onnx_ocr::OcrEngine,
}

#[wasm_bindgen]
impl OcrEngine {
    /// Runs OCR on an encoded image (PNG, JPEG, ...).
    ///
    /// Returns `[{ text, confidence, box, polygon }, ...]`: `box` is the rotated
    /// 4-point rectangle (top-left first, clockwise). `polygon` contains the
    /// engine's bounding-box exterior without its repeated closing point;
    /// it is not the original detection contour. Coordinates are input pixels.
    pub fn run(&self, image: &[u8]) -> Result<Array, JsError> {
        let results = self.inner.run_from_bytes(image).map_err(to_js_error)?;
        results.iter().map(result_to_js).collect()
    }

    /// Same as `run`, plus stage timings in milliseconds and the detected
    /// page orientation: `{ results, timings, docOrientationAngle }`.
    #[wasm_bindgen(js_name = runWithMetrics)]
    pub fn run_with_metrics(&self, image: &[u8]) -> Result<Object, JsError> {
        let run = self
            .inner
            .run_with_metrics_from_bytes(image)
            .map_err(to_js_error)?;
        let results: Array = run
            .results
            .iter()
            .map(result_to_js)
            .collect::<Result<_, _>>()?;

        let ms = |d: std::time::Duration| JsValue::from_f64(d.as_secs_f64() * 1000.0);
        let timings = Object::new();
        set(&timings, "total", ms(run.timings.total))?;
        set(&timings, "imageDecode", ms(run.timings.image_decode))?;
        set(&timings, "orientation", ms(run.timings.orientation))?;
        set(
            &timings,
            "detection",
            ms(stage_total(&run.timings.detection)),
        )?;
        set(
            &timings,
            "recognition",
            ms(stage_total(&run.timings.recognition)),
        )?;

        let out = Object::new();
        set(&out, "results", results.into())?;
        set(&out, "timings", timings.into())?;
        set(
            &out,
            "docOrientationAngle",
            run.doc_orientation_angle
                .map(|a| JsValue::from_f64(a as f64))
                .unwrap_or(JsValue::NULL),
        )?;
        Ok(out)
    }
}

fn stage_total(stage: &pure_onnx_ocr::StageTimings) -> std::time::Duration {
    stage.preprocess + stage.inference + stage.postprocess
}

fn set(target: &Object, key: &str, value: JsValue) -> Result<(), JsError> {
    Reflect::set(target, &JsValue::from_str(key), &value)
        .map(|_| ())
        .map_err(|_| JsError::new("failed to build result object"))
}

fn result_to_js(result: &OcrResult) -> Result<JsValue, JsError> {
    let polygon = Array::new();
    let mut points: Vec<_> = result.bounding_box.exterior().points().collect();
    if points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    for point in points {
        let pair = Array::new();
        pair.push(&JsValue::from_f64(point.x()));
        pair.push(&JsValue::from_f64(point.y()));
        polygon.push(&pair);
    }
    // Minimum-area rotated rectangle (tl, tr, br, bl): a compact 4-point box,
    // which is usually what a UI draws.
    let quad = Array::new();
    if let Some(corners) = min_area_quad(&result.bounding_box) {
        for (x, y) in corners {
            let pair = Array::new();
            pair.push(&JsValue::from_f64(x));
            pair.push(&JsValue::from_f64(y));
            quad.push(&pair);
        }
    }
    let object = Object::new();
    set(&object, "text", JsValue::from_str(&result.text))?;
    set(
        &object,
        "confidence",
        JsValue::from_f64(result.confidence as f64),
    )?;
    set(&object, "box", quad.into())?;
    set(&object, "polygon", polygon.into())?;
    Ok(object.into())
}
