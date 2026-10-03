//! # Pure ONNX OCR
//!
//! A Pure Rust OCR pipeline that mirrors the PaddleOCR DBNet + CTC recognition
//! stack and runs the PP-OCRv5 and PP-OCRv6 ONNX exports.
//! The crate exposes ergonomic builders and processing stages that let you
//! load ONNX models, prepare image batches, and decode recognition logits
//! without any C/C++ dependencies.
//!
//! Most consumers interact with [`OcrEngineBuilder`] to construct an
//! [`OcrEngine`], then call [`OcrEngine::run_from_path`] or
//! [`OcrEngine::run_from_image`].  Lower-level modules remain available
//! when you need to plug specific stages into an existing pipeline.

pub mod crop;
pub mod ctc;
pub mod detection;
pub mod dictionary;
pub mod engine;
mod onnx_model;
pub mod orientation;
pub mod paddle_config;
pub mod postprocessing;
pub mod preprocessing;
pub mod recognition;

/// Re-export of the CTC decoding utilities so applications can customise
/// post-processing while keeping consistent types.
pub use crop::{crop_quad, min_area_quad, Quad, RecCropMode};
pub use ctc::{CtcGreedyDecoder, CtcGreedyDecoderConfig, CtcGreedyDecoderError, DecodedSequence};
/// Re-export of detection inference helpers for direct DBNet integration.
pub use detection::{DetInferenceOutput, DetInferenceSession};
pub use dictionary::{DictionaryError, RecDictionary};
/// High-level façade providing an ergonomic OCR API.
pub use engine::{
    OcrEngine, OcrEngineBuilder, OcrEngineConfig, OcrError, OcrResult, OcrRunWithMetrics,
    OcrTimings, StageTimings, PADDLE_CONFIG_FILE, PADDLE_MODEL_FILE,
};
/// Geometry primitives surfaced at the crate root for convenience.
pub use geo_types::{Point, Polygon};
/// PaddleOCR `inference.yml` reader used for PP-OCRv5 / PP-OCRv6 model directories.
pub use orientation::{OrientationClassifier, OrientationError, OrientationPrediction};
pub use paddle_config::{ColorOrder, PaddleConfigError, PaddleInferenceConfig};
pub use postprocessing::{
    DetPolygonScaler, DetPolygonScalerConfig, DetPolygonUnclipper, DetPolygonUnclipperConfig,
    DetPostProcessor, DetPostProcessorConfig, DetPostProcessorError, DetScaleRounding,
    DetUnclipLineJoin,
};
pub use preprocessing::{
    DetLimitType, DetPreProcessor, DetPreProcessorConfig, DetPreProcessorError,
    PreprocessedDetInput, PreprocessedRecBatch, RecPreProcessor, RecPreProcessorConfig,
    RecPreProcessorError, RecTextRegion, IMAGENET_MEAN, IMAGENET_STD,
};
pub use recognition::{
    RecInferenceOutput, RecInferenceSession, RecPostProcessor, RecPostProcessorConfig,
    RecPostProcessorError,
};

use std::path::Path;
use tract_onnx::prelude::*;

const DBNET_DUMMY_SHAPE: [usize; 4] = [1, 3, 320, 320];
const SVTR_DUMMY_SHAPE: [usize; 4] = [1, 3, 48, 320];

/// Run a `tract-onnx` dummy inference against a DBNet detection model.
///
/// The helper loads `model_path`, feeds a zero-filled tensor with the
/// expected DBNet input shape, and returns the produced tensors.  The
/// logs include timing information for each optimisation step, which is
/// particularly helpful when first validating an ONNX export.
///
/// # Examples
///
/// ```no_run
/// use pure_onnx_ocr::run_dbnet_dummy_inference;
///
/// let outputs = run_dbnet_dummy_inference("models/ppocrv5/det.onnx")
///     .expect("model should load and execute");
/// assert!(!outputs.is_empty());
/// ```
///
/// # Errors
///
/// Returns [`tract_onnx::prelude::TractError`] when the model cannot be
/// loaded, optimised, or executed with the provided dummy tensor.
pub fn run_dbnet_dummy_inference(model_path: impl AsRef<Path>) -> TractResult<TVec<Tensor>> {
    let dummy_input: Tensor = tract_ndarray::Array4::<f32>::zeros(DBNET_DUMMY_SHAPE)
        .into_dyn()
        .into();
    run_dummy_inference(model_path, dummy_input, "DBNet")
}

/// Run a `tract-onnx` dummy inference against an SVTR recognition model.
///
/// The helper constructs a synthetic sinusoidal input tensor to exercise
/// the model, runs end-to-end optimisation, and returns the resulting
/// logits.  Use this to confirm that the SVTR export can be handled by
/// `tract-onnx` before attempting full OCR integration.
///
/// # Examples
///
/// ```no_run
/// use pure_onnx_ocr::run_svtr_dummy_inference;
///
/// let outputs = run_svtr_dummy_inference("models/ppocrv5/rec.onnx")
///     .expect("model should load and execute");
/// assert!(!outputs.is_empty());
/// ```
///
/// # Errors
///
/// Returns [`tract_onnx::prelude::TractError`] when the model cannot be
/// loaded, optimised, or executed.
pub fn run_svtr_dummy_inference(model_path: impl AsRef<Path>) -> TractResult<TVec<Tensor>> {
    let dummy_input: Tensor =
        tract_ndarray::Array4::<f32>::from_shape_fn(SVTR_DUMMY_SHAPE, |(_, channel, row, col)| {
            // 正規化された斜めグラデーション: チャンネルごとにスケールを変えて変化を持たせる
            let spatial_size = (SVTR_DUMMY_SHAPE[2] * SVTR_DUMMY_SHAPE[3]) as f32;
            let base = (row * SVTR_DUMMY_SHAPE[3] + col) as f32 / spatial_size;
            let channel_scale = 0.1 * channel as f32;
            (base + channel_scale).sin()
        })
        .into_dyn()
        .into();
    run_dummy_inference(model_path, dummy_input, "SVTR")
}

fn run_dummy_inference(
    model_path: impl AsRef<Path>,
    dummy_input: Tensor,
    label: &str,
) -> TractResult<TVec<Tensor>> {
    let model_path = model_path.as_ref();
    log::debug!("[{}] Loading model from {:?}", label, model_path);

    let start = std::time::Instant::now();

    let mut model = onnx_model::load_paddle_onnx(model_path)?;
    log::debug!("[{}] Model loaded, elapsed: {:?}", label, start.elapsed());

    model.set_input_fact(
        0,
        InferenceFact::dt_shape(dummy_input.datum_type(), dummy_input.shape()),
    )?;
    log::debug!("[{}] Input fact set, elapsed: {:?}", label, start.elapsed());

    log::debug!(
        "[{}] Starting model conversion to typed, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_typed()?;

    log::debug!(
        "[{}] Starting decluttering, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_decluttered()?;

    log::debug!(
        "[{}] Starting optimization, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_optimized()?;

    log::debug!(
        "[{}] Making runnable, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let model = model.into_runnable()?;

    log::debug!("[{}] Total preparation time: {:?}", label, start.elapsed());

    log::debug!(
        "[{}] Running inference, elapsed: {:?}",
        label,
        start.elapsed()
    );
    let outputs = model.run(tvec!(dummy_input.into()))?;
    log::debug!(
        "[{}] Inference complete, elapsed: {:?}",
        label,
        start.elapsed()
    );

    Ok(outputs
        .into_iter()
        .map(|value| value.into_tensor())
        .collect::<TVec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    #[ignore = "dummy inference takes >60s; run with `cargo test -- --ignored`"]
    fn dbnet_dummy_inference_runs_successfully() -> TractResult<()> {
        let model_path = Path::new("models/ppocrv5/det.onnx");
        assert!(
            model_path.exists(),
            "expected DBNet model at {:?} to exist",
            model_path
        );

        println!("Starting inference test");
        let outputs = run_dbnet_dummy_inference(model_path)?;
        println!("Test completed with {} outputs", outputs.len());

        assert!(
            !outputs.is_empty(),
            "inference should return at least one output tensor"
        );

        // 出力のシェイプを表示
        for (i, tensor) in outputs.iter().enumerate() {
            println!("Output tensor #{} shape: {:?}", i, tensor.shape());
        }

        Ok(())
    }

    #[test]
    #[ignore = "dummy inference takes >60s; run with `cargo test -- --ignored`"]
    fn svtr_dummy_inference_runs_successfully() -> TractResult<()> {
        let model_path = Path::new("models/ppocrv5/rec.onnx");
        assert!(
            model_path.exists(),
            "expected SVTR model at {:?} to exist",
            model_path
        );

        println!("Starting SVTR inference test");
        let outputs = run_svtr_dummy_inference(model_path)?;
        println!("SVTR test completed with {} outputs", outputs.len());

        assert!(
            !outputs.is_empty(),
            "SVTR inference should return at least one output tensor"
        );

        let first = &outputs[0];
        println!("SVTR output tensor shape: {:?}", first.shape());

        let shape = first.shape().to_vec();
        assert_eq!(
            shape.first().copied(),
            Some(1),
            "SVTR batch dimension should be 1"
        );
        assert!(
            shape.iter().skip(1).all(|dim| *dim > 0),
            "SVTR tensor dimensions after batch should be positive"
        );

        let view = first.to_plain_array_view::<f32>()?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for value in view.iter() {
            min = min.min(*value);
            max = max.max(*value);
        }
        println!("SVTR output value range: min={:.6}, max={:.6}", min, max);

        assert!(
            min.is_finite() && max.is_finite(),
            "SVTR output values should be finite numbers"
        );
        assert!(
            max > min,
            "SVTR output values should have a non-zero dynamic range"
        );

        Ok(())
    }
}
