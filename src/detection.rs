use std::path::Path;
use std::sync::Arc;

use crate::onnx_model::PlanCache;
use crate::preprocessing::PreprocessedDetInput;

/// Default number of compiled detection plans (one per input size) kept in memory.
pub const DEFAULT_DET_PLAN_CACHE: usize = 4;
use ndarray::{Array2, Axis};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::anyhow;

/// Result of running DBNet detection inference.
#[derive(Debug, Clone)]
pub struct DetInferenceOutput {
    pub probability_map: Array2<f32>,
}

/// Runnable inference session for DBNet detection model.
#[derive(Debug)]
pub struct DetInferenceSession {
    base_model: InferenceModel,
    cache: std::sync::Mutex<PlanCache<(u32, u32)>>,
    executor: crate::threading::Executor,
}

impl DetInferenceSession {
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        log::info!("[DetInfer] Loading detection model from {:?}", model_path);

        Self::from_model(crate::onnx_model::load_paddle_onnx(model_path)?)
    }

    /// Loads a DBNet detection model from ONNX bytes held in memory.
    pub fn from_bytes(model_bytes: &[u8]) -> TractResult<Self> {
        log::info!(
            "[DetInfer] Loading detection model from memory ({} bytes)",
            model_bytes.len()
        );
        Self::from_model(crate::onnx_model::load_paddle_onnx_from_bytes(model_bytes)?)
    }

    fn from_model(mut inference_model: InferenceModel) -> TractResult<Self> {
        let height = inference_model.symbols.sym("height");
        let width = inference_model.symbols.sym("width");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![TDim::from(1), TDim::from(3), height.into(), width.into()],
            ),
        )?;

        log::debug!("[DetInfer] Detection model prepared");
        Ok(Self {
            base_model: inference_model,
            cache: std::sync::Mutex::new(PlanCache::new(DEFAULT_DET_PLAN_CACHE)),
            executor: crate::threading::Executor::SingleThread,
        })
    }

    pub fn run(&self, input: &PreprocessedDetInput) -> TractResult<DetInferenceOutput> {
        log::debug!(
            "[DetInfer] Running inference with input dims {:?}",
            input.tensor.shape()
        );

        let (width, height) = input.resized_dims;
        let plan = self.runnable_for_dims(width, height)?;

        let outputs = crate::threading::run_with(&self.executor, || {
            plan.run(tvec!(input.tensor.clone().into()))
        })?;
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("DBNet model did not return any outputs"))?;

        let view = output_tensor.to_plain_array_view::<f32>()?;
        let view = view.into_dimensionality::<ndarray::Ix4>()?;
        let probability_map = view
            .index_axis(Axis(0), 0)
            .index_axis(Axis(0), 0)
            .to_owned();

        log::debug!(
            "[DetInfer] Inference complete, output dims {:?}",
            probability_map.raw_dim()
        );

        Ok(DetInferenceOutput { probability_map })
    }

    /// Runs inference on a pool of `threads` worker threads (`1` runs
    /// single-threaded). Has no effect without the `multithread` feature or
    /// on WebAssembly.
    pub fn set_inference_threads(&mut self, threads: usize) {
        self.executor = crate::threading::executor_for(threads);
    }

    pub(crate) fn set_executor(&mut self, executor: crate::threading::Executor) {
        self.executor = executor;
    }

    /// Sets how many compiled plans (one per input shape) are kept in memory.
    /// The least recently used plan is dropped when the limit is exceeded.
    pub fn set_plan_cache_capacity(&self, capacity: usize) {
        crate::onnx_model::lock_cache(&self.cache).set_capacity(capacity);
    }

    /// Returns the number of compiled plans currently cached.
    pub fn cached_plan_count(&self) -> usize {
        crate::onnx_model::lock_cache(&self.cache).len()
    }

    fn runnable_for_dims(&self, width: u32, height: u32) -> TractResult<Arc<TypedRunnableModel>> {
        if let Some(plan) = crate::onnx_model::lock_cache(&self.cache).get((width, height)) {
            return Ok(plan);
        }

        log::debug!(
            "[DetInfer] Preparing runnable model for dims ({}, {})",
            width,
            height
        );

        let mut model = self.base_model.clone();
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    TDim::from(1),
                    TDim::from(3),
                    TDim::from(height as i64),
                    TDim::from(width as i64)
                ],
            ),
        )?;

        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        crate::onnx_model::lock_cache(&self.cache).insert((width, height), Arc::clone(&plan));

        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::{DetPreProcessor, DetPreProcessorConfig};
    use image::{ImageBuffer, Rgb};
    use std::env;
    use std::path::{Path, PathBuf};

    fn dummy_image(width: u32, height: u32) -> image::DynamicImage {
        let pixel = Rgb([128, 64, 200]);
        let buffer = ImageBuffer::from_pixel(width, height, pixel);
        image::DynamicImage::ImageRgb8(buffer)
    }

    fn locate_ppocrv5_asset(file_name: &str) -> Option<PathBuf> {
        let mut bases: Vec<PathBuf> = Vec::new();
        if let Some(dir) = env::var_os("PURE_ONNX_OCR_FIXTURE_DIR") {
            let env_path = PathBuf::from(dir);
            bases.push(env_path.clone());
            bases.push(env_path.join("models"));
        }

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        bases.push(manifest.join("tests").join("fixtures").join("models"));
        bases.push(manifest.join("tests").join("fixtures"));
        bases.push(manifest.join("models"));

        for base in bases {
            let ppocr_dir = base.join("ppocrv5");
            let candidate = ppocr_dir.join(file_name);
            if candidate.exists() {
                return Some(candidate);
            }

            let alt = base.join(file_name);
            if alt.exists() {
                return Some(alt);
            }
        }

        None
    }

    #[test]
    fn detection_inference_runs() -> TractResult<()> {
        let model_path =
            locate_ppocrv5_asset("det.onnx").expect("expected DBNet model under models/ppocrv5/");

        let session = DetInferenceSession::load(model_path)?;

        let image = dummy_image(320, 320);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 320,
            ..DetPreProcessorConfig::default()
        });
        let preprocessed = preprocessor
            .process(&image)
            .expect("preprocessing should succeed");

        let output = session.run(&preprocessed)?;
        assert_eq!(
            output.probability_map.dim(),
            (
                preprocessed.resized_dims.1 as usize,
                preprocessed.resized_dims.0 as usize
            )
        );

        Ok(())
    }
}
