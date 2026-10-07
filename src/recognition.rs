//! Recognition inference and dictionary-backed CTC postprocessing.
//!
//! Input tensors use `[batch, channel, height, width]`; outputs use
//! `[batch, time, class]`. Valid crop widths determine how much of each
//! output sequence to decode, excluding the right-hand padding.

use crate::ctc::{
    CtcGreedyDecoder, CtcGreedyDecoderConfig, CtcGreedyDecoderError, DecodedSequence,
};
use crate::dictionary::RecDictionary;
use crate::onnx_model::SharedPlanCache;
use crate::preprocessing::PreprocessedRecBatch;

/// Default number of compiled recognition plans (one per batch size and width) kept in memory.
pub const DEFAULT_REC_PLAN_CACHE: usize = 16;
use ndarray::Array3;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::anyhow;

/// Result of running SVTR recognition inference.
#[derive(Debug, Clone)]
pub struct RecInferenceOutput {
    /// Scores in `[batch, time, class]` order; may already be probabilities.
    pub logits: Array3<f32>,
    /// Number of leading time steps to decode for each batch entry.
    pub valid_timesteps: Vec<usize>,
}

/// Runnable inference session for SVTR recognition model.
#[derive(Debug)]
pub struct RecInferenceSession {
    base_model: InferenceModel,
    input_height: u32,
    cache: SharedPlanCache<(usize, u32)>,
    executor: crate::threading::Executor,
}

impl RecInferenceSession {
    /// Loads a recognition model expecting 48-pixel high inputs (PP-OCRv3 and later).
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        Self::load_with_input_height(model_path, 48)
    }

    /// Loads a recognition model with an explicit input height
    /// (PaddleOCR `RecResizeImg.image_shape[1]`).
    pub fn load_with_input_height(
        model_path: impl AsRef<Path>,
        input_height: u32,
    ) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        if input_height == 0 {
            return Err(anyhow!("recognition input height must be positive"));
        }
        log::info!("[RecInfer] Loading recognition model from {:?}", model_path);
        Self::from_model(
            crate::onnx_model::load_paddle_onnx(model_path)?,
            input_height,
        )
    }

    /// Loads a recognition model from ONNX bytes held in memory.
    pub fn from_bytes_with_input_height(
        model_bytes: &[u8],
        input_height: u32,
    ) -> TractResult<Self> {
        if input_height == 0 {
            return Err(anyhow!("recognition input height must be positive"));
        }
        log::info!(
            "[RecInfer] Loading recognition model from memory ({} bytes)",
            model_bytes.len()
        );
        Self::from_model(
            crate::onnx_model::load_paddle_onnx_from_bytes(model_bytes)?,
            input_height,
        )
    }

    fn from_model(mut inference_model: InferenceModel, input_height: u32) -> TractResult<Self> {
        let batch = inference_model.symbols.sym("batch");
        let width = inference_model.symbols.sym("width");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    batch.into(),
                    TDim::from(3),
                    TDim::from(input_height as i64),
                    width.into()
                ],
            ),
        )?;

        log::debug!("[RecInfer] Recognition model prepared");
        Ok(Self {
            base_model: inference_model,
            input_height,
            cache: SharedPlanCache::new(DEFAULT_REC_PLAN_CACHE),
            executor: crate::threading::Executor::SingleThread,
        })
    }

    /// Returns the input height this session was prepared for.
    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    /// Runs a batch and estimates valid output lengths from its crop widths.
    ///
    /// The tensor must have three channels and the configured input height.
    /// Compilation, inference and incompatible output-shape errors propagate.
    pub fn run(&self, batch: &PreprocessedRecBatch) -> TractResult<RecInferenceOutput> {
        self.run_on(batch, &self.executor)
    }

    /// Runs one batch single-threaded. Used when several batches already run
    /// in parallel: nesting tract's parallel matrix multiplication inside
    /// another rayon job lets a worker steal a second batch while the first
    /// holds tract's thread-local scratch space, which panics with
    /// "RefCell already borrowed".
    pub(crate) fn run_single_threaded(
        &self,
        batch: &PreprocessedRecBatch,
    ) -> TractResult<RecInferenceOutput> {
        self.run_on(batch, &crate::threading::Executor::SingleThread)
    }

    fn run_on(
        &self,
        batch: &PreprocessedRecBatch,
        executor: &crate::threading::Executor,
    ) -> TractResult<RecInferenceOutput> {
        let tensor_shape = batch.tensor.shape();
        if tensor_shape.len() != 4 {
            return Err(anyhow!(
                "expected recognition input tensor to have 4 dimensions, got {:?}",
                tensor_shape
            ));
        }

        let batch_size = tensor_shape[0];
        let channel = tensor_shape[1];
        let height = tensor_shape[2];
        let width = tensor_shape[3];

        log::debug!(
            "[RecInfer] Running inference with input shape {:?}",
            tensor_shape
        );

        if channel != 3 || height != self.input_height as usize {
            return Err(anyhow!(
                "expected recognition input to have shape [*, 3, {}, *], got {:?}",
                self.input_height,
                tensor_shape
            ));
        }

        let plan = self.runnable_for_dims(batch_size, width as u32)?;
        let run_start = crate::time::Instant::now();
        let outputs =
            crate::threading::run_with(executor, || plan.run(tvec!(batch.tensor.clone().into())))?;
        log::debug!(
            "[RecInfer] Ran batch {:?} in {:?}",
            tensor_shape,
            run_start.elapsed()
        );
        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("SVTR model did not return any outputs"))?;

        let view = output_tensor.to_plain_array_view::<f32>()?;
        if view.ndim() != 3 {
            return Err(anyhow!(
                "expected recognition output to have 3 dimensions, got {:?}",
                view.shape()
            ));
        }

        let logits = view.into_dimensionality::<ndarray::Ix3>()?.to_owned();
        let (logit_batch, time_steps, _classes) = logits.dim();
        if logit_batch != batch_size {
            return Err(anyhow!(
                "batch dimension mismatch between input ({}) and output ({})",
                batch_size,
                logit_batch
            ));
        }

        // Infer temporal downsampling from the actual output rather than a
        // model-specific stride. Only the unpadded prefix should reach CTC.
        let max_width = batch.max_width as f32;
        let scale = if max_width > 0.0 {
            time_steps as f32 / max_width
        } else {
            0.0
        };
        let valid_timesteps = batch
            .valid_widths
            .iter()
            .map(|width| {
                let mut steps = if scale > 0.0 {
                    (scale * *width as f32).round() as isize
                } else {
                    time_steps as isize
                };
                if steps < 1 {
                    steps = 1;
                }
                if steps as usize > time_steps {
                    steps = time_steps as isize;
                }
                steps as usize
            })
            .collect::<Vec<_>>();

        Ok(RecInferenceOutput {
            logits,
            valid_timesteps,
        })
    }

    /// Runs inference on a pool of `threads` worker threads (`1` runs
    /// single-threaded). Has no effect without the `multithread` feature or
    /// on WebAssembly builds without the `atomics` target feature.
    pub fn set_inference_threads(&mut self, threads: usize) {
        self.executor = crate::threading::executor_for(threads);
    }

    pub(crate) fn set_executor(&mut self, executor: crate::threading::Executor) {
        self.executor = executor;
    }

    /// Sets how many compiled plans (one per input shape) are kept in memory.
    /// The least recently used plan is dropped when the limit is exceeded.
    pub fn set_plan_cache_capacity(&self, capacity: usize) {
        self.cache.set_capacity(capacity);
    }

    /// Compiles and caches the plan for batches of `batch_size` crops
    /// padded to `width`.
    pub(crate) fn prepare_plan(&self, batch_size: usize, width: u32) -> TractResult<()> {
        self.runnable_for_dims(batch_size, width).map(drop)
    }

    /// Returns the number of compiled plans currently cached.
    pub fn cached_plan_count(&self) -> usize {
        self.cache.len()
    }

    fn runnable_for_dims(
        &self,
        batch_size: usize,
        width: u32,
    ) -> TractResult<Arc<TypedRunnableModel>> {
        self.cache
            .get_or_compile((batch_size, width), || self.compile(batch_size, width))
    }

    fn compile(&self, batch_size: usize, width: u32) -> TractResult<Arc<TypedRunnableModel>> {
        log::debug!(
            "[RecInfer] Preparing runnable model for batch {} width {}",
            batch_size,
            width
        );

        // Height is fixed per session, so batch size and width identify a plan.
        let mut model = self.base_model.clone();
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    TDim::from(batch_size as i64),
                    TDim::from(3),
                    TDim::from(self.input_height as i64),
                    TDim::from(width as i64)
                ],
            ),
        )?;

        let compile_start = crate::time::Instant::now();
        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;
        log::debug!("[RecInfer] Compiled plan in {:?}", compile_start.elapsed());
        Ok(plan)
    }
}

/// Configuration for recognition post processing (CTC decoding stage).
#[derive(Debug, Clone)]
pub struct RecPostProcessorConfig {
    /// CTC blank class; normally the dictionary's index 0.
    pub blank_id: usize,
    /// Text emitted for an unknown model class (default: `[UNK]`).
    pub fallback_token: String,
}

impl Default for RecPostProcessorConfig {
    fn default() -> Self {
        Self {
            blank_id: 0,
            fallback_token: "[UNK]".to_string(),
        }
    }
}

/// Errors that can occur while decoding recognition logits into text.
#[derive(Debug)]
pub enum RecPostProcessorError {
    Decoder(CtcGreedyDecoderError),
}

impl std::fmt::Display for RecPostProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecPostProcessorError::Decoder(err) => write!(f, "ctc decoder failed: {}", err),
        }
    }
}

impl std::error::Error for RecPostProcessorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecPostProcessorError::Decoder(err) => Some(err),
        }
    }
}

impl From<CtcGreedyDecoderError> for RecPostProcessorError {
    fn from(value: CtcGreedyDecoderError) -> Self {
        RecPostProcessorError::Decoder(value)
    }
}

/// Recognition post processor that converts logits into decoded text.
#[derive(Debug, Clone)]
pub struct RecPostProcessor {
    decoder: CtcGreedyDecoder,
    dictionary: Arc<RecDictionary>,
}

impl RecPostProcessor {
    /// Shares the vocabulary and configures the decoder's unknown-class fallback.
    pub fn new(dictionary: Arc<RecDictionary>, config: RecPostProcessorConfig) -> Self {
        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: config.blank_id,
            fallback_token: Some(config.fallback_token),
        });
        Self {
            decoder,
            dictionary,
        }
    }

    /// Decodes each inference sample in order, using its valid time-step prefix.
    /// Decoder errors are preserved inside [`RecPostProcessorError`].
    pub fn process(
        &self,
        output: &RecInferenceOutput,
    ) -> Result<Vec<DecodedSequence>, RecPostProcessorError> {
        self.decoder
            .decode(&output.logits, &output.valid_timesteps, &self.dictionary)
            .map_err(RecPostProcessorError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::RecDictionary;
    use crate::preprocessing::{RecPreProcessor, RecPreProcessorConfig, RecTextRegion};
    use image::{DynamicImage, ImageBuffer, Rgb};
    use ndarray::Array3;
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut buffer = ImageBuffer::new(width, height);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let base = ((x + y) % 256) as u8;
            let green = base.saturating_add(16);
            let blue = base.saturating_add(32);
            *pixel = Rgb([base, green, blue]);
        }
        DynamicImage::ImageRgb8(buffer)
    }

    fn dictionary_from_tokens(tokens: &[&str]) -> RecDictionary {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rec_post_dict_{}.txt", timestamp));
        fs::write(&path, tokens.join("\n")).unwrap();
        let dict = RecDictionary::from_path(&path).unwrap();
        fs::remove_file(path).ok();
        dict
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
    fn recognition_inference_runs() -> TractResult<()> {
        let model_path =
            locate_ppocrv5_asset("rec.onnx").expect("expected SVTR model under models/ppocrv5/");

        let session = RecInferenceSession::load(model_path)?;

        let image = gradient_image(320, 160);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let regions = vec![RecTextRegion {
            x: 10,
            y: 20,
            width: 120,
            height: 60,
        }];
        let batch = preprocessor
            .process(&image, &regions)
            .expect("recognition preprocessing should succeed");

        let output = session.run(&batch)?;
        let shape = output.logits.dim();

        assert_eq!(shape.0, 1);
        assert!(shape.1 > 0);
        assert!(shape.2 > 0);
        assert_eq!(output.valid_timesteps.len(), 1);
        assert!(output.valid_timesteps[0] <= shape.1);

        Ok(())
    }

    #[test]
    fn post_processor_decodes_with_fallback() {
        let logits = Array3::from_shape_vec(
            (2, 4, 4),
            vec![
                5.0, 0.1, -1.0, -2.0, //
                -2.0, 4.5, 0.0, -3.0, //
                -3.0, 4.2, -0.5, -3.5, //
                -4.0, -1.0, 4.8, -3.0, //
                // second sequence with unknown indices
                -6.0, -5.0, 1.0, 4.5, //
                5.0, 0.0, -1.0, -2.0, //
                5.0, 0.0, -1.0, -2.0, //
                5.0, 0.0, -1.0, -2.0, //
            ],
        )
        .unwrap();
        let output = RecInferenceOutput {
            logits,
            valid_timesteps: vec![4, 1],
        };

        let dictionary = Arc::new(dictionary_from_tokens(&["a", "b"]));
        let processor = RecPostProcessor::new(
            Arc::clone(&dictionary),
            RecPostProcessorConfig {
                blank_id: 0,
                fallback_token: "[UNK]".to_string(),
            },
        );

        let sequences = processor.process(&output).expect("decoding succeeds");
        assert_eq!(sequences.len(), 2);

        assert_eq!(sequences[0].text, "ab");
        assert_eq!(sequences[0].fallback_count, 0);

        assert_eq!(sequences[1].text, "[UNK]");
        assert_eq!(sequences[1].fallback_count, 1);
    }
}
