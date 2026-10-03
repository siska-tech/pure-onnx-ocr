use crate::ctc::{
    CtcGreedyDecoder, CtcGreedyDecoderConfig, CtcGreedyDecoderError, DecodedSequence,
};
use crate::dictionary::RecDictionary;
use crate::preprocessing::PreprocessedRecBatch;
use ndarray::Array3;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::anyhow;

/// Result of running SVTR recognition inference.
#[derive(Debug, Clone)]
pub struct RecInferenceOutput {
    pub logits: Array3<f32>,
    pub valid_timesteps: Vec<usize>,
}

/// Runnable inference session for SVTR recognition model.
#[derive(Debug)]
pub struct RecInferenceSession {
    base_model: InferenceModel,
    input_height: u32,
    cache: RefCell<HashMap<(usize, u32), Arc<TypedRunnableModel>>>,
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
        println!("[RecInfer] Loading recognition model from {:?}", model_path);

        let mut inference_model = crate::onnx_model::load_paddle_onnx(model_path)?;

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

        println!("[RecInfer] Recognition model prepared");
        Ok(Self {
            base_model: inference_model,
            input_height,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Returns the input height this session was prepared for.
    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    pub fn run(&self, batch: &PreprocessedRecBatch) -> TractResult<RecInferenceOutput> {
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

        println!(
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
        let outputs = plan.run(tvec!(batch.tensor.clone().into()))?;
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

    fn runnable_for_dims(
        &self,
        batch_size: usize,
        width: u32,
    ) -> TractResult<Arc<TypedRunnableModel>> {
        if let Some(plan) = self.cache.borrow().get(&(batch_size, width)) {
            return Ok(Arc::clone(plan));
        }

        println!(
            "[RecInfer] Preparing runnable model for batch {} width {}",
            batch_size, width
        );

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

        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        self.cache
            .borrow_mut()
            .insert((batch_size, width), Arc::clone(&plan));

        Ok(plan)
    }
}

/// Configuration for recognition post processing (CTC decoding stage).
#[derive(Debug, Clone)]
pub struct RecPostProcessorConfig {
    pub blank_id: usize,
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
