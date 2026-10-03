use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::preprocessing::PreprocessedDetInput;
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
    cache: RefCell<HashMap<(u32, u32), Arc<TypedRunnableModel>>>,
}

impl DetInferenceSession {
    pub fn load(model_path: impl AsRef<Path>) -> TractResult<Self> {
        let model_path = model_path.as_ref();
        println!("[DetInfer] Loading detection model from {:?}", model_path);

        let mut inference_model = crate::onnx_model::load_paddle_onnx(model_path)?;

        let height = inference_model.symbols.sym("height");
        let width = inference_model.symbols.sym("width");
        inference_model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![TDim::from(1), TDim::from(3), height.into(), width.into()],
            ),
        )?;

        println!("[DetInfer] Detection model prepared");
        Ok(Self {
            base_model: inference_model,
            cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn run(&self, input: &PreprocessedDetInput) -> TractResult<DetInferenceOutput> {
        println!(
            "[DetInfer] Running inference with input dims {:?}",
            input.tensor.shape()
        );

        let (width, height) = input.resized_dims;
        let plan = self.runnable_for_dims(width, height)?;

        let outputs = plan.run(tvec!(input.tensor.clone().into()))?;
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

        println!(
            "[DetInfer] Inference complete, output dims {:?}",
            probability_map.raw_dim()
        );

        Ok(DetInferenceOutput { probability_map })
    }

    fn runnable_for_dims(&self, width: u32, height: u32) -> TractResult<Arc<TypedRunnableModel>> {
        if let Some(plan) = self.cache.borrow().get(&(width, height)) {
            return Ok(Arc::clone(plan));
        }

        println!(
            "[DetInfer] Preparing runnable model for dims ({}, {})",
            width, height
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

        self.cache
            .borrow_mut()
            .insert((width, height), Arc::clone(&plan));

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
