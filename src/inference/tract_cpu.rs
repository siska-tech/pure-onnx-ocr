//! CPU inference backend using tract-onnx.

use crate::engine::OcrError;
use crate::inference::{DetInference, DetInferenceOutput, RecInference};
use crate::preprocessing::{PreprocessedDetInput, PreprocessedRecBatch};
use crate::recognition::RecInferenceOutput;
use ndarray::Axis;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::anyhow::anyhow;

/// CPU-based detection inference session using tract-onnx.
#[derive(Debug)]
pub struct TractDetSession {
    base_model: InferenceModel,
    cache: RwLock<HashMap<(u32, u32), Arc<TypedRunnableModel<TypedModel>>>>,
}

impl TractDetSession {
    /// Loads a detection model from the given path.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, OcrError> {
        let model_path = model_path.as_ref();
        println!("[DetInfer] Loading detection model from {:?}", model_path);

        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(model_path)
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: model_path.to_path_buf(),
            })?;

        let height = inference_model.symbol_table.sym("height");
        let width = inference_model.symbol_table.sym("width");
        inference_model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![TDim::from(1), TDim::from(3), height.into(), width.into()],
                ),
            )
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: model_path.to_path_buf(),
            })?;

        println!("[DetInfer] Detection model prepared");
        Ok(Self {
            base_model: inference_model,
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn runnable_for_dims(
        &self,
        width: u32,
        height: u32,
    ) -> TractResult<Arc<TypedRunnableModel<TypedModel>>> {
        if let Ok(cache) = self.cache.read() {
            if let Some(plan) = cache.get(&(width, height)) {
                return Ok(Arc::clone(plan));
            }
        }

        println!(
            "[DetInfer] Preparing runnable model for dims ({}, {})",
            width, height
        );

        let mut model = self.base_model.clone();
        model
            .set_input_fact(
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
            )
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: std::path::PathBuf::new(), // Path not available in cache lookup
            })?;

        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        let plan = Arc::new(plan);
        if let Ok(mut cache) = self.cache.write() {
            cache.insert((width, height), Arc::clone(&plan));
        }

        Ok(plan)
    }
}

impl DetInference for TractDetSession {
    fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError> {
        println!(
            "[DetInfer] Running inference with input dims {:?}",
            input.tensor.shape()
        );

        let (width, height) = input.resized_dims;
        let plan = self
            .runnable_for_dims(width, height)
            .map_err(|source| OcrError::DetectionInference { source })?;

        let outputs = plan
            .run(tvec!(input.tensor.clone().into()))
            .map_err(|source| OcrError::DetectionInference { source })?;
        let output_tensor =
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| OcrError::DetectionInference {
                    source: anyhow!("DBNet model did not return any outputs").into(),
                })?;

        let view = output_tensor.to_array_view::<f32>().map_err(|source| {
            OcrError::DetectionInference {
                source: anyhow!("Failed to convert output tensor: {}", source).into(),
            }
        })?;
        let view = view
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|source| OcrError::DetectionInference {
                source: anyhow!("Failed to reshape output tensor: {}", source).into(),
            })?;
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
}

/// CPU-based recognition inference session using tract-onnx.
#[derive(Debug)]
pub struct TractRecSession {
    base_model: InferenceModel,
    cache: RwLock<HashMap<(usize, u32), Arc<TypedRunnableModel<TypedModel>>>>,
}

impl TractRecSession {
    /// Loads a recognition model from the given path.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, OcrError> {
        let model_path = model_path.as_ref();
        println!("[RecInfer] Loading recognition model from {:?}", model_path);

        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(model_path)
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: model_path.to_path_buf(),
            })?;

        let batch = inference_model.symbol_table.sym("batch");
        let width = inference_model.symbol_table.sym("width");
        inference_model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![batch.into(), TDim::from(3), TDim::from(48), width.into()],
                ),
            )
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: model_path.to_path_buf(),
            })?;

        println!("[RecInfer] Recognition model prepared");
        Ok(Self {
            base_model: inference_model,
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn runnable_for_dims(
        &self,
        batch_size: usize,
        width: u32,
    ) -> TractResult<Arc<TypedRunnableModel<TypedModel>>> {
        if let Ok(cache) = self.cache.read() {
            if let Some(plan) = cache.get(&(batch_size, width)) {
                return Ok(Arc::clone(plan));
            }
        }

        println!(
            "[RecInfer] Preparing runnable model for batch {} width {}",
            batch_size, width
        );

        let mut model = self.base_model.clone();
        model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        TDim::from(batch_size as i64),
                        TDim::from(3),
                        TDim::from(48),
                        TDim::from(width as i64)
                    ],
                ),
            )
            .map_err(|source| OcrError::ModelLoad {
                source,
                path: std::path::PathBuf::new(), // Path not available in cache lookup
            })?;

        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;

        let plan = Arc::new(plan);
        if let Ok(mut cache) = self.cache.write() {
            cache.insert((batch_size, width), Arc::clone(&plan));
        }

        Ok(plan)
    }
}

impl RecInference for TractRecSession {
    fn run(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput, OcrError> {
        let tensor_shape = batch.tensor.shape();
        if tensor_shape.len() != 4 {
            return Err(OcrError::RecognitionInference {
                source: anyhow!(
                    "expected recognition input tensor to have 4 dimensions, got {:?}",
                    tensor_shape
                )
                .into(),
            });
        }

        let batch_size = tensor_shape[0];
        let channel = tensor_shape[1];
        let height = tensor_shape[2];
        let width = tensor_shape[3];

        println!(
            "[RecInfer] Running inference with input shape {:?}",
            tensor_shape
        );

        if channel != 3 || height != 48 {
            return Err(OcrError::RecognitionInference {
                source: anyhow!(
                    "expected recognition input to have shape [*, 3, 48, *], got {:?}",
                    tensor_shape
                )
                .into(),
            });
        }

        let plan = self
            .runnable_for_dims(batch_size, width as u32)
            .map_err(|source| OcrError::RecognitionInference { source })?;
        let outputs = plan
            .run(tvec!(batch.tensor.clone().into()))
            .map_err(|source| OcrError::RecognitionInference { source })?;
        let output_tensor =
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| OcrError::RecognitionInference {
                    source: anyhow!("SVTR model did not return any outputs").into(),
                })?;

        let view = output_tensor.to_array_view::<f32>().map_err(|source| {
            OcrError::RecognitionInference {
                source: anyhow!("Failed to convert output tensor: {}", source).into(),
            }
        })?;
        if view.ndim() != 3 {
            return Err(OcrError::RecognitionInference {
                source: anyhow!(
                    "expected recognition output to have 3 dimensions, got {:?}",
                    view.shape()
                )
                .into(),
            });
        }

        let logits = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|source| OcrError::RecognitionInference {
                source: anyhow!("Failed to reshape output tensor: {}", source).into(),
            })?
            .to_owned();
        let (logit_batch, time_steps, _classes) = logits.dim();
        if logit_batch != batch_size {
            return Err(OcrError::RecognitionInference {
                source: anyhow!(
                    "batch dimension mismatch between input ({}) and output ({})",
                    batch_size,
                    logit_batch
                )
                .into(),
            });
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
}
