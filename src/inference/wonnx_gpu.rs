//! GPU inference backend using wonnx.
//!
//! This module provides GPU-accelerated inference using wonnx (WebGPU-based ONNX runtime).

#[cfg(feature = "gpu")]
use crate::engine::OcrError;
#[cfg(feature = "gpu")]
use crate::inference::{DetInference, DetInferenceOutput, RecInference};
#[cfg(feature = "gpu")]
use crate::preprocessing::{PreprocessedDetInput, PreprocessedRecBatch};
#[cfg(feature = "gpu")]
use crate::recognition::RecInferenceOutput;
#[cfg(feature = "gpu")]
use ndarray::{Array3, Axis};
#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use std::path::Path;
#[cfg(feature = "gpu")]
use tract_onnx::prelude::Tensor;
#[cfg(feature = "gpu")]
use wonnx::utils::{InputTensor, OutputTensor};

/// GPU-based detection inference session using wonnx.
#[cfg(feature = "gpu")]
pub struct WonnxDetSession {
    session: wonnx::Session,
    input_name: String,
    output_name: String,
}

#[cfg(feature = "gpu")]
impl WonnxDetSession {
    /// Loads a detection model from the given path.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, OcrError> {
        let model_path = model_path.as_ref();
        println!("[WonnxDet] Loading detection model from {:?}", model_path);

        let session = pollster::block_on(async {
            wonnx::Session::from_path(model_path)
                .await
                .map_err(|e| OcrError::GpuInit(format!("Failed to load wonnx session: {}", e)))
        })?;

        // Get input and output names from the model
        // For ONNX models, inputs/outputs are typically named, but we'll use defaults
        // Common names: "input", "x", "images" for input, "output", "output0" for output
        let input_name = "x".to_string(); // Default input name, may need adjustment
        let output_name = "output".to_string(); // Default output name, may need adjustment

        println!("[WonnxDet] Detection model loaded successfully");
        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    async fn run_async(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError> {
        // Convert tract Tensor to wonnx input format
        let input_tensor = tensor_to_wonnx_input(&input.tensor)?;
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), input_tensor);

        // Run inference
        let outputs = self
            .session
            .run(&inputs)
            .await
            .map_err(|e| OcrError::GpuInference(format!("wonnx inference failed: {}", e)))?;

        // Get the output
        let output_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| OcrError::GpuInference(format!("Output '{}' not found", self.output_name)))?;

        // Extract shape from output data
        let data_len = match output_tensor {
            OutputTensor::F32(data) => data.len(),
            _ => return Err(OcrError::GpuInference("Unsupported output tensor type".to_string())),
        };

        // For detection, output is typically [1, 1, H, W]
        // We can infer H and W from the input dimensions
        let (width, height) = input.resized_dims;
        let expected_shape = vec![1, 1, height as usize, width as usize];
        let expected_len: usize = expected_shape.iter().product();

        // If data length doesn't match, try to infer from actual data
        let actual_shape = if data_len == expected_len {
            expected_shape
        } else {
            // Try to infer shape: assume it's [1, 1, H, W] where H*W = data_len
            // This is a heuristic and may not work for all models
            let hw = data_len;
            // Try to find reasonable H and W that multiply to hw
            let h = height as usize;
            let w = (hw / h).max(1);
            vec![1, 1, h, w]
        };

        // Convert wonnx output to ndarray
        let output_array = wonnx_output_to_ndarray_4d(output_tensor, &actual_shape)?;

        // Extract probability map: [1, 1, H, W] -> [H, W]
        let probability_map = output_array
            .index_axis(Axis(0), 0)
            .index_axis(Axis(0), 0)
            .to_owned();

        println!(
            "[WonnxDet] Inference complete, output dims {:?}",
            probability_map.raw_dim()
        );

        Ok(DetInferenceOutput { probability_map })
    }
}

#[cfg(feature = "gpu")]
impl DetInference for WonnxDetSession {
    fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError> {
        println!(
            "[WonnxDet] Running inference with input dims {:?}",
            input.tensor.shape()
        );
        pollster::block_on(self.run_async(input))
    }
}

/// GPU-based recognition inference session using wonnx.
#[cfg(feature = "gpu")]
pub struct WonnxRecSession {
    session: wonnx::Session,
    input_name: String,
    output_name: String,
}

#[cfg(feature = "gpu")]
impl WonnxRecSession {
    /// Loads a recognition model from the given path.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, OcrError> {
        let model_path = model_path.as_ref();
        println!("[WonnxRec] Loading recognition model from {:?}", model_path);

        let session = pollster::block_on(async {
            wonnx::Session::from_path(model_path)
                .await
                .map_err(|e| OcrError::GpuInit(format!("Failed to load wonnx session: {}", e)))
        })?;

        // Get input and output names from the model
        let input_name = "x".to_string(); // Default input name
        let output_name = "output".to_string(); // Default output name

        println!("[WonnxRec] Recognition model loaded successfully");
        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    async fn run_async(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput, OcrError> {
        // Convert tract Tensor to wonnx input format
        let input_tensor = tensor_to_wonnx_input(&batch.tensor)?;
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), input_tensor);

        // Run inference
        let outputs = self
            .session
            .run(&inputs)
            .await
            .map_err(|e| OcrError::GpuInference(format!("wonnx inference failed: {}", e)))?;

        // Get the output
        let output_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| OcrError::GpuInference(format!("Output '{}' not found", self.output_name)))?;

        // Extract shape from output data
        let tensor_shape = batch.tensor.shape();
        let batch_size = tensor_shape[0];

        let data_len = match output_tensor {
            OutputTensor::F32(data) => data.len(),
            _ => return Err(OcrError::GpuInference("Unsupported output tensor type".to_string())),
        };

        // Infer shape: [batch, time_steps, classes]
        // We know batch_size, so we can infer time_steps and classes
        // This is a heuristic - ideally we'd get this from metadata
        let remaining = data_len / batch_size;
        // Try to infer reasonable time_steps and classes
        // For SVTR, time_steps is typically related to input width
        let max_width = batch.max_width as usize;
        let estimated_timesteps = (max_width * 2).max(1); // Heuristic
        let classes = remaining / estimated_timesteps.max(1);
        let timesteps = if classes > 0 { remaining / classes } else { remaining };

        let actual_shape = vec![batch_size, timesteps, classes.max(1)];

        // Convert wonnx output to ndarray
        let logits = wonnx_output_to_ndarray_3d(output_tensor, &actual_shape)?;

        let tensor_shape = batch.tensor.shape();
        let batch_size = tensor_shape[0];
        let (logit_batch, time_steps, _classes) = logits.dim();

        if logit_batch != batch_size {
            return Err(OcrError::GpuInference(format!(
                "batch dimension mismatch between input ({}) and output ({})",
                batch_size, logit_batch
            )));
        }

        // Calculate valid timesteps (same logic as TractRecSession)
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

        println!(
            "[WonnxRec] Inference complete, output shape {:?}, valid_timesteps {:?}",
            logits.dim(),
            valid_timesteps
        );

        Ok(RecInferenceOutput {
            logits,
            valid_timesteps,
        })
    }
}

#[cfg(feature = "gpu")]
impl RecInference for WonnxRecSession {
    fn run(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput, OcrError> {
        let tensor_shape = batch.tensor.shape();
        println!(
            "[WonnxRec] Running inference with input shape {:?}",
            tensor_shape
        );
        pollster::block_on(self.run_async(batch))
    }
}

/// Converts a tract Tensor to wonnx InputTensor format.
#[cfg(feature = "gpu")]
fn tensor_to_wonnx_input(tensor: &Tensor) -> Result<InputTensor, OcrError> {
    let array_view = tensor
        .to_array_view::<f32>()
        .map_err(|e| OcrError::GpuInference(format!("Failed to convert tensor to array: {}", e)))?;

    // Extract data as Vec<f32>
    let data: Vec<f32> = array_view.iter().cloned().collect();

    Ok(InputTensor::F32(data.into()))
}

/// Converts a wonnx OutputTensor to ndarray Array4<f32>.
#[cfg(feature = "gpu")]
fn wonnx_output_to_ndarray_4d(
    output: &OutputTensor,
    shape: &[usize],
) -> Result<ndarray::Array4<f32>, OcrError> {
    match output {
        OutputTensor::F32(data) => {
            if shape.len() != 4 {
                return Err(OcrError::GpuInference(format!(
                    "Expected 4D shape, got {:?}",
                    shape
                )));
            }

            let total_elements: usize = shape.iter().product();
            if data.len() != total_elements {
                return Err(OcrError::GpuInference(format!(
                    "Shape {:?} requires {} elements, but got {}",
                    shape,
                    total_elements,
                    data.len()
                )));
            }

            ndarray::Array4::from_shape_vec(
                (shape[0], shape[1], shape[2], shape[3]),
                data.clone(),
            )
            .map_err(|e| OcrError::GpuInference(format!("Failed to create Array4: {}", e)))
        }
        _ => Err(OcrError::GpuInference(
            "Unsupported output tensor type (expected F32)".to_string(),
        )),
    }
}

/// Converts a wonnx OutputTensor to ndarray Array3<f32>.
#[cfg(feature = "gpu")]
fn wonnx_output_to_ndarray_3d(
    output: &OutputTensor,
    shape: &[usize],
) -> Result<Array3<f32>, OcrError> {
    match output {
        OutputTensor::F32(data) => {
            if shape.len() != 3 {
                return Err(OcrError::GpuInference(format!(
                    "Expected 3D shape, got {:?}",
                    shape
                )));
            }

            let total_elements: usize = shape.iter().product();
            if data.len() != total_elements {
                return Err(OcrError::GpuInference(format!(
                    "Shape {:?} requires {} elements, but got {}",
                    shape,
                    total_elements,
                    data.len()
                )));
            }

            Array3::from_shape_vec((shape[0], shape[1], shape[2]), data.clone())
                .map_err(|e| OcrError::GpuInference(format!("Failed to create Array3: {}", e)))
        }
        _ => Err(OcrError::GpuInference(
            "Unsupported output tensor type (expected F32)".to_string(),
        )),
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::preprocessing::{DetPreProcessor, DetPreProcessorConfig, RecPreProcessor, RecPreProcessorConfig, RecTextRegion};
    use image::{DynamicImage, ImageBuffer, Rgb};
    use std::env;
    use std::path::{Path, PathBuf};

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

    fn dummy_image(width: u32, height: u32) -> DynamicImage {
        let pixel = Rgb([128, 64, 200]);
        let buffer = ImageBuffer::from_pixel(width, height, pixel);
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn test_wonnx_det_session_load() {
        let model_path = locate_ppocrv5_asset("det.onnx");
        if model_path.is_none() {
            println!("Skipping test: det.onnx not found");
            return;
        }

        let session = WonnxDetSession::load(model_path.unwrap());
        assert!(session.is_ok(), "WonnxDetSession should load successfully");
    }

    #[test]
    fn test_wonnx_rec_session_load() {
        let model_path = locate_ppocrv5_asset("rec.onnx");
        if model_path.is_none() {
            println!("Skipping test: rec.onnx not found");
            return;
        }

        let session = WonnxRecSession::load(model_path.unwrap());
        assert!(session.is_ok(), "WonnxRecSession should load successfully");
    }

    #[test]
    fn test_wonnx_det_session_inference() {
        let model_path = locate_ppocrv5_asset("det.onnx");
        if model_path.is_none() {
            println!("Skipping test: det.onnx not found");
            return;
        }

        let session = WonnxDetSession::load(model_path.unwrap())
            .expect("WonnxDetSession should load successfully");

        let image = dummy_image(320, 320);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());
        let preprocessed = preprocessor.process(&image)
            .expect("Preprocessing should succeed");

        let output = session.run(&preprocessed);
        assert!(output.is_ok(), "WonnxDetSession inference should succeed");
        let output = output.unwrap();
        assert!(!output.probability_map.is_empty(), "Output should not be empty");
    }

    #[test]
    fn test_wonnx_rec_session_inference() {
        let model_path = locate_ppocrv5_asset("rec.onnx");
        if model_path.is_none() {
            println!("Skipping test: rec.onnx not found");
            return;
        }

        let session = WonnxRecSession::load(model_path.unwrap())
            .expect("WonnxRecSession should load successfully");

        let image = dummy_image(200, 100);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let regions = vec![RecTextRegion {
            x: 10,
            y: 20,
            width: 120,
            height: 60,
        }];
        let batch = preprocessor.process(&image, &regions)
            .expect("Preprocessing should succeed");

        let output = session.run(&batch);
        assert!(output.is_ok(), "WonnxRecSession inference should succeed");
        let output = output.unwrap();
        assert!(!output.logits.is_empty(), "Output should not be empty");
        assert_eq!(output.valid_timesteps.len(), 1);
    }

    #[test]
    fn test_gpu_cpu_output_equivalence() {
        // This test compares CPU and GPU outputs to ensure they are equivalent
        // Note: This test requires both CPU and GPU backends to be available
        let det_model_path = locate_ppocrv5_asset("det.onnx");
        let rec_model_path = locate_ppocrv5_asset("rec.onnx");
        
        if det_model_path.is_none() || rec_model_path.is_none() {
            println!("Skipping test: model files not found");
            return;
        }

        // Skip if GPU is not available
        if !crate::engine::OcrEngineBuilder::is_gpu_available_static() {
            println!("Skipping test: GPU not available");
            return;
        }

        use crate::inference::TractDetSession;
        use crate::preprocessing::{DetPreProcessor, DetPreProcessorConfig};

        let image = dummy_image(320, 320);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());
        let preprocessed = preprocessor.process(&image)
            .expect("Preprocessing should succeed");

        // Run CPU inference
        let cpu_session = TractDetSession::load(det_model_path.as_ref().unwrap())
            .expect("CPU session should load");
        let cpu_output = cpu_session.run(&preprocessed)
            .expect("CPU inference should succeed");

        // Run GPU inference
        let gpu_session = WonnxDetSession::load(det_model_path.unwrap())
            .expect("GPU session should load");
        let gpu_output = gpu_session.run(&preprocessed)
            .expect("GPU inference should succeed");

        // Compare outputs (allow for small floating point differences)
        let cpu_map = &cpu_output.probability_map;
        let gpu_map = &gpu_output.probability_map;

        assert_eq!(
            cpu_map.shape(),
            gpu_map.shape(),
            "CPU and GPU outputs should have the same shape"
        );

        // Check that values are approximately equal (within tolerance)
        let tolerance = 1e-3;
        let max_diff = cpu_map
            .iter()
            .zip(gpu_map.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < tolerance,
            "CPU and GPU outputs should be approximately equal (max diff: {})",
            max_diff
        );
    }
}
