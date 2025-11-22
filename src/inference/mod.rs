//! Inference backend abstraction layer.
//!
//! This module provides traits for abstracting over different inference backends
//! (CPU via tract-onnx, GPU via wonnx, etc.).

mod tract_cpu;

pub use tract_cpu::{TractDetSession, TractRecSession};

use crate::engine::OcrError;
use crate::preprocessing::{PreprocessedDetInput, PreprocessedRecBatch};
use crate::recognition::RecInferenceOutput;

/// Detection inference output.
#[derive(Debug, Clone)]
pub struct DetInferenceOutput {
    pub probability_map: ndarray::Array2<f32>,
}

/// Trait for detection inference backends.
///
/// Implementations must be `Send + Sync` to allow safe use across threads.
pub trait DetInference: Send + Sync {
    /// Runs detection inference on the preprocessed input.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails (model execution error, shape mismatch, etc.).
    fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput, OcrError>;
}

/// Trait for recognition inference backends.
///
/// Implementations must be `Send + Sync` to allow safe use across threads.
pub trait RecInference: Send + Sync {
    /// Runs recognition inference on the preprocessed batch.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails (model execution error, shape mismatch, etc.).
    fn run(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput, OcrError>;
}

#[cfg(feature = "gpu")]
mod wonnx_gpu;

#[cfg(feature = "gpu")]
pub use wonnx_gpu::{WonnxDetSession, WonnxRecSession};
