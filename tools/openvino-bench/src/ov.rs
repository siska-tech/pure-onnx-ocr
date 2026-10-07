//! OpenVINO Runtime sessions that are drop-in replacements for
//! `DetInferenceSession::run` / `RecInferenceSession::run`: same inputs
//! (`PreprocessedDetInput` / `PreprocessedRecBatch`), same outputs
//! (`DetInferenceOutput` / `RecInferenceOutput`). The ONNX files are read
//! directly (`ov_core_read_model`), so no IR conversion is involved.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array2, Array3};
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, PropertyKey, RwPropertyKey, Shape,
    Tensor,
};
use pure_onnx_ocr::{
    DetInferenceOutput, PreprocessedDetInput, PreprocessedRecBatch, RecInferenceOutput,
};

/// CPU plugin settings applied before compiling a model.
#[derive(Debug, Clone)]
pub struct OvSettings {
    /// `PERFORMANCE_HINT`: `LATENCY` or `THROUGHPUT`.
    pub hint: String,
    /// `INFERENCE_NUM_THREADS`; 0 keeps OpenVINO's automatic choice.
    pub threads: usize,
    /// `NUM_STREAMS`; `None` keeps the value derived from the hint.
    pub streams: Option<String>,
}

pub fn new_core() -> Result<Core> {
    Core::new().map_err(|e| anyhow!("OpenVINO setup failed: {e:?}"))
}

pub fn version_string(core: &Core) -> String {
    core.versions("CPU")
        .ok()
        .and_then(|v| v.into_iter().next())
        .map(|(_, v)| format!("{} {}", v.description, v.build_number))
        .unwrap_or_default()
}

/// Reads an ONNX file and compiles it for the CPU with `settings`.
pub fn compile(core: &mut Core, onnx: &Path, settings: &OvSettings) -> Result<CompiledModel> {
    let cpu = DeviceType::CPU;
    core.set_property(&cpu, &RwPropertyKey::HintPerformanceMode, &settings.hint)?;
    core.set_property(
        &cpu,
        &RwPropertyKey::InferenceNumThreads,
        &settings.threads.to_string(),
    )?;
    if let Some(streams) = &settings.streams {
        core.set_property(&cpu, &RwPropertyKey::NumStreams, streams)?;
    }
    let path = onnx.to_str().context("non UTF-8 model path")?;
    let model = core
        .read_model_from_file(path, "")
        .with_context(|| format!("reading {path}"))?;
    Ok(core.compile_model(&model, cpu)?)
}

/// Effective CPU plugin properties of a compiled model, for the report.
pub fn describe(compiled: &CompiledModel) -> serde_json::Value {
    let get = |key: PropertyKey| {
        compiled
            .get_property(&key)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "?".into())
    };
    serde_json::json!({
        "PERFORMANCE_HINT": get(PropertyKey::Rw(RwPropertyKey::HintPerformanceMode)),
        "INFERENCE_NUM_THREADS": get(PropertyKey::Rw(RwPropertyKey::InferenceNumThreads)),
        "NUM_STREAMS": get(PropertyKey::Rw(RwPropertyKey::NumStreams)),
        "SCHEDULING_CORE_TYPE": get(PropertyKey::Rw(RwPropertyKey::HintSchedulingCoreType)),
        "ENABLE_HYPER_THREADING": get(PropertyKey::Rw(RwPropertyKey::HintEnableHyperThreading)),
        "ENABLE_CPU_PINNING": get(PropertyKey::Rw(RwPropertyKey::HintEnableCpuPinning)),
        "INFERENCE_PRECISION_HINT": get(PropertyKey::Rw(RwPropertyKey::HintInferencePrecision)),
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS": get(PropertyKey::OptimalNumberOfInferRequests),
    })
}

fn f32_tensor(shape: &[usize], data: &[f32]) -> Result<Tensor> {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let mut tensor = Tensor::new(ElementType::F32, &Shape::new(&dims)?)?;
    let dst = tensor.get_data_mut::<f32>()?;
    if dst.len() != data.len() {
        bail!("tensor size mismatch: {} vs {}", dst.len(), data.len());
    }
    dst.copy_from_slice(data);
    Ok(tensor)
}

/// Runs one inference and returns the first output as (shape, data).
fn infer(
    request: &mut InferRequest,
    shape: &[usize],
    data: &[f32],
) -> Result<(Vec<usize>, Vec<f32>)> {
    let input = f32_tensor(shape, data)?;
    request.set_input_tensor(&input)?;
    request.infer()?;
    let output = request.get_output_tensor_by_index(0)?;
    let out_shape: Vec<usize> = output
        .get_shape()?
        .get_dimensions()
        .iter()
        .map(|&d| d as usize)
        .collect();
    Ok((out_shape, output.get_data::<f32>()?.to_vec()))
}

/// DBNet detection on OpenVINO (one infer request, used sequentially).
pub struct OvDetSession {
    pub compiled: CompiledModel,
    request: Mutex<InferRequest>,
}

impl OvDetSession {
    pub fn new(mut compiled: CompiledModel) -> Result<Self> {
        let request = compiled.create_infer_request()?;
        Ok(Self {
            compiled,
            request: Mutex::new(request),
        })
    }

    /// Same contract as `DetInferenceSession::run`.
    pub fn run(&self, input: &PreprocessedDetInput) -> Result<DetInferenceOutput> {
        let view = input
            .tensor
            .to_plain_array_view::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        let data = view.as_slice().context("non-contiguous input")?;
        let (shape, out) = infer(
            &mut self.request.lock().unwrap(),
            input.tensor.shape(),
            data,
        )?;
        // [1, 1, H, W]
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
            bail!("unexpected detection output shape {shape:?}");
        }
        let probability_map = Array2::from_shape_vec((shape[2], shape[3]), out)?;
        Ok(DetInferenceOutput { probability_map })
    }
}

/// SVTR recognition on OpenVINO with `n` infer requests. With several
/// requests, batches run concurrently on separate threads (OpenVINO streams),
/// mirroring how pure-onnx-ocr runs recognition batches in parallel.
pub struct OvRecSession {
    pub compiled: CompiledModel,
    requests: Vec<Mutex<InferRequest>>,
}

impl OvRecSession {
    pub fn new(mut compiled: CompiledModel, requests: usize) -> Result<Self> {
        let requests = (0..requests.max(1))
            .map(|_| compiled.create_infer_request().map(Mutex::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { compiled, requests })
    }

    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Same contract as `RecInferenceSession::run` (one batch, request 0).
    pub fn run(&self, batch: &PreprocessedRecBatch) -> Result<RecInferenceOutput> {
        run_rec(&self.requests[0], batch)
    }

    /// Runs every batch, spreading them over all infer requests. Results keep
    /// the input order.
    pub fn run_all(&self, batches: &[PreprocessedRecBatch]) -> Result<Vec<RecInferenceOutput>> {
        let workers = self.requests.len().min(batches.len());
        if workers <= 1 {
            return batches.iter().map(|b| self.run(b)).collect();
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut results: Vec<Option<Result<RecInferenceOutput>>> =
            (0..batches.len()).map(|_| None).collect();
        let slots: Vec<Mutex<&mut Option<Result<RecInferenceOutput>>>> =
            results.iter_mut().map(Mutex::new).collect();
        // Only the requests (Sync) cross threads; CompiledModel is not Sync.
        let requests = &self.requests;
        std::thread::scope(|scope| {
            for request in &requests[..workers] {
                let (next, slots) = (&next, &slots);
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index >= batches.len() {
                        break;
                    }
                    **slots[index].lock().unwrap() = Some(run_rec(request, &batches[index]));
                });
            }
        });
        drop(slots);
        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

fn run_rec(
    request: &Mutex<InferRequest>,
    batch: &PreprocessedRecBatch,
) -> Result<RecInferenceOutput> {
    let view = batch
        .tensor
        .to_plain_array_view::<f32>()
        .map_err(|e| anyhow!("{e}"))?;
    let data = view.as_slice().context("non-contiguous input")?;
    let in_shape = batch.tensor.shape();
    let (shape, out) = infer(&mut request.lock().unwrap(), in_shape, data)?;
    if shape.len() != 3 || shape[0] != in_shape[0] {
        bail!("unexpected recognition output shape {shape:?} for input {in_shape:?}");
    }
    let logits = Array3::from_shape_vec((shape[0], shape[1], shape[2]), out)?;
    Ok(RecInferenceOutput {
        valid_timesteps: valid_timesteps(batch, shape[1]),
        logits,
    })
}

/// Copy of the valid-length estimate in `RecInferenceSession::run_on`:
/// the number of output time steps that cover each crop's unpadded width.
fn valid_timesteps(batch: &PreprocessedRecBatch, time_steps: usize) -> Vec<usize> {
    let max_width = batch.max_width as f32;
    let scale = if max_width > 0.0 {
        time_steps as f32 / max_width
    } else {
        0.0
    };
    batch
        .valid_widths
        .iter()
        .map(|width| {
            let steps = if scale > 0.0 {
                (scale * *width as f32).round() as isize
            } else {
                time_steps as isize
            };
            steps.clamp(1, time_steps as isize) as usize
        })
        .collect()
}
