use crate::ctc::DecodedSequence;
use crate::detection::DetInferenceSession;
use crate::dictionary::{DictionaryError, RecDictionary};
use crate::paddle_config::{PaddleConfigError, PaddleInferenceConfig};
use crate::postprocessing::{
    DetPolygonScaler, DetPolygonScalerConfig, DetPolygonUnclipper, DetPolygonUnclipperConfig,
    DetPostProcessor, DetPostProcessorConfig, DetPostProcessorError,
};
use crate::preprocessing::{
    DetLimitType, DetPreProcessor, DetPreProcessorConfig, DetPreProcessorError, RecPreProcessor,
    RecPreProcessorConfig, RecPreProcessorError, RecTextRegion,
};
use crate::recognition::{
    RecInferenceSession, RecPostProcessor, RecPostProcessorConfig, RecPostProcessorError,
};
use geo_types::Polygon;
use image::{DynamicImage, GenericImageView, ImageError};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tract_onnx::prelude::TractError;

/// Errors that can occur while building or using the OCR engine.
#[derive(Debug)]
pub enum OcrError {
    /// A required builder field was not provided.
    MissingField { field: &'static str },
    /// An IO error occurred while accessing a resource.
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    /// Loading an ONNX model failed.
    ModelLoad { source: TractError, path: PathBuf },
    /// Loading the recognition dictionary failed.
    Dictionary { source: DictionaryError },
    /// Reading a PaddleOCR `inference.yml` failed.
    ModelConfig {
        source: PaddleConfigError,
        path: PathBuf,
    },
    /// The provided configuration contained invalid values.
    InvalidConfiguration { message: String },
    /// Failed to decode the input image.
    ImageDecode { source: ImageError, path: PathBuf },
    /// Detection preprocessing failed.
    DetectionPreprocess { source: DetPreProcessorError },
    /// Detection inference failed.
    DetectionInference { source: TractError },
    /// Detection post-processing failed.
    DetectionPostProcess { source: DetPostProcessorError },
    /// Recognition preprocessing failed.
    RecognitionPreprocess { source: RecPreProcessorError },
    /// Recognition inference failed.
    RecognitionInference { source: TractError },
    /// Recognition post-processing failed.
    RecognitionPostProcess { source: RecPostProcessorError },
    /// The number of recognition results did not match detected regions.
    PipelineMismatch {
        detection_regions: usize,
        recognition_results: usize,
    },
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::MissingField { field } => {
                write!(f, "required builder field `{}` was not provided", field)
            }
            OcrError::Io { path, source } => {
                write!(f, "failed to access resource {:?}: {}", path, source)
            }
            OcrError::ModelLoad { path, source } => {
                write!(f, "failed to load ONNX model {:?}: {}", path, source)
            }
            OcrError::Dictionary { source } => write!(f, "failed to load dictionary: {}", source),
            OcrError::ModelConfig { path, source } => {
                write!(f, "failed to read model config {:?}: {}", path, source)
            }
            OcrError::InvalidConfiguration { message } => write!(f, "{}", message),
            OcrError::ImageDecode { path, source } => {
                write!(f, "failed to decode image {:?}: {}", path, source)
            }
            OcrError::DetectionPreprocess { source } => {
                write!(f, "detection preprocessing failed: {}", source)
            }
            OcrError::DetectionInference { source } => {
                write!(f, "detection inference failed: {}", source)
            }
            OcrError::DetectionPostProcess { source } => {
                write!(f, "detection post-processing failed: {}", source)
            }
            OcrError::RecognitionPreprocess { source } => {
                write!(f, "recognition preprocessing failed: {}", source)
            }
            OcrError::RecognitionInference { source } => {
                write!(f, "recognition inference failed: {}", source)
            }
            OcrError::RecognitionPostProcess { source } => {
                write!(f, "recognition post-processing failed: {}", source)
            }
            OcrError::PipelineMismatch {
                detection_regions,
                recognition_results,
            } => write!(
                f,
                "pipeline mismatch: detection produced {} regions but recognition returned {} results",
                detection_regions, recognition_results
            ),
        }
    }
}

impl From<DetPreProcessorError> for OcrError {
    fn from(source: DetPreProcessorError) -> Self {
        OcrError::DetectionPreprocess { source }
    }
}

impl From<DetPostProcessorError> for OcrError {
    fn from(source: DetPostProcessorError) -> Self {
        OcrError::DetectionPostProcess { source }
    }
}

impl From<RecPreProcessorError> for OcrError {
    fn from(source: RecPreProcessorError) -> Self {
        OcrError::RecognitionPreprocess { source }
    }
}

impl From<RecPostProcessorError> for OcrError {
    fn from(source: RecPostProcessorError) -> Self {
        OcrError::RecognitionPostProcess { source }
    }
}

fn polygons_to_text_regions(
    polygons: &[Polygon<f64>],
    image_dims: (u32, u32),
) -> Vec<RecTextRegion> {
    polygons
        .iter()
        .map(|polygon| polygon_to_text_region(polygon, image_dims))
        .collect()
}

fn polygon_to_text_region(polygon: &Polygon<f64>, image_dims: (u32, u32)) -> RecTextRegion {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for point in polygon.exterior().points() {
        let x = point.x();
        let y = point.y();
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    let image_width = image_dims.0.max(1);
    let image_height = image_dims.1.max(1);
    let width_limit = image_width as f64;
    let height_limit = image_height as f64;

    let mut x1 = min_x.floor().max(0.0);
    let mut y1 = min_y.floor().max(0.0);
    let mut x2 = max_x.ceil().min(width_limit);
    let mut y2 = max_y.ceil().min(height_limit);

    if x2 <= x1 {
        x2 = (x1 + 1.0).min(width_limit);
    }
    if y2 <= y1 {
        y2 = (y1 + 1.0).min(height_limit);
    }

    if x2 <= x1 {
        x1 = (width_limit - 1.0).max(0.0);
        x2 = width_limit;
    }
    if y2 <= y1 {
        y1 = (height_limit - 1.0).max(0.0);
        y2 = height_limit;
    }

    let mut x = x1.floor() as u32;
    let mut y = y1.floor() as u32;
    if x >= image_width {
        x = image_width - 1;
    }
    if y >= image_height {
        y = image_height - 1;
    }

    let mut width = (x2 - x1).ceil().max(1.0) as u32;
    let mut height = (y2 - y1).ceil().max(1.0) as u32;

    if x + width > image_width {
        width = image_width.saturating_sub(x);
    }
    if y + height > image_height {
        height = image_height.saturating_sub(y);
    }

    if width == 0 {
        width = 1;
    }
    if height == 0 {
        height = 1;
    }

    RecTextRegion {
        x,
        y,
        width,
        height,
    }
}

impl Error for OcrError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OcrError::MissingField { .. } => None,
            OcrError::Io { source, .. } => Some(source),
            OcrError::ModelLoad { .. } => None,
            OcrError::Dictionary { source } => Some(source),
            OcrError::ModelConfig { source, .. } => Some(source),
            OcrError::InvalidConfiguration { .. } => None,
            OcrError::ImageDecode { source, .. } => Some(source),
            OcrError::DetectionPreprocess { source } => Some(source),
            OcrError::DetectionInference { .. } => None,
            OcrError::DetectionPostProcess { source } => Some(source),
            OcrError::RecognitionPreprocess { source } => Some(source),
            OcrError::RecognitionInference { .. } => None,
            OcrError::RecognitionPostProcess { source } => Some(source),
            OcrError::PipelineMismatch { .. } => None,
        }
    }
}

impl From<DictionaryError> for OcrError {
    fn from(source: DictionaryError) -> Self {
        Self::Dictionary { source }
    }
}

/// Aggregated configuration used by [`OcrEngine`] during inference.
#[derive(Debug, Clone)]
pub struct OcrEngineConfig {
    pub det_preprocessor: DetPreProcessorConfig,
    pub det_postprocessor: DetPostProcessorConfig,
    pub det_unclipper: DetPolygonUnclipperConfig,
    pub det_polygon_scaler: DetPolygonScalerConfig,
    pub rec_preprocessor: RecPreProcessorConfig,
    pub rec_postprocessor: RecPostProcessorConfig,
    pub rec_batch_size: usize,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            det_preprocessor: DetPreProcessorConfig::default(),
            det_postprocessor: DetPostProcessorConfig::default(),
            det_unclipper: DetPolygonUnclipperConfig::default(),
            det_polygon_scaler: DetPolygonScalerConfig::default(),
            rec_preprocessor: RecPreProcessorConfig::default(),
            rec_postprocessor: RecPostProcessorConfig::default(),
            rec_batch_size: 8,
        }
    }
}

/// Fully prepared OCR engine orchestrating the detection and recognition pipelines.
///
/// The engine executes inference synchronously: upcoming methods such as
/// [`OcrEngine::run_from_path`](#method.run_from_path) and
/// [`OcrEngine::run_from_image`](#method.run_from_image) (implemented in later tasks)
/// will block the caller until the complete pipeline finishes. Internally, every heavy-weight
/// component (preprocessors, ONNX sessions, dictionary and post-processors) is wrapped in
/// `Arc`, allowing callers to share a single engine instance across threads or to clone the
/// engine for concurrent use when needed.
#[derive(Debug)]
pub struct OcrEngine {
    assets: EngineAssets,
    detection: DetectionPipeline,
    recognition: RecognitionPipeline,
    config: OcrEngineConfig,
}

/// Result of running the full OCR pipeline for a single detected region.
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: Polygon<f64>,
}

#[derive(Debug, Clone)]
pub struct StageTimings {
    pub preprocess: Duration,
    pub inference: Duration,
    pub postprocess: Duration,
}

impl StageTimings {
    fn zero() -> Self {
        Self {
            preprocess: Duration::ZERO,
            inference: Duration::ZERO,
            postprocess: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrTimings {
    pub total: Duration,
    pub image_decode: Duration,
    pub detection: StageTimings,
    pub recognition: StageTimings,
}

impl OcrTimings {
    fn new() -> Self {
        Self {
            total: Duration::ZERO,
            image_decode: Duration::ZERO,
            detection: StageTimings::zero(),
            recognition: StageTimings::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrRunWithMetrics {
    pub results: Vec<OcrResult>,
    pub timings: OcrTimings,
}

impl OcrEngine {
    fn new(
        det_model_path: PathBuf,
        rec_model_path: PathBuf,
        dictionary_path: PathBuf,
        det_session: DetInferenceSession,
        rec_session: RecInferenceSession,
        dictionary: RecDictionary,
        config: OcrEngineConfig,
    ) -> Self {
        let assets = EngineAssets::new(det_model_path, rec_model_path, dictionary_path);

        let det_session = Arc::new(det_session);
        let rec_session = Arc::new(rec_session);
        let dictionary = Arc::new(dictionary);

        let detection = DetectionPipeline::new(
            Arc::clone(&det_session),
            config.det_preprocessor,
            config.det_postprocessor,
            config.det_unclipper,
            config.det_polygon_scaler,
        );

        let recognition = RecognitionPipeline::new(
            Arc::clone(&rec_session),
            Arc::clone(&dictionary),
            config.rec_preprocessor.clone(),
            config.rec_postprocessor.clone(),
        );

        Self {
            assets,
            detection,
            recognition,
            config,
        }
    }

    /// Executes the full OCR pipeline on an image located on disk.
    pub fn run_from_path<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OcrResult>, OcrError> {
        let run = self.run_with_metrics_from_path(path)?;
        Ok(run.results)
    }

    /// Executes the full OCR pipeline on an image located on disk and returns benchmarking data.
    pub fn run_with_metrics_from_path<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<OcrRunWithMetrics, OcrError> {
        let overall_start = Instant::now();
        let path_ref = path.as_ref();
        let decode_start = Instant::now();
        let image = image::open(path_ref).map_err(|source| OcrError::ImageDecode {
            source,
            path: path_ref.to_path_buf(),
        })?;
        let mut run = self.run_with_metrics_from_image_impl(&image)?;
        run.timings.image_decode = decode_start.elapsed();
        run.timings.total = overall_start.elapsed();
        Ok(run)
    }

    /// Executes the full OCR pipeline on an image already loaded in memory.
    /// Returns the effective configuration for this engine.
    pub fn config(&self) -> &OcrEngineConfig {
        &self.config
    }

    /// Returns the path used for the detection model.
    pub fn det_model_path(&self) -> &Path {
        self.assets.det_model_path()
    }

    /// Returns the path used for the recognition model.
    pub fn rec_model_path(&self) -> &Path {
        self.assets.rec_model_path()
    }

    /// Returns the path used for the recognition dictionary.
    pub fn dictionary_path(&self) -> &Path {
        self.assets.dictionary_path()
    }

    /// Returns the configured recognition batch size.
    pub fn rec_batch_size(&self) -> usize {
        self.config.rec_batch_size
    }

    pub fn run_from_image(&self, image: &DynamicImage) -> Result<Vec<OcrResult>, OcrError> {
        let run = self.run_with_metrics_from_image_impl(image)?;
        Ok(run.results)
    }

    pub fn run_with_metrics_from_image(
        &self,
        image: &DynamicImage,
    ) -> Result<OcrRunWithMetrics, OcrError> {
        self.run_with_metrics_from_image_impl(image)
    }

    fn run_with_metrics_from_image_impl(
        &self,
        image: &DynamicImage,
    ) -> Result<OcrRunWithMetrics, OcrError> {
        let pipeline_start = Instant::now();
        let mut timings = OcrTimings::new();
        let image_dims = image.dimensions();

        let (polygons, detection_timings) = self
            .detection
            .detect_polygons_with_timings(image, image_dims)?;
        timings.detection = detection_timings;

        if polygons.is_empty() {
            timings.total = pipeline_start.elapsed();
            return Ok(OcrRunWithMetrics {
                results: Vec::new(),
                timings,
            });
        }

        let regions = polygons_to_text_regions(&polygons, image_dims);
        let (sequences, recognition_timings) =
            self.recognition
                .run_with_timings(image, &regions, self.config.rec_batch_size)?;
        timings.recognition = recognition_timings;

        if sequences.len() != polygons.len() {
            return Err(OcrError::PipelineMismatch {
                detection_regions: polygons.len(),
                recognition_results: sequences.len(),
            });
        }

        let results: Vec<OcrResult> = polygons
            .into_iter()
            .zip(sequences.into_iter())
            .map(|(polygon, sequence)| OcrResult {
                text: sequence.text,
                confidence: sequence.confidence,
                bounding_box: polygon,
            })
            .collect();

        timings.total = pipeline_start.elapsed();

        Ok(OcrRunWithMetrics { results, timings })
    }
}

/// File name of the ONNX graph inside a PaddleOCR 3.x model directory.
pub const PADDLE_MODEL_FILE: &str = "inference.onnx";
/// File name of the inference config inside a PaddleOCR 3.x model directory.
pub const PADDLE_CONFIG_FILE: &str = "inference.yml";

/// Builder for constructing [`OcrEngine`] instances.
///
/// Models can be supplied either as individual files
/// ([`det_model_path`](Self::det_model_path),
/// [`rec_model_path`](Self::rec_model_path),
/// [`dictionary_path`](Self::dictionary_path)) or as PaddleOCR 3.x model
/// directories ([`det_model_dir`](Self::det_model_dir),
/// [`rec_model_dir`](Self::rec_model_dir)) such as the PP-OCRv6 exports
/// published on Hugging Face, which contain `inference.onnx` and
/// `inference.yml`. When a model directory is used, the preprocessing
/// parameters and the recognition dictionary are read from `inference.yml`.
#[derive(Debug, Clone)]
pub struct OcrEngineBuilder {
    det_model_path: Option<PathBuf>,
    rec_model_path: Option<PathBuf>,
    dictionary_path: Option<PathBuf>,
    det_config_path: Option<PathBuf>,
    rec_config_path: Option<PathBuf>,
    det_limit_side_len: u32,
    det_limit_type: DetLimitType,
    det_max_side_limit: u32,
    det_unclip_ratio: Option<f32>,
    det_threshold: Option<f32>,
    det_box_threshold: Option<f32>,
    det_postprocess_from_model_config: bool,
    rec_batch_size: usize,
    rec_use_space_char: bool,
    det_plan_cache_capacity: usize,
    rec_plan_cache_capacity: usize,
}

impl Default for OcrEngineBuilder {
    fn default() -> Self {
        let pre = DetPreProcessorConfig::default();
        Self {
            det_model_path: None,
            rec_model_path: None,
            dictionary_path: None,
            det_config_path: None,
            rec_config_path: None,
            det_limit_side_len: pre.limit_side_len,
            det_limit_type: pre.limit_type,
            det_max_side_limit: pre.max_side_limit,
            det_unclip_ratio: None,
            det_threshold: None,
            det_box_threshold: None,
            det_postprocess_from_model_config: false,
            rec_batch_size: OcrEngineConfig::default().rec_batch_size,
            rec_use_space_char: true,
            det_plan_cache_capacity: crate::detection::DEFAULT_DET_PLAN_CACHE,
            rec_plan_cache_capacity: crate::recognition::DEFAULT_REC_PLAN_CACHE,
        }
    }
}

impl OcrEngineBuilder {
    /// Creates a new builder instance using default configuration values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the path to the DBNet detection ONNX model.
    pub fn det_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.det_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the path to the CTC recognition ONNX model.
    pub fn rec_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.rec_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the path to the recognition dictionary.
    ///
    /// Accepts either a plain text dictionary (one character per line, e.g.
    /// `ppocrv5_dict.txt`) or a PaddleOCR `inference.yml` whose
    /// `PostProcess.character_dict` holds the characters.
    pub fn dictionary_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.dictionary_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Uses a PaddleOCR 3.x detection model directory containing
    /// `inference.onnx` and (optionally) `inference.yml`.
    pub fn det_model_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        let dir = dir.as_ref();
        self.det_model_path = Some(dir.join(PADDLE_MODEL_FILE));
        let config = dir.join(PADDLE_CONFIG_FILE);
        self.det_config_path = config.exists().then_some(config);
        self
    }

    /// Uses a PaddleOCR 3.x recognition model directory containing
    /// `inference.onnx` and `inference.yml`. Unless
    /// [`dictionary_path`](Self::dictionary_path) is set explicitly, the
    /// dictionary embedded in `inference.yml` is used.
    pub fn rec_model_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        let dir = dir.as_ref();
        self.rec_model_path = Some(dir.join(PADDLE_MODEL_FILE));
        let config = dir.join(PADDLE_CONFIG_FILE);
        self.rec_config_path = config.exists().then_some(config);
        self
    }

    /// Reads detection preprocessing parameters (channel order and
    /// normalisation) from a PaddleOCR `inference.yml`.
    pub fn det_config_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.det_config_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Reads recognition parameters (channel order, input shape and, when no
    /// explicit dictionary is configured, the dictionary) from a PaddleOCR
    /// `inference.yml`.
    pub fn rec_config_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.rec_config_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the maximum side length for detection preprocessing.
    pub fn det_limit_side_len(mut self, len: u32) -> Self {
        self.det_limit_side_len = len;
        self
    }

    /// Selects how `det_limit_side_len` is applied (PaddleOCR `limit_type`).
    ///
    /// The default [`DetLimitType::Max`] with 960 downscales large images,
    /// which is fast on CPU. PaddleOCR 3.x runs detection with
    /// `DetLimitType::Min` and a limit of 64, i.e. at native resolution:
    ///
    /// ```no_run
    /// # use pure_onnx_ocr::{DetLimitType, OcrEngineBuilder};
    /// let builder = OcrEngineBuilder::new()
    ///     .det_limit_side_len(64)
    ///     .det_limit_type(DetLimitType::Min);
    /// ```
    pub fn det_limit_type(mut self, limit_type: DetLimitType) -> Self {
        self.det_limit_type = limit_type;
        self
    }

    /// Sets the hard upper bound for the longest detection input side
    /// (PaddleOCR `max_side_limit`, default 4000). `0` disables it.
    pub fn det_max_side_limit(mut self, limit: u32) -> Self {
        self.det_max_side_limit = limit;
        self
    }

    /// When enabled, the detection `inference.yml` `PostProcess` values
    /// (`thresh`, `box_thresh`, `unclip_ratio`, `max_candidates`) are used
    /// instead of the PaddleOCR pipeline defaults (0.3 / 0.6 / 1.5 / 1000).
    /// Values set explicitly through [`det_threshold`](Self::det_threshold),
    /// [`det_box_threshold`](Self::det_box_threshold) or
    /// [`det_unclip_ratio`](Self::det_unclip_ratio) still take precedence.
    pub fn det_postprocess_from_model_config(mut self, enabled: bool) -> Self {
        self.det_postprocess_from_model_config = enabled;
        self
    }

    /// Sets the unclip ratio used during polygon offsetting.
    pub fn det_unclip_ratio(mut self, ratio: f64) -> Self {
        self.det_unclip_ratio = Some(ratio as f32);
        self
    }

    /// Sets the probability threshold used to binarise the DBNet output
    /// (PaddleOCR `thresh`, default `0.3`).
    pub fn det_threshold(mut self, threshold: f32) -> Self {
        self.det_threshold = Some(threshold);
        self
    }

    /// Sets the minimum mean probability for a detected region
    /// (PaddleOCR `box_thresh`, default `0.6`).
    pub fn det_box_threshold(mut self, threshold: f32) -> Self {
        self.det_box_threshold = Some(threshold);
        self
    }

    /// Sets the maximum batch size for recognition.
    pub fn rec_batch_size(mut self, size: usize) -> Self {
        self.rec_batch_size = size;
        self
    }

    /// Controls whether `" "` is appended to the dictionary as the last class
    /// (PaddleOCR `use_space_char`, default `true`). PP-OCRv5 and PP-OCRv6
    /// recognition models are trained with the space class.
    pub fn rec_use_space_char(mut self, enabled: bool) -> Self {
        self.rec_use_space_char = enabled;
        self
    }

    /// Limits how many compiled inference plans are cached per model.
    ///
    /// `tract` compiles one plan per input shape (detection: image size,
    /// recognition: batch size and width). Each plan keeps its own optimised
    /// weights, so large models (PP-OCRv6 medium) benefit from a small limit.
    /// Defaults: 4 detection plans, 16 recognition plans.
    pub fn plan_cache_capacity(mut self, detection: usize, recognition: usize) -> Self {
        self.det_plan_cache_capacity = detection;
        self.rec_plan_cache_capacity = recognition;
        self
    }

    /// Consumes the builder and attempts to construct an [`OcrEngine`].
    pub fn build(self) -> Result<OcrEngine, OcrError> {
        let det_model_path = self.det_model_path.ok_or(OcrError::MissingField {
            field: "det_model_path",
        })?;
        let rec_model_path = self.rec_model_path.ok_or(OcrError::MissingField {
            field: "rec_model_path",
        })?;
        let dictionary_path = self
            .dictionary_path
            .or_else(|| self.rec_config_path.clone())
            .ok_or(OcrError::MissingField {
                field: "dictionary_path",
            })?;

        if self.rec_batch_size == 0 {
            return Err(OcrError::InvalidConfiguration {
                message: "rec_batch_size must be greater than zero".to_string(),
            });
        }

        verify_file_exists(&det_model_path)?;
        verify_file_exists(&rec_model_path)?;
        verify_file_exists(&dictionary_path)?;

        let det_model_config = self
            .det_config_path
            .as_deref()
            .map(load_model_config)
            .transpose()?;
        let rec_model_config = self
            .rec_config_path
            .as_deref()
            .map(load_model_config)
            .transpose()?;

        let mut config = OcrEngineConfig::default();
        config.det_preprocessor.limit_side_len = self.det_limit_side_len;
        config.det_preprocessor.limit_type = self.det_limit_type;
        config.det_preprocessor.max_side_limit = self.det_max_side_limit;
        config.rec_batch_size = self.rec_batch_size;

        if let Some(det_config) = &det_model_config {
            apply_det_model_config(&mut config, det_config)?;
            if self.det_postprocess_from_model_config {
                apply_det_postprocess_config(&mut config, det_config);
            }
        }
        if let Some(ratio) = self.det_unclip_ratio {
            config.det_unclipper.unclip_ratio = ratio;
        }
        if let Some(threshold) = self.det_threshold {
            config.det_postprocessor.threshold = threshold;
        }
        if let Some(threshold) = self.det_box_threshold {
            config.det_postprocessor.box_threshold = threshold;
        }
        if let Some(rec_config) = &rec_model_config {
            apply_rec_model_config(&mut config, rec_config)?;
        }

        let det_session =
            DetInferenceSession::load(&det_model_path).map_err(|source| OcrError::ModelLoad {
                source,
                path: det_model_path.clone(),
            })?;
        let rec_session = RecInferenceSession::load_with_input_height(
            &rec_model_path,
            config.rec_preprocessor.target_height,
        )
        .map_err(|source| OcrError::ModelLoad {
            source,
            path: rec_model_path.clone(),
        })?;

        det_session.set_plan_cache_capacity(self.det_plan_cache_capacity);
        rec_session.set_plan_cache_capacity(self.rec_plan_cache_capacity);

        let mut dictionary = RecDictionary::from_path(&dictionary_path)?;
        if self.rec_use_space_char {
            dictionary = dictionary.with_space_char();
        }
        config.rec_postprocessor.blank_id = dictionary.blank_id();

        Ok(OcrEngine::new(
            det_model_path,
            rec_model_path,
            dictionary_path,
            det_session,
            rec_session,
            dictionary,
            config,
        ))
    }
}

fn load_model_config(path: &Path) -> Result<PaddleInferenceConfig, OcrError> {
    PaddleInferenceConfig::from_path(path).map_err(|source| OcrError::ModelConfig {
        source,
        path: path.to_path_buf(),
    })
}

fn apply_det_model_config(
    config: &mut OcrEngineConfig,
    model_config: &PaddleInferenceConfig,
) -> Result<(), OcrError> {
    if let Some(name) = model_config.post_process_name.as_deref() {
        if name != "DBPostProcess" {
            return Err(OcrError::InvalidConfiguration {
                message: format!(
                    "detection model config uses unsupported post-process `{}` (expected DBPostProcess)",
                    name
                ),
            });
        }
    }
    if let Some(order) = model_config.color_order {
        config.det_preprocessor.color_order = order;
    }
    if let Some(mean) = model_config.normalize_mean {
        config.det_preprocessor.mean = mean;
    }
    if let Some(std) = model_config.normalize_std {
        config.det_preprocessor.std = std;
    }
    Ok(())
}

fn apply_det_postprocess_config(
    config: &mut OcrEngineConfig,
    model_config: &PaddleInferenceConfig,
) {
    if let Some(threshold) = model_config.det_thresh {
        config.det_postprocessor.threshold = threshold;
    }
    if let Some(threshold) = model_config.det_box_thresh {
        config.det_postprocessor.box_threshold = threshold;
    }
    if let Some(ratio) = model_config.det_unclip_ratio {
        config.det_unclipper.unclip_ratio = ratio;
    }
    if let Some(max) = model_config.det_max_candidates {
        config.det_postprocessor.max_candidates = max;
    }
}

fn apply_rec_model_config(
    config: &mut OcrEngineConfig,
    model_config: &PaddleInferenceConfig,
) -> Result<(), OcrError> {
    if let Some(name) = model_config.post_process_name.as_deref() {
        if name != "CTCLabelDecode" {
            return Err(OcrError::InvalidConfiguration {
                message: format!(
                    "recognition model config uses unsupported post-process `{}` (expected CTCLabelDecode)",
                    name
                ),
            });
        }
    }
    if let Some(order) = model_config.color_order {
        config.rec_preprocessor.color_order = order;
    }
    if let Some([channels, height, width]) = model_config.rec_image_shape {
        if channels != 3 || height == 0 || width == 0 {
            return Err(OcrError::InvalidConfiguration {
                message: format!(
                    "unsupported recognition image shape [{}, {}, {}]",
                    channels, height, width
                ),
            });
        }
        config.rec_preprocessor.target_height = height;
        config.rec_preprocessor.max_width = width;
        if config.rec_preprocessor.max_dynamic_width < width {
            config.rec_preprocessor.max_dynamic_width = width;
        }
    }
    Ok(())
}

fn verify_file_exists(path: &Path) -> Result<(), OcrError> {
    if let Err(source) = fs::metadata(path) {
        return Err(OcrError::Io {
            source,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[derive(Debug)]
struct EngineAssets {
    det_model_path: PathBuf,
    rec_model_path: PathBuf,
    dictionary_path: PathBuf,
}

impl EngineAssets {
    fn new(det_model_path: PathBuf, rec_model_path: PathBuf, dictionary_path: PathBuf) -> Self {
        Self {
            det_model_path,
            rec_model_path,
            dictionary_path,
        }
    }

    fn det_model_path(&self) -> &Path {
        self.det_model_path.as_path()
    }

    fn rec_model_path(&self) -> &Path {
        self.rec_model_path.as_path()
    }

    fn dictionary_path(&self) -> &Path {
        self.dictionary_path.as_path()
    }
}

#[derive(Debug)]
struct DetectionPipeline {
    preprocessor: DetPreProcessor,
    session: Arc<DetInferenceSession>,
    postprocessor: DetPostProcessor,
    unclipper: DetPolygonUnclipper,
    scaler: DetPolygonScaler,
}

impl DetectionPipeline {
    fn new(
        session: Arc<DetInferenceSession>,
        preprocessor: DetPreProcessorConfig,
        postprocessor: DetPostProcessorConfig,
        unclipper: DetPolygonUnclipperConfig,
        scaler: DetPolygonScalerConfig,
    ) -> Self {
        Self {
            preprocessor: DetPreProcessor::new(preprocessor),
            session,
            postprocessor: DetPostProcessor::new(postprocessor),
            unclipper: DetPolygonUnclipper::new(unclipper),
            scaler: DetPolygonScaler::new(scaler),
        }
    }

    fn detect_polygons_with_timings(
        &self,
        image: &DynamicImage,
        image_dims: (u32, u32),
    ) -> Result<(Vec<Polygon<f64>>, StageTimings), OcrError> {
        let preprocess_start = Instant::now();
        let preprocessed = self.preprocessor.process(image).map_err(OcrError::from)?;
        let preprocess_elapsed = preprocess_start.elapsed();

        let inference_start = Instant::now();
        let inference = self
            .session
            .run(&preprocessed)
            .map_err(|source| OcrError::DetectionInference { source })?;
        let inference_elapsed = inference_start.elapsed();

        let post_start = Instant::now();
        let contours = self
            .postprocessor
            .process(&inference)
            .map_err(OcrError::from)?;
        let unclipped = self.unclipper.unclip_contours(&contours);
        let scaled = self
            .scaler
            .scale_polygons(&unclipped, preprocessed.scale_ratio, image_dims);
        let post_elapsed = post_start.elapsed();

        let timings = StageTimings {
            preprocess: preprocess_elapsed,
            inference: inference_elapsed,
            postprocess: post_elapsed,
        };

        Ok((scaled, timings))
    }
}

#[derive(Debug)]
struct RecognitionPipeline {
    preprocessor: RecPreProcessor,
    session: Arc<RecInferenceSession>,
    postprocessor: RecPostProcessor,
}

impl RecognitionPipeline {
    fn new(
        session: Arc<RecInferenceSession>,
        dictionary: Arc<RecDictionary>,
        preprocessor: RecPreProcessorConfig,
        postprocessor: RecPostProcessorConfig,
    ) -> Self {
        let postprocessor = RecPostProcessor::new(Arc::clone(&dictionary), postprocessor);

        Self {
            preprocessor: RecPreProcessor::new(preprocessor),
            session,
            postprocessor,
        }
    }

    /// Recognises `regions` in batches of at most `batch_size`.
    ///
    /// Like PaddleOCR, regions are sorted by aspect ratio first so that each
    /// batch holds crops of similar width, which keeps padding (and thus
    /// wasted compute) small. Results are returned in the original order.
    fn run_with_timings(
        &self,
        image: &DynamicImage,
        regions: &[RecTextRegion],
        batch_size: usize,
    ) -> Result<(Vec<DecodedSequence>, StageTimings), OcrError> {
        let mut timings = StageTimings::zero();
        let mut order: Vec<usize> = (0..regions.len()).collect();
        order.sort_by(|&a, &b| aspect_ratio(&regions[a]).total_cmp(&aspect_ratio(&regions[b])));

        let mut results: Vec<Option<DecodedSequence>> = vec![None; regions.len()];
        for chunk in order.chunks(batch_size.max(1)) {
            let batch_regions: Vec<RecTextRegion> =
                chunk.iter().map(|&index| regions[index]).collect();

            let preprocess_start = Instant::now();
            let batch = self
                .preprocessor
                .process(image, &batch_regions)
                .map_err(OcrError::from)?;
            timings.preprocess += preprocess_start.elapsed();

            let inference_start = Instant::now();
            let inference = self
                .session
                .run(&batch)
                .map_err(|source| OcrError::RecognitionInference { source })?;
            timings.inference += inference_start.elapsed();

            let post_start = Instant::now();
            let sequences = self
                .postprocessor
                .process(&inference)
                .map_err(OcrError::from)?;
            timings.postprocess += post_start.elapsed();

            if sequences.len() != chunk.len() {
                return Err(OcrError::PipelineMismatch {
                    detection_regions: chunk.len(),
                    recognition_results: sequences.len(),
                });
            }
            for (&index, sequence) in chunk.iter().zip(sequences) {
                results[index] = Some(sequence);
            }
        }

        let sequences = results.into_iter().flatten().collect();
        Ok((sequences, timings))
    }
}

fn aspect_ratio(region: &RecTextRegion) -> f64 {
    region.width as f64 / region.height.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc::CtcGreedyDecoderError;
    use crate::dictionary::RecDictionary;
    use crate::postprocessing::DetPostProcessorError;
    use crate::preprocessing::{DetPreProcessorError, RecPreProcessorError};
    use crate::recognition::RecPostProcessorError;
    use std::env;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn existing_model_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
        let det = locate_ppocrv5_asset("det.onnx")?;
        let rec = locate_ppocrv5_asset("rec.onnx")?;
        let dict = locate_ppocrv5_asset("ppocrv5_dict.txt")?;
        Some((det, rec, dict))
    }

    fn temp_image_path(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.png", prefix, timestamp))
    }

    #[test]
    fn missing_det_model_path_returns_error() {
        let err = OcrEngineBuilder::new()
            .rec_model_path("rec.onnx")
            .dictionary_path("dict.txt")
            .build()
            .unwrap_err();

        match err {
            OcrError::MissingField { field } => assert_eq!(field, "det_model_path"),
            other => panic!("expected MissingField error, got {:?}", other),
        }
    }

    #[test]
    fn missing_dictionary_path_returns_error() {
        let err = OcrEngineBuilder::new()
            .det_model_path("det.onnx")
            .rec_model_path("rec.onnx")
            .build()
            .unwrap_err();

        match err {
            OcrError::MissingField { field } => assert_eq!(field, "dictionary_path"),
            other => panic!("expected MissingField error, got {:?}", other),
        }
    }

    #[test]
    fn zero_recognition_batch_size_is_rejected() {
        let err = OcrEngineBuilder::new()
            .det_model_path("det.onnx")
            .rec_model_path("rec.onnx")
            .dictionary_path("dict.txt")
            .rec_batch_size(0)
            .build()
            .unwrap_err();

        match err {
            OcrError::InvalidConfiguration { message } => {
                assert!(message.contains("rec_batch_size"));
            }
            other => panic!("expected InvalidConfiguration error, got {:?}", other),
        }
    }

    #[test]
    fn build_succeeds_when_paths_exist() {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .det_limit_side_len(1024)
            .det_unclip_ratio(2.0)
            .rec_batch_size(4)
            .build()
            .expect("engine should build successfully");

        assert_eq!(engine.config().det_preprocessor.limit_side_len, 1024);
        assert!((engine.config().det_unclipper.unclip_ratio - 2.0).abs() < f32::EPSILON);
        assert_eq!(engine.config().rec_batch_size, 4);
    }

    #[test]
    fn engine_reports_asset_paths_and_batch_size() {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .rec_batch_size(6)
            .build()
            .expect("engine should build successfully");

        assert_eq!(engine.det_model_path(), det.as_path());
        assert_eq!(engine.rec_model_path(), rec.as_path());
        assert_eq!(engine.dictionary_path(), dict.as_path());
        assert_eq!(engine.rec_batch_size(), 6);
    }

    #[test]
    fn recognition_blank_id_matches_dictionary_blank_id() {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let dictionary_blank_id = RecDictionary::from_path(&dict)
            .expect("dictionary should load successfully")
            .blank_id();

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .build()
            .expect("engine should build successfully");

        assert_eq!(
            engine.config().rec_postprocessor.blank_id,
            dictionary_blank_id
        );
    }

    #[test]
    fn run_from_path_processes_blank_image() -> Result<(), OcrError> {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .build()
            .expect("engine should build successfully");

        let temp_path = temp_image_path("run_path_blank");
        let image_buffer = image::ImageBuffer::from_pixel(64, 32, image::Rgb([0, 0, 0]));
        DynamicImage::ImageRgb8(image_buffer)
            .save(&temp_path)
            .expect("failed to save temporary image");

        let results = engine.run_from_path(&temp_path)?;
        assert!(
            results.len() <= engine.rec_batch_size(),
            "number of results should not exceed configured batch size"
        );

        std::fs::remove_file(&temp_path).ok();
        Ok(())
    }

    #[test]
    fn run_from_image_reuses_pipeline() -> Result<(), OcrError> {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .build()
            .expect("engine should build successfully");

        let image_buffer = image::ImageBuffer::from_pixel(32, 192, image::Rgb([255, 255, 255]));
        let dynamic_image = DynamicImage::ImageRgb8(image_buffer);
        let results = engine.run_from_image(&dynamic_image)?;

        assert!(
            results.len() <= engine.rec_batch_size(),
            "number of results should not exceed configured batch size"
        );

        Ok(())
    }

    #[test]
    fn run_with_metrics_reports_timings() -> Result<(), OcrError> {
        let (det, rec, dict) = existing_model_paths()
            .expect("expected PP-OCRv5 assets to be present under models/ppocrv5/");

        let engine = OcrEngineBuilder::new()
            .det_model_path(&det)
            .rec_model_path(&rec)
            .dictionary_path(&dict)
            .build()
            .expect("engine should build successfully");

        let image_buffer = image::ImageBuffer::from_pixel(16, 16, image::Rgb([0, 0, 0]));
        let dynamic_image = DynamicImage::ImageRgb8(image_buffer);

        let run_with_metrics = engine.run_with_metrics_from_image(&dynamic_image)?;
        let baseline_results = engine.run_from_image(&dynamic_image)?;

        assert_eq!(run_with_metrics.results.len(), baseline_results.len());
        assert!(run_with_metrics.timings.total >= run_with_metrics.timings.detection.preprocess);
        assert!(run_with_metrics.timings.recognition.preprocess <= run_with_metrics.timings.total);

        Ok(())
    }

    #[test]
    fn component_errors_convert_to_ocr_error_variants() {
        match OcrError::from(DetPreProcessorError::EmptyImage) {
            OcrError::DetectionPreprocess { .. } => {}
            other => panic!("expected DetectionPreprocess variant, got {:?}", other),
        }

        match OcrError::from(DetPostProcessorError::EmptyProbabilityMap) {
            OcrError::DetectionPostProcess { .. } => {}
            other => panic!("expected DetectionPostProcess variant, got {:?}", other),
        }

        match OcrError::from(RecPreProcessorError::EmptyRegions) {
            OcrError::RecognitionPreprocess { .. } => {}
            other => panic!("expected RecognitionPreprocess variant, got {:?}", other),
        }

        let rec_post_err = RecPostProcessorError::from(CtcGreedyDecoderError::EmptyBatch);
        match OcrError::from(rec_post_err) {
            OcrError::RecognitionPostProcess { .. } => {}
            other => panic!("expected RecognitionPostProcess variant, got {:?}", other),
        }
    }
}
