//! Orientation classifiers used by the PaddleOCR 3.x OCR pipeline.
//!
//! * Text-line orientation (`PP-LCNet_x*_textline_ori`): predicts whether a
//!   cropped text line is upright (`0_degree`) or upside down
//!   (`180_degree`). Upside-down crops are rotated by 180 degrees before
//!   recognition.
//! * Document orientation (`PP-LCNet_x1_0_doc_ori`): predicts whether the
//!   whole page is rotated by 0, 90, 180 or 270 degrees counter-clockwise.
//!   The page is rotated back before detection.
//!
//! Both are image classifiers with ImageNet normalisation on RGB input and a
//! softmax output of shape `[N, classes]`.

use crate::onnx_model::{load_paddle_onnx, load_paddle_onnx_from_bytes, PlanCache};
use crate::paddle_config::{PaddleConfigError, PaddleInferenceConfig};
use crate::preprocessing::{IMAGENET_MEAN, IMAGENET_STD};
use image::{imageops, imageops::FilterType, RgbImage};
use ndarray::Array4;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::anyhow;

/// How the classifier input is produced from an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierResize {
    /// Resize to exactly `width x height`, ignoring the aspect ratio
    /// (`ResizeImage.size`).
    Exact { width: u32, height: u32 },
    /// Resize the short side to `short` keeping the aspect ratio, then take a
    /// `crop x crop` centre crop (`ResizeImage.resize_short` + `CropImage`).
    ShortThenCenterCrop { short: u32, crop: u32 },
}

impl ClassifierResize {
    fn input_dims(self) -> (u32, u32) {
        match self {
            ClassifierResize::Exact { width, height } => (width, height),
            ClassifierResize::ShortThenCenterCrop { crop, .. } => (crop, crop),
        }
    }
}

/// Prediction for one image.
#[derive(Debug, Clone, PartialEq)]
pub struct OrientationPrediction {
    /// Index into the classifier's label list.
    pub class_index: usize,
    /// Label text from `inference.yml`, e.g. `180_degree` or `90`.
    pub label: String,
    /// Probability of the predicted class.
    pub score: f32,
    /// Angle parsed from the label (0, 90, 180 or 270). Rotating the image
    /// counter-clockwise by this angle makes it upright, as PaddleX does with
    /// `rotate_image(img, angle)`.
    pub angle: u32,
}

/// Errors produced while loading an orientation classifier.
#[derive(Debug)]
pub enum OrientationError {
    /// Reading `inference.yml` failed.
    Config {
        source: PaddleConfigError,
        path: PathBuf,
    },
    /// `inference.yml` lacks a field the classifier needs.
    MissingConfig { path: PathBuf, field: &'static str },
    /// A label could not be interpreted as an angle.
    InvalidLabel { label: String },
    /// Loading or running the ONNX model failed.
    Model(TractError),
}

impl fmt::Display for OrientationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrientationError::Config { source, path } => {
                write!(f, "failed to read classifier config {:?}: {}", path, source)
            }
            OrientationError::MissingConfig { path, field } => {
                write!(f, "classifier config {:?} does not define {}", path, field)
            }
            OrientationError::InvalidLabel { label } => {
                write!(
                    f,
                    "classifier label `{}` is not an orientation angle",
                    label
                )
            }
            OrientationError::Model(source) => write!(f, "classifier model error: {}", source),
        }
    }
}

impl std::error::Error for OrientationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OrientationError::Config { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Parses `0`, `90`, `180_degree`, ... into degrees.
fn label_to_angle(label: &str) -> Result<u32, OrientationError> {
    let digits: String = label.chars().take_while(|c| c.is_ascii_digit()).collect();
    match digits.parse::<u32>() {
        Ok(angle @ (0 | 90 | 180 | 270)) => Ok(angle),
        _ => Err(OrientationError::InvalidLabel {
            label: label.to_string(),
        }),
    }
}

/// Orientation classifier backed by a PaddleOCR `PP-LCNet` ONNX export.
#[derive(Debug)]
pub struct OrientationClassifier {
    base_model: InferenceModel,
    cache: std::sync::Mutex<PlanCache<usize>>,
    resize: ClassifierResize,
    mean: [f32; 3],
    std: [f32; 3],
    labels: Vec<String>,
    angles: Vec<u32>,
}

impl OrientationClassifier {
    /// Loads a classifier from a PaddleOCR model directory containing
    /// `inference.onnx` and `inference.yml`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, OrientationError> {
        let dir = dir.as_ref();
        Self::load(
            dir.join(crate::engine::PADDLE_MODEL_FILE),
            dir.join(crate::engine::PADDLE_CONFIG_FILE),
        )
    }

    /// Loads a classifier from an ONNX file and its `inference.yml`.
    pub fn load(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, OrientationError> {
        let config_path = config_path.as_ref();
        let config = PaddleInferenceConfig::from_path(config_path).map_err(|source| {
            OrientationError::Config {
                source,
                path: config_path.to_path_buf(),
            }
        })?;
        let model = load_paddle_onnx(model_path.as_ref()).map_err(OrientationError::Model)?;
        Self::from_parts(model, config, config_path)
    }

    /// Loads a classifier from ONNX bytes and the text of its
    /// `inference.yml`, both held in memory (for example fetched by a
    /// browser).
    pub fn from_bytes(model_bytes: &[u8], config_yaml: &str) -> Result<Self, OrientationError> {
        let origin = Path::new(crate::dictionary::IN_MEMORY);
        let config = PaddleInferenceConfig::from_yaml_str(config_yaml).map_err(|source| {
            OrientationError::Config {
                source,
                path: origin.to_path_buf(),
            }
        })?;
        let model = load_paddle_onnx_from_bytes(model_bytes).map_err(OrientationError::Model)?;
        Self::from_parts(model, config, origin)
    }

    fn from_parts(
        model: InferenceModel,
        config: PaddleInferenceConfig,
        config_path: &Path,
    ) -> Result<Self, OrientationError> {
        let missing = |field| OrientationError::MissingConfig {
            path: config_path.to_path_buf(),
            field,
        };

        let resize = match (config.cls_resize_size, config.cls_resize_short) {
            (Some([width, height]), _) => ClassifierResize::Exact { width, height },
            (None, Some(short)) => ClassifierResize::ShortThenCenterCrop {
                short,
                crop: config.cls_crop_size.unwrap_or(short),
            },
            (None, None) => return Err(missing("PreProcess ResizeImage")),
        };
        let labels = config
            .label_list
            .ok_or_else(|| missing("PostProcess.Topk.label_list"))?;
        let angles = labels
            .iter()
            .map(|label| label_to_angle(label))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            base_model: model,
            cache: std::sync::Mutex::new(PlanCache::new(2)),
            resize,
            mean: config.normalize_mean.unwrap_or(IMAGENET_MEAN),
            std: config.normalize_std.unwrap_or(IMAGENET_STD),
            labels,
            angles,
        })
    }

    /// Returns the class labels in model output order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Classifies every image. Images are processed in a single batch, so
    /// callers should chunk large inputs.
    pub fn classify(&self, images: &[RgbImage]) -> TractResult<Vec<OrientationPrediction>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let (width, height) = self.resize.input_dims();
        let mut batch = Array4::<f32>::zeros((images.len(), 3, height as usize, width as usize));
        for (index, image) in images.iter().enumerate() {
            let input = self.prepare(image);
            for (x, y, pixel) in input.enumerate_pixels() {
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    batch[[index, c, y as usize, x as usize]] =
                        (value - self.mean[c]) / self.std[c];
                }
            }
        }

        let plan = self.plan_for_batch(images.len())?;
        let tensor: Tensor = batch.into_dyn().into();
        let run_start = crate::time::Instant::now();
        let outputs = plan.run(tvec!(tensor.into()))?;
        log::debug!(
            "[Orientation] classified {} image(s) in {:?}",
            images.len(),
            run_start.elapsed()
        );
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("orientation classifier returned no outputs"))?;
        let scores = output.to_plain_array_view::<f32>()?;
        let scores = scores.into_dimensionality::<ndarray::Ix2>()?;
        if scores.ncols() != self.labels.len() {
            return Err(anyhow!(
                "classifier produced {} classes but its config lists {} labels",
                scores.ncols(),
                self.labels.len()
            ));
        }

        Ok(scores
            .rows()
            .into_iter()
            .map(|row| {
                let (class_index, score) =
                    row.iter()
                        .copied()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |best, (i, v)| {
                            if v > best.1 {
                                (i, v)
                            } else {
                                best
                            }
                        });
                OrientationPrediction {
                    class_index,
                    label: self.labels[class_index].clone(),
                    score,
                    angle: self.angles[class_index],
                }
            })
            .collect())
    }

    fn prepare(&self, image: &RgbImage) -> RgbImage {
        // Bilinear filtering mirrors the `cv2.resize` default used by PaddleX.
        match self.resize {
            ClassifierResize::Exact { width, height } => {
                imageops::resize(image, width, height, FilterType::Triangle)
            }
            ClassifierResize::ShortThenCenterCrop { short, crop } => {
                let (w, h) = image.dimensions();
                let scale = short as f64 / w.min(h).max(1) as f64;
                let new_w = ((w as f64 * scale).round() as u32).max(crop);
                let new_h = ((h as f64 * scale).round() as u32).max(crop);
                let resized = imageops::resize(image, new_w, new_h, FilterType::Triangle);
                let x = (new_w - crop) / 2;
                let y = (new_h - crop) / 2;
                imageops::crop_imm(&resized, x, y, crop, crop).to_image()
            }
        }
    }

    fn plan_for_batch(&self, batch: usize) -> TractResult<Arc<TypedRunnableModel>> {
        if let Some(plan) = crate::onnx_model::lock_cache(&self.cache).get(batch) {
            return Ok(plan);
        }
        let (width, height) = self.resize.input_dims();
        let mut model = self.base_model.clone();
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![batch, 3, height as usize, width as usize],
            ),
        )?;
        let compile_start = crate::time::Instant::now();
        let plan = model
            .into_typed()?
            .into_decluttered()?
            .into_optimized()?
            .into_runnable()?;
        log::debug!(
            "[Orientation] compiled plan for batch {} in {:?}",
            batch,
            compile_start.elapsed()
        );
        crate::onnx_model::lock_cache(&self.cache).insert(batch, Arc::clone(&plan));
        Ok(plan)
    }
}

/// Rotates `image` counter-clockwise by `angle` degrees (0, 90, 180, 270).
pub fn rotate_ccw(image: &RgbImage, angle: u32) -> RgbImage {
    match angle % 360 {
        90 => imageops::rotate270(image),
        180 => imageops::rotate180(image),
        270 => imageops::rotate90(image),
        _ => image.clone(),
    }
}

/// Maps a point from an image rotated counter-clockwise by `angle` back to
/// the coordinates of the original `(width, height)` image.
pub fn unrotate_point(x: f64, y: f64, angle: u32, original_dims: (u32, u32)) -> (f64, f64) {
    let (w, h) = (original_dims.0 as f64, original_dims.1 as f64);
    match angle % 360 {
        // CCW 90: (x, y) -> (y, w - x); inverse (u, v) -> (w - v, u)
        90 => (w - y, x),
        180 => (w - x, h - y),
        // CCW 270: (x, y) -> (h - y, x); inverse (u, v) -> (v, h - u)
        270 => (y, h - x),
        _ => (x, y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn parses_labels_into_angles() {
        assert_eq!(label_to_angle("0_degree").unwrap(), 0);
        assert_eq!(label_to_angle("180_degree").unwrap(), 180);
        assert_eq!(label_to_angle("90").unwrap(), 90);
        assert_eq!(label_to_angle("270").unwrap(), 270);
        assert!(label_to_angle("45").is_err());
        assert!(label_to_angle("upright").is_err());
    }

    #[test]
    fn unrotate_point_inverts_rotate_ccw() {
        // A 4 x 2 image with a single marked pixel at (3, 0).
        let mut image = RgbImage::new(4, 2);
        image.put_pixel(3, 0, Rgb([255, 0, 0]));
        for angle in [0, 90, 180, 270] {
            let rotated = rotate_ccw(&image, angle);
            let (px, py) = rotated
                .enumerate_pixels()
                .find(|(_, _, p)| p[0] == 255)
                .map(|(x, y, _)| (x, y))
                .unwrap();
            // Use the pixel centre so continuous coordinates map cleanly.
            let (x, y) = unrotate_point(px as f64 + 0.5, py as f64 + 0.5, angle, (4, 2));
            assert!(
                (x - 3.5).abs() < 1e-9 && (y - 0.5).abs() < 1e-9,
                "angle {}",
                angle
            );
        }
    }
}
