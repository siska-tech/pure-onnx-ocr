//! Image-to-tensor conversion for detection and recognition.
//!
//! Detection stretches to dimensions rounded to multiples of 32; recognition
//! resizes crops to a fixed height and pads on the right to a shared batch width.
//! Tensors use NCHW order, while dimension pairs use `(width, height)`.

use crate::paddle_config::ColorOrder;
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{s, Array4};
use tract_onnx::prelude::Tensor;

/// ImageNet mean used by PaddleOCR detection models (`NormalizeImage.mean`).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet std used by PaddleOCR detection models (`NormalizeImage.std`).
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// How `limit_side_len` constrains the detection input size
/// (PaddleOCR `DetResizeForTest.limit_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetLimitType {
    /// Downscale so the longest side is at most `limit_side_len`.
    /// Never upscales. This crate's historical default (960).
    Max,
    /// Upscale so the shortest side is at least `limit_side_len`; larger
    /// images keep their native resolution. PaddleOCR 3.x uses this with
    /// `limit_side_len = 64`, so detection effectively runs at full size.
    Min,
}

/// Configuration parameters for `DetPreProcessor`.
#[derive(Debug, Clone, Copy)]
pub struct DetPreProcessorConfig {
    /// Side length limit, interpreted according to `limit_type`.
    pub limit_side_len: u32,
    /// Whether `limit_side_len` bounds the longest or the shortest side.
    pub limit_type: DetLimitType,
    /// Hard upper bound for the longest side after the `limit_type` rule
    /// (PaddleOCR `max_side_limit`, 4000). `0` disables the bound.
    pub max_side_limit: u32,
    /// Per-channel mean subtracted after scaling pixels to `[0, 1]`.
    /// Indexed in model channel order (see `color_order`).
    pub mean: [f32; 3],
    /// Per-channel standard deviation applied after mean subtraction.
    pub std: [f32; 3],
    /// Channel order of the model input tensor.
    pub color_order: ColorOrder,
}

impl Default for DetPreProcessorConfig {
    fn default() -> Self {
        Self {
            limit_side_len: 960,
            limit_type: DetLimitType::Max,
            max_side_limit: 4000,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            color_order: ColorOrder::Bgr,
        }
    }
}

/// Error returned when detection preprocessing fails.
#[derive(Debug)]
pub enum DetPreProcessorError {
    /// The provided image has zero width or height.
    EmptyImage,
}

impl std::fmt::Display for DetPreProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetPreProcessorError::EmptyImage => {
                write!(f, "input image dimensions must be positive")
            }
        }
    }
}

impl std::error::Error for DetPreProcessorError {}

/// Result of detection preprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessedDetInput {
    /// Normalized `f32` tensor shaped `[1, 3, height, width]`.
    pub tensor: Tensor,
    /// Actual resized image size `(width, height)`, after alignment.
    pub resized_dims: (u32, u32),
    /// Uniform resize ratio before alignment; use `scale_xy` for coordinates.
    pub scale_ratio: f64,
    /// Per-axis scale `(resized_width / original_width, resized_height /
    /// original_height)`; differs slightly from `scale_ratio` because each
    /// side is rounded to a multiple of 32.
    pub scale_xy: (f64, f64),
}

impl PreprocessedDetInput {
    /// Factors that map probability-map coordinates back to the original
    /// image: `(original_width / resized_width, original_height / resized_height)`.
    pub fn inverse_scale(&self) -> (f64, f64) {
        let (sx, sy) = self.scale_xy;
        (
            if sx > 0.0 { 1.0 / sx } else { 1.0 },
            if sy > 0.0 { 1.0 / sy } else { 1.0 },
        )
    }
}

/// DBNet detection preprocessor.
#[derive(Debug, Clone)]
pub struct DetPreProcessor {
    config: DetPreProcessorConfig,
}

impl DetPreProcessor {
    /// Stores resize and normalization settings for subsequent images.
    pub fn new(config: DetPreProcessorConfig) -> Self {
        Self { config }
    }

    /// Resizes and normalizes an image, retaining the scales needed to map
    /// detections back to the original. Returns an error for an empty image.
    pub fn process(
        &self,
        image: &DynamicImage,
    ) -> Result<PreprocessedDetInput, DetPreProcessorError> {
        let (orig_w, orig_h) = image.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Err(DetPreProcessorError::EmptyImage);
        }

        let (resized_w, resized_h, scale_ratio) = compute_resized_dims(
            orig_w,
            orig_h,
            self.config.limit_side_len,
            self.config.limit_type,
            self.config.max_side_limit,
        );

        // Stretch (not pad) to the multiple-of-32 size with OpenCV-compatible
        // bilinear interpolation, as PaddleOCR does.
        let rgb_image = crate::imgproc::resize_bilinear(&image.to_rgb8(), resized_w, resized_h);
        let mut array = Array4::<f32>::zeros((1, 3, resized_h as usize, resized_w as usize));
        let order = self.config.color_order;
        let mut scale = [0f32; 3];
        let mut offset = [0f32; 3];
        for c in 0..3 {
            let std = if self.config.std[c].abs() > f32::EPSILON {
                self.config.std[c]
            } else {
                1.0
            };
            scale[c] = 1.0 / (255.0 * std);
            offset[c] = self.config.mean[c] / std;
        }

        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            for c in 0..3 {
                let value = pixel[order.source_channel(c)] as f32;
                array[[0, c, y as usize, x as usize]] = value * scale[c] - offset[c];
            }
        }

        let tensor: Tensor = array.into_dyn().into();

        Ok(PreprocessedDetInput {
            tensor,
            resized_dims: (resized_w, resized_h),
            scale_ratio,
            scale_xy: (
                resized_w as f64 / orig_w as f64,
                resized_h as f64 / orig_h as f64,
            ),
        })
    }
}

/// PaddleOCR `DetResizeForTest.resize_image_type0`: scale by the limit
/// rule, cap the longest side, then round each side to the nearest multiple
/// of 32 (ties to even, at least 32). The image is stretched to that size,
/// so the horizontal and vertical scales can differ slightly.
fn compute_resized_dims(
    orig_w: u32,
    orig_h: u32,
    limit_side_len: u32,
    limit_type: DetLimitType,
    max_side_limit: u32,
) -> (u32, u32, f64) {
    let (w, h) = (orig_w as f64, orig_h as f64);
    let limit = limit_side_len as f64;
    let ratio = if limit_side_len == 0 {
        1.0
    } else {
        match limit_type {
            DetLimitType::Max if w.max(h) > limit => limit / w.max(h),
            DetLimitType::Min if w.min(h) < limit => limit / w.min(h),
            _ => 1.0,
        }
    };
    let mut resize_w = (w * ratio).trunc();
    let mut resize_h = (h * ratio).trunc();
    let mut ratio = ratio;
    if max_side_limit > 0 && resize_w.max(resize_h) > max_side_limit as f64 {
        let cap = max_side_limit as f64 / resize_w.max(resize_h);
        resize_w = (resize_w * cap).trunc();
        resize_h = (resize_h * cap).trunc();
        ratio *= cap;
    }
    let to_32 = |v: f64| ((v / 32.0).round_ties_even() * 32.0).max(32.0) as u32;
    (to_32(resize_w), to_32(resize_h), ratio)
}

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }

    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + multiple - remainder
    }
}

/// Rectangle specifying the area to crop for recognition preprocessing.
#[derive(Debug, Clone, Copy)]
pub struct RecTextRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Configuration parameters for recognition preprocessing.
///
/// Defaults mirror PaddleOCR 3.x (`OCRReisizeNormImg` with
/// `RecResizeImg.image_shape = [3, 48, 320]`): each crop is resized to the
/// target height keeping its aspect ratio, the batch is padded to the widest
/// crop (at least `max_width`, at most `max_dynamic_width`), and pixels are
/// normalised with `(x / 255 - 0.5) / 0.5` in BGR order.
#[derive(Debug, Clone)]
pub struct RecPreProcessorConfig {
    /// Input height expected by the recognition model.
    pub target_height: u32,
    /// Minimum width of the batch tensor (PaddleOCR `image_shape[2]`).
    pub max_width: u32,
    /// Upper bound for the batch tensor width. Crops wider than this are
    /// squeezed horizontally. Set it equal to `max_width` to force a fixed
    /// input width.
    pub max_dynamic_width: u32,
    /// The batch width is rounded up to a multiple of this value so that
    /// compiled inference plans can be reused across batches.
    pub width_alignment: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Raw pixel value in `[0, 1]` used for padding. The default `0.5`
    /// becomes `0.0` after normalisation, matching PaddleOCR's zero padding.
    pub pad_value: [f32; 3],
    /// Channel order of the model input tensor.
    pub color_order: ColorOrder,
}

impl Default for RecPreProcessorConfig {
    fn default() -> Self {
        Self {
            target_height: 48,
            max_width: 320,
            max_dynamic_width: 3200,
            width_alignment: 32,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            pad_value: [0.5, 0.5, 0.5],
            color_order: ColorOrder::Bgr,
        }
    }
}

/// Errors that can be produced by recognition preprocessing.
#[derive(Debug)]
pub enum RecPreProcessorError {
    /// The provided batch of regions is empty.
    EmptyRegions,
    /// The input image has zero width or height.
    EmptyImage,
    /// The configuration contains an invalid parameter (e.g. zero height/width).
    InvalidConfiguration,
    /// A region had zero width or height.
    ZeroArea { index: usize },
    /// A region extended beyond the bounds of the image.
    RegionOutOfBounds {
        index: usize,
        image_dims: (u32, u32),
        region: RecTextRegion,
    },
}

impl std::fmt::Display for RecPreProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecPreProcessorError::EmptyRegions => {
                write!(f, "at least one text region is required for recognition")
            }
            RecPreProcessorError::EmptyImage => {
                write!(f, "input image dimensions must be positive")
            }
            RecPreProcessorError::InvalidConfiguration => {
                write!(f, "recognition preprocessor configuration is invalid")
            }
            RecPreProcessorError::ZeroArea { index } => {
                write!(f, "text region at index {} has zero area", index)
            }
            RecPreProcessorError::RegionOutOfBounds {
                index,
                image_dims,
                region,
            } => write!(
                f,
                "text region at index {} (x={}, y={}, w={}, h={}) exceeds image bounds {:?}",
                index, region.x, region.y, region.width, region.height, image_dims
            ),
        }
    }
}

impl std::error::Error for RecPreProcessorError {}

/// Result of recognition preprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessedRecBatch {
    /// Normalized `f32` tensor shaped `[batch, 3, target_height, max_width]`.
    pub tensor: Tensor,
    /// Unpadded resized width of each sample, in the same order as the tensor.
    pub valid_widths: Vec<u32>,
    /// Shared tensor width including right-hand padding.
    pub max_width: u32,
}

impl PreprocessedRecBatch {
    /// Fractions of the canvas occupied by each crop; zero for a zero-width canvas.
    pub fn valid_width_ratios(&self) -> Vec<f32> {
        if self.max_width == 0 {
            return vec![0.0; self.valid_widths.len()];
        }
        self.valid_widths
            .iter()
            .map(|width| *width as f32 / self.max_width as f32)
            .collect()
    }
}

/// SVTR recognition preprocessor.
#[derive(Debug, Clone)]
pub struct RecPreProcessor {
    config: RecPreProcessorConfig,
}

impl RecPreProcessor {
    /// Stores crop normalization and batch-width settings.
    pub fn new(config: RecPreProcessorConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration used by this preprocessor.
    pub fn config(&self) -> &RecPreProcessorConfig {
        &self.config
    }

    /// Crops axis-aligned regions from an image, preserving their input order.
    /// Empty images/regions, zero-area crops, out-of-bounds regions and zero
    /// target dimensions are reported as preprocessing errors.
    pub fn process(
        &self,
        image: &DynamicImage,
        regions: &[RecTextRegion],
    ) -> Result<PreprocessedRecBatch, RecPreProcessorError> {
        if regions.is_empty() {
            return Err(RecPreProcessorError::EmptyRegions);
        }

        if self.config.target_height == 0 || self.config.max_width == 0 {
            return Err(RecPreProcessorError::InvalidConfiguration);
        }

        let (img_w, img_h) = image.dimensions();
        if img_w == 0 || img_h == 0 {
            return Err(RecPreProcessorError::EmptyImage);
        }

        for (index, region) in regions.iter().copied().enumerate() {
            if region.width == 0 || region.height == 0 {
                return Err(RecPreProcessorError::ZeroArea { index });
            }

            // Check origins first so the remaining-space subtraction is safe.
            // Adding an untrusted width/height to an origin can overflow u32.
            if region.x >= img_w
                || region.y >= img_h
                || region.width > img_w - region.x
                || region.height > img_h - region.y
            {
                return Err(RecPreProcessorError::RegionOutOfBounds {
                    index,
                    image_dims: (img_w, img_h),
                    region,
                });
            }
        }

        let crops: Vec<RgbImage> = regions
            .iter()
            .map(|region| {
                image
                    .crop_imm(region.x, region.y, region.width, region.height)
                    .to_rgb8()
            })
            .collect();
        self.process_images(&crops)
    }

    /// Builds a recognition batch from already cropped text images (for
    /// example the perspective-corrected crops produced by
    /// [`crate::crop::crop_quad`]).
    ///
    /// Samples retain their order. Each crop is resized to `target_height`,
    /// capped to the allowed width, and padded on the right. Padding values
    /// pass through the same normalization as image pixels.
    pub fn process_images(
        &self,
        crops: &[RgbImage],
    ) -> Result<PreprocessedRecBatch, RecPreProcessorError> {
        if crops.is_empty() {
            return Err(RecPreProcessorError::EmptyRegions);
        }
        if self.config.target_height == 0 || self.config.max_width == 0 {
            return Err(RecPreProcessorError::InvalidConfiguration);
        }
        for (index, crop) in crops.iter().enumerate() {
            if crop.width() == 0 || crop.height() == 0 {
                return Err(RecPreProcessorError::ZeroArea { index });
            }
        }

        let target_height = self.config.target_height;
        let desired_widths: Vec<u32> = crops
            .iter()
            .map(|crop| {
                // PaddleOCR: the canvas is int(h * max_ratio) wide and the crop
                // ceil(h * ratio), so wide crops end up truncated, narrow ones
                // rounded up.
                let aspect_ratio = crop.width() as f64 / crop.height() as f64;
                let scaled = aspect_ratio * target_height as f64;
                let min_ratio = self.config.max_width as f64 / target_height as f64;
                let width = if aspect_ratio > min_ratio {
                    scaled.trunc()
                } else {
                    scaled.ceil()
                };
                width.max(1.0) as u32
            })
            .collect();
        let max_width = self.batch_width(desired_widths.iter().copied().max().unwrap_or(1));
        let batch_size = crops.len();

        let mut batch =
            Array4::<f32>::zeros((batch_size, 3, target_height as usize, max_width as usize));

        for sample in 0..batch_size {
            for channel in 0..3 {
                let pad = normalize_value(
                    self.config.pad_value[channel],
                    self.config.mean[channel],
                    self.config.std[channel],
                );
                batch.slice_mut(s![sample, channel, .., ..]).fill(pad);
            }
        }

        let width_cap = max_width.min(self.config.max_dynamic_width.max(self.config.max_width));
        let order = self.config.color_order;
        let mut valid_widths = Vec::with_capacity(batch_size);

        for (index, crop) in crops.iter().enumerate() {
            let target_width = desired_widths[index].clamp(1, width_cap);
            // Bilinear filtering mirrors the `cv2.resize` default used by PaddleOCR.
            let rgb_image = crate::imgproc::resize_bilinear(crop, target_width, target_height);

            for (x, y, pixel) in rgb_image.enumerate_pixels() {
                for channel in 0..3 {
                    let value = pixel[order.source_channel(channel)] as f32 / 255.0;
                    batch[[index, channel, y as usize, x as usize]] =
                        normalize_value(value, self.config.mean[channel], self.config.std[channel]);
                }
            }

            valid_widths.push(target_width);
        }

        let tensor: Tensor = batch.into_dyn().into();
        Ok(PreprocessedRecBatch {
            tensor,
            valid_widths,
            max_width,
        })
    }

    /// Computes the batch tensor width for the widest desired crop width.
    fn batch_width(&self, widest: u32) -> u32 {
        let min_width = self.config.max_width;
        let limit = self.config.max_dynamic_width.max(min_width);
        let width = widest.clamp(min_width, limit);
        let alignment = self.config.width_alignment.max(1);
        round_up_to_multiple(width, alignment)
    }
}

fn normalize_value(value: f32, mean: f32, std: f32) -> f32 {
    if std == 0.0 {
        0.0
    } else {
        (value - mean) / std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_image(width: u32, height: u32, value: u8) -> DynamicImage {
        let pixel = Rgb([value, value.saturating_sub(1), value.saturating_add(1)]);
        let buffer = ImageBuffer::from_pixel(width, height, pixel);
        DynamicImage::ImageRgb8(buffer)
    }

    fn gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut buffer = ImageBuffer::new(width, height);
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let base = ((x + y) % 256) as u8;
            let green = base.saturating_add(32);
            let blue = base.saturating_add(64);
            *pixel = Rgb([base, green, blue]);
        }
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn resize_long_side_to_limit() {
        let image = solid_image(1920, 1080, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (960, 544));
        assert!((result.scale_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn min_limit_keeps_native_resolution_for_large_images() {
        let image = solid_image(1920, 1080, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 64,
            limit_type: DetLimitType::Min,
            ..DetPreProcessorConfig::default()
        });
        let result = preprocessor.process(&image).unwrap();
        assert_eq!(result.resized_dims, (1920, 1088));
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_limit_upscales_small_images() {
        let image = solid_image(200, 32, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 64,
            limit_type: DetLimitType::Min,
            ..DetPreProcessorConfig::default()
        });
        let result = preprocessor.process(&image).unwrap();
        assert!((result.scale_ratio - 2.0).abs() < 1e-9);
        // 400 / 32 = 12.5 rounds half to even (Python round) -> 384.
        assert_eq!(result.resized_dims, (384, 64));
    }

    #[test]
    fn max_side_limit_caps_native_resolution() {
        let image = solid_image(8000, 1000, 128);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 64,
            limit_type: DetLimitType::Min,
            max_side_limit: 4000,
            ..DetPreProcessorConfig::default()
        });
        let result = preprocessor.process(&image).unwrap();
        assert!((result.scale_ratio - 0.5).abs() < 1e-9);
        assert_eq!(result.resized_dims, (4000, 512));
    }

    #[test]
    fn keep_original_size_when_within_limit() {
        let image = solid_image(800, 600, 64);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (800, 608));
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tensor_shape_and_normalization() {
        let image = solid_image(320, 320, 255);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig {
            limit_side_len: 320,
            ..DetPreProcessorConfig::default()
        });

        let result = preprocessor.process(&image).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 3, 320, 320]);

        let array = result.tensor.to_plain_array_view::<f32>().unwrap();
        // solid_image(255) yields RGB (255, 254, 255); tensor channels are B, G, R.
        let bgr = [255.0f32, 254.0, 255.0];
        for c in 0..3 {
            let expected = (bgr[c] / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            assert!((array[[0, c, 10, 10]] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn detection_uses_bgr_channel_order_by_default() {
        let buffer = ImageBuffer::from_pixel(32, 32, Rgb([255u8, 0, 0]));
        let image = DynamicImage::ImageRgb8(buffer);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());
        let result = preprocessor.process(&image).unwrap();
        let array = result.tensor.to_plain_array_view::<f32>().unwrap();

        // Red lands in the last channel (B, G, R).
        let red = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        let blue = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        assert!((array[[0, 2, 0, 0]] - red).abs() < 1e-5);
        assert!((array[[0, 0, 0, 0]] - blue).abs() < 1e-5);
    }

    #[test]
    fn recognition_width_grows_with_long_regions() {
        let image = gradient_image(1200, 60);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let regions = vec![
            RecTextRegion {
                x: 0,
                y: 0,
                width: 1200,
                height: 48,
            },
            RecTextRegion {
                x: 0,
                y: 0,
                width: 96,
                height: 48,
            },
        ];
        let batch = preprocessor.process(&image, &regions).unwrap();

        assert_eq!(batch.max_width, 1216);
        assert_eq!(batch.tensor.shape(), &[2, 3, 48, 1216]);
        assert_eq!(batch.valid_widths, vec![1200, 96]);
    }

    #[test]
    fn recognition_width_is_capped() {
        let image = gradient_image(1200, 60);
        let config = RecPreProcessorConfig {
            max_dynamic_width: 320,
            ..RecPreProcessorConfig::default()
        };
        let preprocessor = RecPreProcessor::new(config);
        let regions = vec![RecTextRegion {
            x: 0,
            y: 0,
            width: 1200,
            height: 48,
        }];
        let batch = preprocessor.process(&image, &regions).unwrap();

        assert_eq!(batch.max_width, 320);
        assert_eq!(batch.valid_widths, vec![320]);
    }

    #[test]
    fn detection_dims_round_to_nearest_multiple_of_32() {
        // PaddleOCR stretches each side to the nearest multiple of 32
        // (123 -> 128, 77 -> 64) instead of padding.
        let image = solid_image(123, 77, 200);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (128, 64));
        assert_eq!(result.tensor.shape(), &[1, 3, 64, 128]);
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
        let (sx, sy) = result.inverse_scale();
        assert!((sx - 123.0 / 128.0).abs() < 1e-12);
        assert!((sy - 77.0 / 64.0).abs() < 1e-12);
    }

    #[test]
    fn recognition_single_region_preprocessing() {
        let image = gradient_image(200, 100);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 20,
            y: 10,
            width: 80,
            height: 40,
        }];

        let preprocessor = RecPreProcessor::new(config.clone());
        let batch = preprocessor.process(&image, &regions).unwrap();

        let expected_shape = [
            1,
            3,
            config.target_height as usize,
            config.max_width as usize,
        ];
        assert_eq!(batch.tensor.shape(), &expected_shape);
        assert_eq!(batch.valid_widths, vec![96]);

        let tensor = batch.tensor.to_plain_array_view::<f32>().unwrap();
        let pad = normalize_value(config.pad_value[0], config.mean[0], config.std[0]);
        assert!(
            (tensor[[0, 0, 0, (config.max_width - 1) as usize]] - pad).abs() < 1e-6,
            "padded area should remain at pad value"
        );
        assert!(
            (tensor[[0, 0, 0, 0]] - pad).abs() > 1e-3,
            "cropped content should differ from pad value"
        );

        let ratios = batch.valid_width_ratios();
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 96.0 / config.max_width as f32).abs() < f32::EPSILON);
    }

    #[test]
    fn recognition_multiple_regions_padding() {
        let image = gradient_image(320, 160);
        let config = RecPreProcessorConfig::default();
        let regions = vec![
            RecTextRegion {
                x: 0,
                y: 0,
                width: 120,
                height: 60,
            },
            RecTextRegion {
                x: 150,
                y: 40,
                width: 40,
                height: 80,
            },
        ];

        let preprocessor = RecPreProcessor::new(config.clone());
        let batch = preprocessor.process(&image, &regions).unwrap();

        assert_eq!(batch.valid_widths, vec![96, 24]);

        let tensor = batch.tensor.to_plain_array_view::<f32>().unwrap();
        let pad = normalize_value(config.pad_value[0], config.mean[0], config.std[0]);

        // Ensure padding column for first sample is untouched.
        assert!((tensor[[0, 0, 10, (config.max_width - 1) as usize]] - pad).abs() < 1e-6);
        // Ensure padding column for second sample is untouched.
        assert!((tensor[[1, 1, 20, (config.max_width - 1) as usize]] - pad).abs() < 1e-6);
    }

    #[test]
    fn recognition_region_overflow_width_is_error() {
        let image = gradient_image(10, 10);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let region = RecTextRegion {
            x: 1,
            y: 0,
            width: u32::MAX,
            height: 1,
        };
        let error = preprocessor.process(&image, &[region]).unwrap_err();
        assert!(matches!(
            error,
            RecPreProcessorError::RegionOutOfBounds { index: 0, .. }
        ));
    }

    #[test]
    fn recognition_region_overflow_height_is_error() {
        let image = gradient_image(10, 10);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let region = RecTextRegion {
            x: 0,
            y: 1,
            width: 1,
            height: u32::MAX,
        };
        let error = preprocessor.process(&image, &[region]).unwrap_err();
        assert!(matches!(
            error,
            RecPreProcessorError::RegionOutOfBounds { index: 0, .. }
        ));
    }

    #[test]
    fn recognition_region_touching_image_edges_is_valid() {
        let image = gradient_image(10, 10);
        let preprocessor = RecPreProcessor::new(RecPreProcessorConfig::default());
        let region = RecTextRegion {
            x: 1,
            y: 1,
            width: 9,
            height: 9,
        };
        assert!(preprocessor.process(&image, &[region]).is_ok());
    }

    #[test]
    fn recognition_region_out_of_bounds_is_error() {
        let image = gradient_image(100, 50);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 80,
            y: 10,
            width: 30,
            height: 20,
        }];

        let preprocessor = RecPreProcessor::new(config);
        let error = preprocessor.process(&image, &regions).unwrap_err();
        assert!(matches!(
            error,
            RecPreProcessorError::RegionOutOfBounds { index: 0, .. }
        ));
    }

    #[test]
    fn recognition_zero_area_region_is_error() {
        let image = gradient_image(100, 50);
        let config = RecPreProcessorConfig::default();
        let regions = vec![RecTextRegion {
            x: 10,
            y: 10,
            width: 0,
            height: 20,
        }];

        let preprocessor = RecPreProcessor::new(config);
        let error = preprocessor.process(&image, &regions).unwrap_err();
        assert!(matches!(error, RecPreProcessorError::ZeroArea { index: 0 }));
    }
}
