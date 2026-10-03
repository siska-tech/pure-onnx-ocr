use crate::paddle_config::ColorOrder;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
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
    pub tensor: Tensor,
    pub resized_dims: (u32, u32),
    pub scale_ratio: f64,
}

/// DBNet detection preprocessor.
#[derive(Debug, Clone)]
pub struct DetPreProcessor {
    config: DetPreProcessorConfig,
}

impl DetPreProcessor {
    pub fn new(config: DetPreProcessorConfig) -> Self {
        Self { config }
    }

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

        let resized = if resized_w == orig_w && resized_h == orig_h {
            image.clone()
        } else {
            // Bilinear filtering mirrors the `cv2.resize` default used by PaddleOCR.
            image.resize_exact(resized_w, resized_h, FilterType::Triangle)
        };

        let rgb_image = resized.to_rgb8();
        let padded_w = round_up_to_multiple(resized_w, 32);
        let padded_h = round_up_to_multiple(resized_h, 32);

        // Padding stays at 0.0, i.e. the normalised mean colour, so the padded
        // border does not introduce artificial edges.
        let mut array = Array4::<f32>::zeros((1, 3, padded_h as usize, padded_w as usize));
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
            resized_dims: (padded_w, padded_h),
            scale_ratio,
        })
    }
}

fn compute_resized_dims(
    orig_w: u32,
    orig_h: u32,
    limit_side_len: u32,
    limit_type: DetLimitType,
    max_side_limit: u32,
) -> (u32, u32, f64) {
    let max_side = orig_w.max(orig_h) as f64;
    let min_side = orig_w.min(orig_h) as f64;
    let limit = limit_side_len as f64;

    let mut scale_ratio = if limit_side_len == 0 {
        1.0
    } else {
        match limit_type {
            DetLimitType::Max if max_side > limit => limit / max_side,
            DetLimitType::Min if min_side < limit => limit / min_side,
            _ => 1.0,
        }
    };

    if max_side_limit > 0 && max_side * scale_ratio > max_side_limit as f64 {
        scale_ratio = max_side_limit as f64 / max_side;
    }

    if (scale_ratio - 1.0).abs() < f64::EPSILON {
        return (orig_w, orig_h, 1.0);
    }

    let resized_w = ((orig_w as f64 * scale_ratio).round().max(1.0)) as u32;
    let resized_h = ((orig_h as f64 * scale_ratio).round().max(1.0)) as u32;

    (resized_w, resized_h, scale_ratio)
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
    pub tensor: Tensor,
    pub valid_widths: Vec<u32>,
    pub max_width: u32,
}

impl PreprocessedRecBatch {
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
    pub fn new(config: RecPreProcessorConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration used by this preprocessor.
    pub fn config(&self) -> &RecPreProcessorConfig {
        &self.config
    }

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

            if region.x >= img_w
                || region.y >= img_h
                || region.x + region.width > img_w
                || region.y + region.height > img_h
            {
                return Err(RecPreProcessorError::RegionOutOfBounds {
                    index,
                    image_dims: (img_w, img_h),
                    region,
                });
            }
        }

        let target_height = self.config.target_height;
        let desired_widths: Vec<u32> = regions
            .iter()
            .map(|region| {
                let aspect_ratio = region.width as f64 / region.height as f64;
                (aspect_ratio * target_height as f64).ceil().max(1.0) as u32
            })
            .collect();
        let max_width = self.batch_width(desired_widths.iter().copied().max().unwrap_or(1));
        let batch_size = regions.len();

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

        for (index, region) in regions.iter().copied().enumerate() {
            let target_width = desired_widths[index].clamp(1, width_cap);
            let cropped = image.crop_imm(region.x, region.y, region.width, region.height);
            // Bilinear filtering mirrors the `cv2.resize` default used by PaddleOCR.
            let resized = cropped.resize_exact(target_width, target_height, FilterType::Triangle);
            let rgb_image = resized.to_rgb8();

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
        assert_eq!(result.resized_dims, (416, 64));
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
    fn detection_tensor_dims_are_padded_to_multiple_of_32() {
        let image = solid_image(123, 77, 200);
        let preprocessor = DetPreProcessor::new(DetPreProcessorConfig::default());

        let result = preprocessor.process(&image).unwrap();

        assert_eq!(result.resized_dims, (128, 96));
        assert_eq!(result.tensor.shape(), &[1, 3, 96, 128]);
        assert!((result.scale_ratio - 1.0).abs() < f64::EPSILON);
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
