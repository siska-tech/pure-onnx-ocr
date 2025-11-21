use crate::inference::DetInferenceOutput;
use geo_types::{Coord, LineString, Polygon};
use i_overlay::float::overlay::OverlayOptions;
use i_overlay::mesh::outline::offset::OutlineOffset;
use i_overlay::mesh::style::{LineJoin, OutlineStyle};
use image::{GrayImage, Luma};
use imageproc::contours::{find_contours, Contour};
use imageproc::point::Point;
use ndarray::Array2;
use std::error::Error;
use std::fmt;

/// Configuration for `DetPostProcessor`.
#[derive(Debug, Clone, Copy)]
pub struct DetPostProcessorConfig {
    /// Probability threshold (0.0 - 1.0) applied before contour extraction.
    pub threshold: f32,
    /// Minimum contour area (in pixels) to keep.
    pub min_area: f32,
}

impl Default for DetPostProcessorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_area: 10.0,
        }
    }
}

/// Errors that can occur during detection post-processing.
#[derive(Debug)]
pub enum DetPostProcessorError {
    /// Probability map contained no elements.
    EmptyProbabilityMap,
    /// Failed to construct an image buffer from the probability map.
    ImageCreationFailed,
}

impl fmt::Display for DetPostProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetPostProcessorError::EmptyProbabilityMap => {
                write!(f, "probability map must contain at least one element")
            }
            DetPostProcessorError::ImageCreationFailed => {
                write!(f, "failed to create grayscale image from probability map")
            }
        }
    }
}

impl Error for DetPostProcessorError {}

/// Extracts text candidate contours from the DBNet probability map.
#[derive(Debug, Clone)]
pub struct DetPostProcessor {
    config: DetPostProcessorConfig,
}

impl DetPostProcessor {
    pub fn new(config: DetPostProcessorConfig) -> Self {
        Self { config }
    }

    pub fn process(
        &self,
        output: &DetInferenceOutput,
    ) -> Result<Vec<Contour<i32>>, DetPostProcessorError> {
        self.process_probability_map(&output.probability_map)
    }

    pub fn process_probability_map(
        &self,
        probability_map: &Array2<f32>,
    ) -> Result<Vec<Contour<i32>>, DetPostProcessorError> {
        if probability_map.is_empty() {
            return Err(DetPostProcessorError::EmptyProbabilityMap);
        }

        let threshold = self.config.threshold.clamp(0.0, 1.0);
        let (height, width) = probability_map.dim();
        let mut buffer = Vec::with_capacity(height * width);

        for &value in probability_map.iter() {
            let clamped = value.clamp(0.0, 1.0);
            let byte = if clamped >= threshold { 255 } else { 0 };
            buffer.push(byte);
        }

        let mut gray = GrayImage::from_vec(width as u32, height as u32, buffer)
            .ok_or(DetPostProcessorError::ImageCreationFailed)?;

        // Ensure the binary image uses full white for foreground for consistent contour detection.
        for pixel in gray.pixels_mut() {
            *pixel = if pixel[0] > 0 { Luma([255]) } else { Luma([0]) };
        }

        let contours = find_contours::<i32>(&gray);
        let min_area = self.config.min_area.max(0.0);

        let filtered = contours
            .into_iter()
            .filter(|contour| contour.points.len() >= 3)
            .filter(|contour| contour_area(contour) >= min_area)
            .collect();

        Ok(filtered)
    }
}

fn contour_area(contour: &Contour<i32>) -> f32 {
    if contour.points.len() < 3 {
        return 0.0;
    }

    let mut area = 0f64;
    for window in contour.points.windows(2) {
        if let [Point { x: x1, y: y1 }, Point { x: x2, y: y2 }] = window {
            area += (*x1 as f64) * (*y2 as f64) - (*x2 as f64) * (*y1 as f64);
        }
    }

    let first = contour.points.first().unwrap();
    let last = contour.points.last().unwrap();
    area += (last.x as f64) * (first.y as f64) - (first.x as f64) * (last.y as f64);

    (area.abs() * 0.5) as f32
}

/// Corner join style for unclip offsetting.
#[derive(Debug, Clone, Copy)]
pub enum DetUnclipLineJoin {
    Bevel,
    Miter(f32),
    Round(f32),
}

impl DetUnclipLineJoin {
    fn to_line_join(self) -> LineJoin<f64> {
        match self {
            DetUnclipLineJoin::Bevel => LineJoin::Bevel,
            DetUnclipLineJoin::Miter(angle) => LineJoin::Miter(angle.max(0.01) as f64),
            DetUnclipLineJoin::Round(angle) => LineJoin::Round(angle.max(0.01) as f64),
        }
    }
}

/// Configuration for polygon offsetting (unclip).
#[derive(Debug, Clone, Copy)]
pub struct DetPolygonUnclipperConfig {
    /// Ratio applied to the DBNet area/perimeter heuristic.
    pub unclip_ratio: f32,
    /// Additional minimum area after unclipping; polygons smaller than this are discarded.
    pub min_result_area: f32,
    /// Join style applied to buffered corners.
    pub join_style: DetUnclipLineJoin,
}

impl Default for DetPolygonUnclipperConfig {
    fn default() -> Self {
        Self {
            unclip_ratio: 1.5,
            min_result_area: 25.0,
            join_style: DetUnclipLineJoin::Round(0.1),
        }
    }
}

/// Applies DBNet-style polygon offsetting (unclip) using `i_overlay`.
#[derive(Debug, Clone)]
pub struct DetPolygonUnclipper {
    config: DetPolygonUnclipperConfig,
}

impl DetPolygonUnclipper {
    pub fn new(config: DetPolygonUnclipperConfig) -> Self {
        Self { config }
    }

    pub fn unclip_contours(&self, contours: &[Contour<i32>]) -> Vec<Polygon<f64>> {
        contours
            .iter()
            .filter_map(|contour| contour_to_polygon(contour))
            .flat_map(|polygon| self.unclip_polygon(&polygon))
            .filter(|polygon| polygon_area(polygon) >= self.config.min_result_area)
            .collect()
    }

    fn unclip_polygon(&self, polygon: &Polygon<f64>) -> Vec<Polygon<f64>> {
        let distance = unclip_distance(polygon, self.config.unclip_ratio.max(0.0));
        if distance <= f64::EPSILON {
            return vec![polygon.clone()];
        }

        let shape = polygon_to_shape(polygon);
        let style = OutlineStyle::default()
            .outer_offset(distance)
            .inner_offset(0.0)
            .line_join(self.config.join_style.to_line_join());

        let options = OverlayOptions::default();
        shape
            .outline_custom(&style, options)
            .into_iter()
            .filter_map(shape_to_polygon)
            .collect()
    }
}

/// Rounding strategy used when restoring polygon coordinates.
#[derive(Debug, Clone, Copy)]
pub enum DetScaleRounding {
    /// Do not apply rounding.
    None,
    /// Round to the specified number of fractional digits.
    FractionalDigits(u32),
}

impl Default for DetScaleRounding {
    fn default() -> Self {
        Self::FractionalDigits(2)
    }
}

/// Configuration for polygon scaling back to original image coordinates.
#[derive(Debug, Clone, Copy)]
pub struct DetPolygonScalerConfig {
    /// Whether to clamp coordinates to the original image bounds.
    pub clamp_to_image: bool,
    /// Rounding strategy applied after scaling.
    pub rounding: DetScaleRounding,
}

impl Default for DetPolygonScalerConfig {
    fn default() -> Self {
        Self {
            clamp_to_image: true,
            rounding: DetScaleRounding::FractionalDigits(2),
        }
    }
}

/// Scales polygons from resized space back to the original image coordinate system.
#[derive(Debug, Clone)]
pub struct DetPolygonScaler {
    config: DetPolygonScalerConfig,
}

impl DetPolygonScaler {
    pub fn new(config: DetPolygonScalerConfig) -> Self {
        Self { config }
    }

    /// Converts polygons generated in resized space back to the original image space.
    ///
    /// * `scale_ratio` - Resize ratio used during preprocessing (resized / original).
    /// * `original_dims` - Width and height of the original image (in pixels).
    pub fn scale_polygons(
        &self,
        polygons: &[Polygon<f64>],
        scale_ratio: f64,
        original_dims: (u32, u32),
    ) -> Vec<Polygon<f64>> {
        if scale_ratio <= f64::EPSILON {
            return polygons.to_vec();
        }

        let inverse_scale = 1.0 / scale_ratio;

        polygons
            .iter()
            .map(|polygon| self.scale_polygon(polygon, inverse_scale, original_dims))
            .collect()
    }

    fn scale_polygon(
        &self,
        polygon: &Polygon<f64>,
        inverse_scale: f64,
        original_dims: (u32, u32),
    ) -> Polygon<f64> {
        let exterior = self.scale_line_string(polygon.exterior(), inverse_scale, original_dims);
        let interiors = polygon
            .interiors()
            .iter()
            .map(|line| self.scale_line_string(line, inverse_scale, original_dims))
            .collect();

        Polygon::new(exterior, interiors)
    }

    fn scale_line_string(
        &self,
        line: &LineString<f64>,
        inverse_scale: f64,
        original_dims: (u32, u32),
    ) -> LineString<f64> {
        let precision = match self.config.rounding {
            DetScaleRounding::None => None,
            DetScaleRounding::FractionalDigits(p) => Some(p),
        };

        let mut coords: Vec<Coord<f64>> = line
            .points()
            .map(|p| {
                let mut x = p.x() * inverse_scale;
                let mut y = p.y() * inverse_scale;

                if self.config.clamp_to_image {
                    x = clamp_to_bounds(x, original_dims.0);
                    y = clamp_to_bounds(y, original_dims.1);
                }

                if let Some(precision) = precision {
                    x = round_fractional(x, precision);
                    y = round_fractional(y, precision);
                }

                Coord { x, y }
            })
            .collect();

        close_if_needed(&mut coords);
        LineString::from(coords)
    }
}

fn contour_to_polygon(contour: &Contour<i32>) -> Option<Polygon<f64>> {
    if contour.points.len() < 3 {
        return None;
    }

    let mut coords: Vec<Coord<f64>> = contour
        .points
        .iter()
        .map(|point| Coord {
            x: point.x as f64,
            y: point.y as f64,
        })
        .collect();

    close_if_needed(&mut coords);

    // Ensure outer contour is counter-clockwise.
    if signed_area_coords(&coords) < 0.0 {
        coords.reverse();
        close_if_needed(&mut coords);
    }

    let exterior = LineString::from(coords);
    Some(Polygon::new(exterior, Vec::new()))
}

fn polygon_to_shape(polygon: &Polygon<f64>) -> Vec<Vec<[f64; 2]>> {
    let mut shape = Vec::with_capacity(1 + polygon.interiors().len());
    shape.push(linestring_to_points(polygon.exterior(), true));
    for interior in polygon.interiors() {
        shape.push(linestring_to_points(interior, false));
    }
    shape
}

fn shape_to_polygon(shape: Vec<Vec<[f64; 2]>>) -> Option<Polygon<f64>> {
    if shape.is_empty() {
        return None;
    }

    let exterior = LineString::from(points_to_coords(&shape[0]));
    let interiors = shape
        .iter()
        .skip(1)
        .map(|points| LineString::from(points_to_coords(points)))
        .collect();

    Some(Polygon::new(exterior, interiors))
}

fn linestring_to_points(line: &LineString<f64>, want_ccw: bool) -> Vec<[f64; 2]> {
    let mut coords: Vec<Coord<f64>> = line
        .points()
        .map(|p| Coord { x: p.x(), y: p.y() })
        .collect();
    close_if_needed(&mut coords);

    let area = signed_area_coords(&coords);
    if want_ccw && area < 0.0 || !want_ccw && area > 0.0 {
        coords.reverse();
        close_if_needed(&mut coords);
    }

    coords.iter().map(|c| [c.x, c.y]).collect()
}

fn points_to_coords(points: &[[f64; 2]]) -> Vec<Coord<f64>> {
    let mut coords: Vec<Coord<f64>> = points
        .iter()
        .map(|point| Coord {
            x: point[0],
            y: point[1],
        })
        .collect();
    close_if_needed(&mut coords);
    coords
}

fn close_if_needed(coords: &mut Vec<Coord<f64>>) {
    if coords.len() < 2 {
        return;
    }
    let first = coords.first().copied().unwrap();
    let last = coords.last().copied().unwrap();
    if first.x != last.x || first.y != last.y {
        coords.push(first);
    }
}

fn signed_area_coords(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let mut area = 0.0;
    for window in coords.windows(2) {
        if let [a, b] = window {
            area += a.x * b.y - b.x * a.y;
        }
    }
    area * 0.5
}

fn perimeter_coords(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;
    for window in coords.windows(2) {
        if let [a, b] = window {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            length += (dx * dx + dy * dy).sqrt();
        }
    }
    length
}

fn polygon_area(polygon: &Polygon<f64>) -> f32 {
    let mut area = signed_area_coords(&points_to_coords(&linestring_to_points(
        polygon.exterior(),
        true,
    )))
    .abs();

    for interior in polygon.interiors() {
        area -= signed_area_coords(&points_to_coords(&linestring_to_points(interior, false))).abs();
    }

    area as f32
}

fn unclip_distance(polygon: &Polygon<f64>, ratio: f32) -> f64 {
    if ratio <= 0.0 {
        return 0.0;
    }

    let exterior_coords = points_to_coords(&linestring_to_points(polygon.exterior(), true));
    let area = signed_area_coords(&exterior_coords).abs();
    let perimeter = perimeter_coords(&exterior_coords);

    if perimeter <= f64::EPSILON {
        0.0
    } else {
        (area / perimeter) * ratio as f64
    }
}

fn clamp_to_bounds(value: f64, bound: u32) -> f64 {
    let upper = bound as f64;
    value.clamp(0.0, upper)
}

fn round_fractional(value: f64, digits: u32) -> f64 {
    let factor = 10_f64.powi(digits as i32);
    (value * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::contours::BorderType;
    use ndarray::array;

    #[test]
    fn extracts_single_square_contour() {
        let probability_map = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let processor = DetPostProcessor::new(DetPostProcessorConfig {
            threshold: 0.5,
            min_area: 1.0,
        });

        let contours = processor.process_probability_map(&probability_map).unwrap();
        assert_eq!(contours.len(), 1);

        let area = contour_area(&contours[0]);
        assert!(area >= 1.0, "expected positive area, got {}", area);
    }

    #[test]
    fn filters_small_regions() {
        let probability_map = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];

        let processor = DetPostProcessor::new(DetPostProcessorConfig {
            threshold: 0.5,
            min_area: 5.0,
        });

        let contours = processor.process_probability_map(&probability_map).unwrap();
        assert!(contours.is_empty());
    }

    #[test]
    fn empty_probability_map_is_error() {
        let probability_map = Array2::<f32>::zeros((0, 0));
        let processor = DetPostProcessor::new(DetPostProcessorConfig::default());

        let err = processor
            .process_probability_map(&probability_map)
            .unwrap_err();
        matches!(err, DetPostProcessorError::EmptyProbabilityMap);
    }

    #[test]
    fn unclip_makes_polygon_larger() {
        let contour = Contour::new(
            vec![
                Point::new(0, 0),
                Point::new(4, 0),
                Point::new(4, 4),
                Point::new(0, 4),
            ],
            BorderType::Outer,
            None,
        );

        let unclipper = DetPolygonUnclipper::new(DetPolygonUnclipperConfig {
            unclip_ratio: 2.0,
            min_result_area: 1.0,
            join_style: DetUnclipLineJoin::Round(0.1),
        });

        let unclipped = unclipper.unclip_contours(&[contour]);
        assert!(!unclipped.is_empty());

        let original_area = 16.0;
        let enlarged = unclipped
            .iter()
            .map(|poly| polygon_area(poly) as f64)
            .fold(0.0, f64::max);

        assert!(
            enlarged > original_area,
            "expected unclip area ({}) to exceed original ({})",
            enlarged,
            original_area
        );
    }

    #[test]
    fn scaler_restores_original_coordinates() {
        let polygon = Polygon::new(
            LineString::from(vec![
                Coord { x: 50.0, y: 20.0 },
                Coord { x: 150.0, y: 20.0 },
                Coord { x: 150.0, y: 120.0 },
                Coord { x: 50.0, y: 120.0 },
                Coord { x: 50.0, y: 20.0 },
            ]),
            Vec::new(),
        );

        let scaler = DetPolygonScaler::new(DetPolygonScalerConfig::default());
        let scaled = scaler.scale_polygons(&[polygon], 0.5, (400, 400));

        assert_eq!(scaled.len(), 1);
        let exterior = scaled[0].exterior();
        let expected = vec![
            Coord { x: 100.0, y: 40.0 },
            Coord { x: 300.0, y: 40.0 },
            Coord { x: 300.0, y: 240.0 },
            Coord { x: 100.0, y: 240.0 },
            Coord { x: 100.0, y: 40.0 },
        ];

        for (point, expected) in exterior.points().zip(expected.iter()) {
            assert!((point.x() - expected.x).abs() < 1e-6 && (point.y() - expected.y).abs() < 1e-6);
        }
    }

    #[test]
    fn scaler_clamps_coordinates_when_enabled() {
        let polygon = Polygon::new(
            LineString::from(vec![
                Coord { x: 500.0, y: 500.0 },
                Coord { x: 600.0, y: 500.0 },
                Coord { x: 600.0, y: 600.0 },
                Coord { x: 500.0, y: 600.0 },
                Coord { x: 500.0, y: 500.0 },
            ]),
            Vec::new(),
        );

        let scaler = DetPolygonScaler::new(DetPolygonScalerConfig {
            clamp_to_image: true,
            rounding: DetScaleRounding::None,
        });

        let scaled = scaler.scale_polygons(&[polygon], 1.0, (256, 256));
        let exterior = scaled[0].exterior();

        for point in exterior.points() {
            assert!(
                (0.0..=256.0).contains(&point.x()),
                "expected x within bounds, got {}",
                point.x()
            );
            assert!(
                (0.0..=256.0).contains(&point.y()),
                "expected y within bounds, got {}",
                point.y()
            );
        }
    }
}
