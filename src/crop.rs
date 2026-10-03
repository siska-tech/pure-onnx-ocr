//! Rotation-aware cropping of detected text regions.
//!
//! Mirrors PaddleOCR's `get_mini_boxes` + `get_rotate_crop_image`: every
//! detected polygon is reduced to its minimum-area bounding rectangle, whose
//! corners are ordered top-left, top-right, bottom-right, bottom-left. The
//! rectangle is then perspective-warped into an upright crop. Crops that are
//! much taller than wide (vertical text) are rotated by 90 degrees
//! counter-clockwise so the recognition model sees horizontal text.

use geo_types::Polygon;
use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};

/// A quadrilateral given as `[top-left, top-right, bottom-right, bottom-left]`.
pub type Quad = [(f64, f64); 4];

/// Height / width ratio above which a crop is treated as vertical text and
/// rotated (PaddleOCR uses 1.5).
pub const VERTICAL_TEXT_RATIO: f64 = 1.5;

/// Strategy used to cut text regions out of the input image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecCropMode {
    /// Minimum-area rotated rectangle + perspective warp, with vertical crops
    /// rotated to horizontal. Matches PaddleOCR.
    #[default]
    Rotated,
    /// Axis-aligned bounding box of the polygon (the behaviour before
    /// PP-OCRv6 support). Faster, but loses accuracy on tilted text.
    AxisAligned,
}

/// Returns the minimum-area rectangle enclosing `polygon`'s exterior, with
/// corners ordered like PaddleOCR (`tl, tr, br, bl`). Returns `None` for an
/// empty polygon.
pub fn min_area_quad(polygon: &Polygon<f64>) -> Option<Quad> {
    let points: Vec<(f64, f64)> = polygon
        .exterior()
        .points()
        .map(|p| (p.x(), p.y()))
        .collect();
    min_area_quad_from_points(&points)
}

/// Same as [`min_area_quad`] for a raw point list.
pub fn min_area_quad_from_points(points: &[(f64, f64)]) -> Option<Quad> {
    if points.is_empty() {
        return None;
    }
    let hull = convex_hull(points);
    let corners = if hull.len() < 3 {
        axis_aligned_corners(points)
    } else {
        rotated_rect_corners(&hull)
    };
    Some(order_corners(corners))
}

/// Cuts `quad` out of `image` and warps it into an upright rectangle.
///
/// The output size is the longer of each pair of opposite edges, like
/// PaddleOCR. Vertical crops (height / width >= [`VERTICAL_TEXT_RATIO`]) are
/// rotated 90 degrees counter-clockwise. Returns `None` when the quad is
/// degenerate.
pub fn crop_quad(image: &RgbImage, quad: &Quad) -> Option<RgbImage> {
    let [tl, tr, br, bl] = *quad;
    // PaddleOCR truncates the crop size (`int(max(norm(...)))`).
    let width = distance(tl, tr).max(distance(bl, br)).trunc().max(1.0) as u32;
    let height = distance(tl, bl).max(distance(tr, br)).trunc().max(1.0) as u32;

    let from = [tl, tr, br, bl].map(|(x, y)| (x as f32, y as f32));
    let to = [
        (0.0, 0.0),
        (width as f32, 0.0),
        (width as f32, height as f32),
        (0.0, height as f32),
    ];
    let projection = Projection::from_control_points(from, to)?;

    let mut out = RgbImage::new(width, height);
    warp_into(
        image,
        &projection,
        // cv2.warpPerspective(..., flags=cv2.INTER_CUBIC) in PaddleOCR.
        Interpolation::Bicubic,
        Rgb([0, 0, 0]),
        &mut out,
    );

    if height as f64 / width as f64 >= VERTICAL_TEXT_RATIO {
        // np.rot90 rotates counter-clockwise, i.e. 270 degrees clockwise.
        out = imageops::rotate270(&out);
    }
    Some(out)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Andrew's monotone chain convex hull (counter-clockwise, no duplicates).
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut sorted: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn axis_aligned_corners(points: &[(f64, f64)]) -> [(f64, f64); 4] {
    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
}

/// Rotating calipers over the hull edges: the minimum-area rectangle has one
/// side collinear with a hull edge.
fn rotated_rect_corners(hull: &[(f64, f64)]) -> [(f64, f64); 4] {
    let mut best_area = f64::INFINITY;
    let mut best = axis_aligned_corners(hull);
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let length = distance(a, b);
        if length <= f64::EPSILON {
            continue;
        }
        let u = ((b.0 - a.0) / length, (b.1 - a.1) / length);
        let v = (-u.1, u.0);

        let (mut min_u, mut max_u, mut min_v, mut max_v) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for &p in hull {
            let du = (p.0 - a.0) * u.0 + (p.1 - a.1) * u.1;
            let dv = (p.0 - a.0) * v.0 + (p.1 - a.1) * v.1;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_v = min_v.min(dv);
            max_v = max_v.max(dv);
        }

        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let at = |su: f64, sv: f64| (a.0 + u.0 * su + v.0 * sv, a.1 + u.1 * su + v.1 * sv);
            best = [
                at(min_u, min_v),
                at(max_u, min_v),
                at(max_u, max_v),
                at(min_u, max_v),
            ];
        }
    }
    best
}

/// Orders corners as PaddleOCR's `get_mini_boxes`: sort by x, then pick the
/// upper/lower point of the left pair and of the right pair.
fn order_corners(mut corners: [(f64, f64); 4]) -> Quad {
    corners.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (tl, bl) = if corners[1].1 > corners[0].1 {
        (corners[0], corners[1])
    } else {
        (corners[1], corners[0])
    };
    let (tr, br) = if corners[3].1 > corners[2].1 {
        (corners[2], corners[3])
    } else {
        (corners[3], corners[2])
    };
    [tl, tr, br, bl]
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString};

    fn polygon(points: &[(f64, f64)]) -> Polygon<f64> {
        let coords: Vec<Coord<f64>> = points.iter().map(|&(x, y)| Coord { x, y }).collect();
        Polygon::new(LineString::from(coords), vec![])
    }

    fn approx(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn axis_aligned_rectangle_is_ordered_tl_tr_br_bl() {
        let quad = min_area_quad(&polygon(&[
            (10.0, 20.0),
            (50.0, 20.0),
            (50.0, 30.0),
            (10.0, 30.0),
        ]))
        .unwrap();
        assert!(approx(quad[0], (10.0, 20.0)));
        assert!(approx(quad[1], (50.0, 20.0)));
        assert!(approx(quad[2], (50.0, 30.0)));
        assert!(approx(quad[3], (10.0, 30.0)));
    }

    #[test]
    fn rotated_rectangle_is_recovered() {
        // 40 x 10 rectangle rotated by 30 degrees around (100, 100), plus an
        // interior point that must not change the result.
        let (s, c) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
        let rot = |x: f64, y: f64| (100.0 + x * c - y * s, 100.0 + x * s + y * c);
        let corners = [
            rot(-20.0, -5.0),
            rot(20.0, -5.0),
            rot(20.0, 5.0),
            rot(-20.0, 5.0),
        ];
        let mut points = corners.to_vec();
        points.push((100.0, 100.0));

        let quad = min_area_quad_from_points(&points).unwrap();
        let width = distance(quad[0], quad[1]);
        let height = distance(quad[0], quad[3]);
        assert!((width - 40.0).abs() < 1e-6, "width {}", width);
        assert!((height - 10.0).abs() < 1e-6, "height {}", height);
        for corner in corners {
            assert!(quad.iter().any(|q| approx(*q, corner)));
        }
    }

    #[test]
    fn degenerate_points_fall_back_to_bounding_box() {
        let quad = min_area_quad_from_points(&[(1.0, 1.0), (5.0, 1.0)]).unwrap();
        assert!(approx(quad[0], (1.0, 1.0)));
        assert!(approx(quad[1], (5.0, 1.0)));
        assert!(min_area_quad_from_points(&[]).is_none());
    }

    #[test]
    fn crop_quad_straightens_region() {
        // Left half black, right half white.
        let image = RgbImage::from_fn(100, 40, |x, _| {
            if x < 50 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        });
        let quad = [(20.0, 10.0), (80.0, 10.0), (80.0, 30.0), (20.0, 30.0)];
        let crop = crop_quad(&image, &quad).unwrap();
        assert_eq!(crop.dimensions(), (60, 20));
        assert!(crop.get_pixel(5, 10)[0] < 10);
        assert!(crop.get_pixel(55, 10)[0] > 245);
    }

    #[test]
    fn vertical_crop_is_rotated_counter_clockwise() {
        // Top part red, bottom part blue; a 10 x 40 vertical region.
        let image = RgbImage::from_fn(30, 60, |_, y| {
            if y < 30 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        });
        let quad = [(10.0, 10.0), (20.0, 10.0), (20.0, 50.0), (10.0, 50.0)];
        let crop = crop_quad(&image, &quad).unwrap();
        assert_eq!(crop.dimensions(), (40, 10));
        // After a counter-clockwise rotation the former top is on the left.
        assert!(crop.get_pixel(2, 5)[0] > 200);
        assert!(crop.get_pixel(37, 5)[2] > 200);
    }
}
