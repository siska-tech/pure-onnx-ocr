//! Image resampling that matches OpenCV, which PaddleOCR uses.
//!
//! `image`'s `FilterType::Triangle` low-pass filters when downscaling, while
//! `cv2.resize(..., INTER_LINEAR)` samples the four nearest pixels only. The
//! difference is visible in OCR output (spaces and look-alike characters), so
//! the pipeline uses this OpenCV-compatible implementation.

use image::{Rgb, RgbImage};

/// `cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)`:
/// half-pixel-centre bilinear interpolation without anti-aliasing, with
/// border pixels replicated.
pub fn resize_bilinear(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    if width == 0 || height == 0 || src_w == 0 || src_h == 0 {
        return RgbImage::new(width, height);
    }
    if (width, height) == (src_w, src_h) {
        return src.clone();
    }

    let taps = |dst: u32, src_len: u32| -> Vec<(u32, u32, f32)> {
        let scale = src_len as f64 / dst as f64;
        (0..dst)
            .map(|d| {
                let s = (d as f64 + 0.5) * scale - 0.5;
                let mut i0 = s.floor();
                let mut frac = s - i0;
                if i0 < 0.0 {
                    i0 = 0.0;
                    frac = 0.0;
                }
                let last = (src_len - 1) as f64;
                if i0 >= last {
                    i0 = last;
                    frac = 0.0;
                }
                let i0 = i0 as u32;
                let i1 = (i0 + 1).min(src_len - 1);
                (i0, i1, frac as f32)
            })
            .collect()
    };
    let xs = taps(width, src_w);
    let ys = taps(height, src_h);

    let mut out = RgbImage::new(width, height);
    for (dy, &(y0, y1, fy)) in ys.iter().enumerate() {
        for (dx, &(x0, x1, fx)) in xs.iter().enumerate() {
            let p00 = src.get_pixel(x0, y0);
            let p01 = src.get_pixel(x1, y0);
            let p10 = src.get_pixel(x0, y1);
            let p11 = src.get_pixel(x1, y1);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p01[c] as f32 * fx;
                let bottom = p10[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                px[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(dx as u32, dy as u32, Rgb(px));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_upscale_match_opencv_semantics() {
        let src = RgbImage::from_fn(2, 1, |x, _| Rgb([if x == 0 { 0 } else { 200 }; 3]));
        assert_eq!(resize_bilinear(&src, 2, 1), src);
        // cv2.resize([[0, 200]], (4, 1)) == [[0, 50, 150, 200]]
        let up = resize_bilinear(&src, 4, 1);
        let row: Vec<u8> = (0..4).map(|x| up.get_pixel(x, 0)[0]).collect();
        assert_eq!(row, vec![0, 50, 150, 200]);
    }

    #[test]
    fn downscale_samples_without_antialiasing() {
        // cv2.resize([[0, 100, 200, 250]], (2, 1)) == [[50, 225]]
        let values = [0u8, 100, 200, 250];
        let src = RgbImage::from_fn(4, 1, |x, _| Rgb([values[x as usize]; 3]));
        let down = resize_bilinear(&src, 2, 1);
        assert_eq!(down.get_pixel(0, 0)[0], 50);
        assert_eq!(down.get_pixel(1, 0)[0], 225);
    }
}
