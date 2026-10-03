//! PP-OCRv6 integration tests.
//!
//! The tests expect the Hugging Face ONNX exports to be laid out as
//!
//! ```text
//! fixtures/models/ppocrv6/{tiny,small,medium}_{det,rec}/inference.{onnx,yml}
//! fixtures/images/general_ocr_002.jpg
//! ```
//!
//! under `tests/fixtures` or `PURE_ONNX_OCR_FIXTURE_DIR`. See
//! `tests/fixtures/README.md` for download instructions. Tests skip with a
//! message when the assets are missing.

use std::env;
use std::path::{Path, PathBuf};

use pure_onnx_ocr::{
    ColorOrder, OcrEngineBuilder, PaddleInferenceConfig, RecDictionary, RecInferenceSession,
    RecPreProcessor, RecPreProcessorConfig, RecTextRegion,
};

const SAMPLE_IMAGE: &str = "images/general_ocr_002.jpg";

fn fixture_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("PURE_ONNX_OCR_FIXTURE_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }
    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    default.exists().then_some(default)
}

fn model_dir(tier: &str, kind: &str) -> Option<PathBuf> {
    let dir = fixture_dir()?
        .join("models")
        .join("ppocrv6")
        .join(format!("{}_{}", tier, kind));
    if dir.join("inference.onnx").exists() && dir.join("inference.yml").exists() {
        Some(dir)
    } else {
        eprintln!("PP-OCRv6 {} {} fixtures not found; skipping", tier, kind);
        None
    }
}

fn sample_image() -> Option<PathBuf> {
    let path = fixture_dir()?.join(SAMPLE_IMAGE);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("{} not found; skipping", SAMPLE_IMAGE);
        None
    }
}

#[test]
fn detection_configs_use_bgr_imagenet_normalisation() {
    for tier in ["tiny", "small", "medium"] {
        let Some(dir) = model_dir(tier, "det") else {
            continue;
        };
        let config = PaddleInferenceConfig::from_path(dir.join("inference.yml")).unwrap();
        assert_eq!(
            config.model_name.as_deref(),
            Some(format!("PP-OCRv6_{}_det", tier).as_str())
        );
        assert_eq!(config.post_process_name.as_deref(), Some("DBPostProcess"));
        assert_eq!(config.color_order, Some(ColorOrder::Bgr));
        assert_eq!(config.normalize_mean, Some([0.485, 0.456, 0.406]));
        assert_eq!(config.normalize_std, Some([0.229, 0.224, 0.225]));
    }
}

#[test]
fn recognition_configs_embed_dictionaries() {
    // Character counts of the dictionaries embedded in the published exports.
    let expected = [("tiny", 6904usize), ("small", 18708), ("medium", 18708)];
    for (tier, count) in expected {
        let Some(dir) = model_dir(tier, "rec") else {
            continue;
        };
        let config = PaddleInferenceConfig::from_path(dir.join("inference.yml")).unwrap();
        assert_eq!(config.post_process_name.as_deref(), Some("CTCLabelDecode"));
        assert_eq!(config.rec_image_shape, Some([3, 48, 320]));
        assert_eq!(config.color_order, Some(ColorOrder::Bgr));
        assert_eq!(config.character_dict.as_ref().map(Vec::len), Some(count));

        let dictionary = RecDictionary::from_path(dir.join("inference.yml"))
            .unwrap()
            .with_space_char();
        // blank + characters + space
        assert_eq!(dictionary.len(), count + 2);
    }
}

#[test]
fn recognition_class_count_matches_dictionary() {
    let Some(dir) = model_dir("tiny", "rec") else {
        return;
    };
    let Some(image_path) = sample_image() else {
        return;
    };

    let dictionary = RecDictionary::from_path(dir.join("inference.yml"))
        .unwrap()
        .with_space_char();
    let session = RecInferenceSession::load(dir.join("inference.onnx")).unwrap();
    let image = image::open(image_path).unwrap();
    let batch = RecPreProcessor::new(RecPreProcessorConfig::default())
        .process(
            &image,
            &[RecTextRegion {
                x: 0,
                y: 0,
                width: 200,
                height: 48,
            }],
        )
        .unwrap();
    let output = session.run(&batch).unwrap();

    assert_eq!(output.logits.dim().2, dictionary.len());
}

fn run_pipeline(tier: &str) -> Option<Vec<String>> {
    let det = model_dir(tier, "det")?;
    let rec = model_dir(tier, "rec")?;
    let image = sample_image()?;

    let engine = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .build()
        .expect("PP-OCRv6 engine should build");
    let results = engine.run_from_path(&image).expect("OCR should succeed");
    Some(results.into_iter().map(|result| result.text).collect())
}

fn assert_boarding_pass(texts: &[String]) {
    let joined = texts.join("\n");
    for expected in [
        "BOARDING",
        "ZHANGQIWEI",
        "TAIYUAN",
        "FUZHOU",
        "登机牌",
        "张祺伟",
    ] {
        assert!(
            joined.contains(expected),
            "expected `{}` in OCR output:\n{}",
            expected,
            joined
        );
    }
    // Spaces are produced by the space class appended to the dictionary.
    assert!(
        joined.contains("GATES CLOSE") && joined.contains("MINUTES BEFORE DEPARTURE TIME"),
        "expected the long footer line with spaces in OCR output:\n{}",
        joined
    );
    assert!(
        !joined.contains("[UNK]"),
        "unexpected [UNK] in:\n{}",
        joined
    );
}

#[test]
fn tiny_pipeline_reads_boarding_pass() {
    if let Some(texts) = run_pipeline("tiny") {
        assert_boarding_pass(&texts);
    }
}

#[test]
#[ignore = "PP-OCRv6 small takes ~7 s on CPU; run with `cargo test --release -- --ignored`"]
fn small_pipeline_reads_boarding_pass() {
    if let Some(texts) = run_pipeline("small") {
        assert_boarding_pass(&texts);
    }
}

#[test]
#[ignore = "PP-OCRv6 medium takes ~25 s on CPU; run with `cargo test --release -- --ignored`"]
fn medium_pipeline_reads_boarding_pass() {
    if let Some(texts) = run_pipeline("medium") {
        assert_boarding_pass(&texts);
    }
}

#[test]
fn ppocrv5_yaml_dictionary_matches_text_dictionary() {
    // PP-OCRv5 ships both forms; they must agree so either can be used.
    let Some(base) = fixture_dir() else {
        return;
    };
    let dir = base.join("models").join("ppocrv5");
    let (yml, txt) = (dir.join("rec.yml"), dir.join("ppocrv5_dict.txt"));
    if !yml.exists() || !txt.exists() {
        eprintln!("PP-OCRv5 rec.yml / ppocrv5_dict.txt not found; skipping");
        return;
    }
    let from_yml = RecDictionary::from_path(&yml).unwrap();
    let from_txt = RecDictionary::from_path(&txt).unwrap();
    assert_eq!(from_yml.len(), from_txt.len());
    for index in 0..from_yml.len() {
        assert_eq!(
            from_yml.token(index),
            from_txt.token(index),
            "index {}",
            index
        );
    }
}

#[test]
fn model_config_postprocess_values_respect_explicit_overrides() {
    let (Some(det), Some(rec)) = (model_dir("tiny", "det"), model_dir("tiny", "rec")) else {
        return;
    };

    // Default: PaddleOCR pipeline values, YAML PostProcess ignored.
    let engine = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .build()
        .unwrap();
    assert_eq!(engine.config().det_postprocessor.threshold, 0.3);
    assert_eq!(engine.config().det_postprocessor.box_threshold, 0.6);
    assert_eq!(engine.config().det_unclipper.unclip_ratio, 1.5);

    // Opt-in: tiny_det inference.yml uses 0.2 / 0.4 / 1.4 / 3000, while an
    // explicit setter still wins.
    let engine = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .det_postprocess_from_model_config(true)
        .det_box_threshold(0.5)
        .build()
        .unwrap();
    let config = engine.config();
    assert_eq!(config.det_postprocessor.threshold, 0.2);
    assert_eq!(config.det_postprocessor.box_threshold, 0.5);
    assert!((config.det_unclipper.unclip_ratio - 1.4).abs() < 1e-6);
    assert_eq!(config.det_postprocessor.max_candidates, 3000);
}

fn read_boarding_pass(
    tier: &str,
    mode: pure_onnx_ocr::RecCropMode,
    degrees: f32,
) -> Option<Vec<String>> {
    use imageproc::geometric_transformations::{rotate_about_center, Interpolation};

    let det = model_dir(tier, "det")?;
    let rec = model_dir(tier, "rec")?;
    let image = image::open(sample_image()?).ok()?.to_rgb8();
    let image = if degrees == 0.0 {
        image
    } else {
        rotate_about_center(
            &image,
            degrees.to_radians(),
            Interpolation::Bilinear,
            image::Rgb([255, 255, 255]),
        )
    };
    let engine = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .rec_crop_mode(mode)
        .build()
        .unwrap();
    let results = engine
        .run_from_image(&image::DynamicImage::ImageRgb8(image))
        .unwrap();
    Some(results.into_iter().map(|r| r.text).collect())
}

#[test]
#[ignore = "runs PP-OCRv6 small three times (~30 s); run with `cargo test --release -- --ignored`"]
fn rotated_crops_read_tilted_text() {
    use pure_onnx_ocr::RecCropMode;

    let Some(reference) = read_boarding_pass("small", RecCropMode::Rotated, 0.0) else {
        return;
    };
    let rotated = read_boarding_pass("small", RecCropMode::Rotated, 10.0).unwrap();
    let axis = read_boarding_pass("small", RecCropMode::AxisAligned, 10.0).unwrap();

    // Number of lines read on the upright image that are reproduced exactly
    // on the 10 degree tilted image.
    let matches = |lines: &[String]| reference.iter().filter(|r| lines.contains(r)).count();
    let (rotated_hits, axis_hits) = (matches(&rotated), matches(&axis));
    eprintln!(
        "reference lines {} / rotated matches {} / axis-aligned matches {}",
        reference.len(),
        rotated_hits,
        axis_hits
    );

    // The axis-aligned crop of the long footer line includes neighbouring
    // rows on a tilted image and is misread; the rotated crop is not.
    assert!(
        rotated
            .iter()
            .any(|l| l.contains("GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME")),
        "rotated output:
{}",
        rotated.join(
            "
"
        )
    );
    assert!(rotated_hits > axis_hits);
}

fn classifier_dir(name: &str) -> Option<PathBuf> {
    let dir = fixture_dir()?.join("models").join(name);
    if dir.join("inference.onnx").exists() && dir.join("inference.yml").exists() {
        Some(dir)
    } else {
        eprintln!("{} fixtures not found; skipping", name);
        None
    }
}

#[test]
fn doc_orientation_restores_rotated_pages() {
    let (Some(det), Some(rec), Some(doc_ori), Some(image_path)) = (
        model_dir("tiny", "det"),
        model_dir("tiny", "rec"),
        classifier_dir("PP-LCNet_x1_0_doc_ori"),
        sample_image(),
    ) else {
        return;
    };
    let engine = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .doc_orientation_model_dir(&doc_ori)
        .build()
        .unwrap();
    let upright = image::open(image_path).unwrap().to_rgb8();

    // Rotating the page clockwise by `cw` degrees must be detected as needing
    // a counter-clockwise rotation of the same angle.
    for (cw, rotated) in [
        (90u32, image::imageops::rotate90(&upright)),
        (180, image::imageops::rotate180(&upright)),
        (270, image::imageops::rotate270(&upright)),
    ] {
        let run = engine
            .run_with_metrics_from_image(&image::DynamicImage::ImageRgb8(rotated.clone()))
            .unwrap();
        assert_eq!(
            run.doc_orientation_angle,
            Some(cw),
            "page rotated {} cw",
            cw
        );
        let texts: Vec<&str> = run.results.iter().map(|r| r.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("ZHANGQIWEI")),
            "rotated {} cw: {:?}",
            cw,
            texts
        );
        // Polygons are reported in the coordinates of the rotated input.
        let (rw, rh) = rotated.dimensions();
        for result in &run.results {
            for point in result.bounding_box.exterior().points() {
                assert!(point.x() >= -1.0 && point.x() <= rw as f64 + 1.0);
                assert!(point.y() >= -1.0 && point.y() <= rh as f64 + 1.0);
            }
        }
    }
}

#[test]
fn textline_orientation_fixes_upside_down_lines() {
    let (Some(det), Some(rec), Some(textline_ori), Some(image_path)) = (
        model_dir("tiny", "det"),
        model_dir("tiny", "rec"),
        classifier_dir("PP-LCNet_x1_0_textline_ori"),
        sample_image(),
    ) else {
        return;
    };
    let flipped = image::DynamicImage::ImageRgb8(image::imageops::rotate180(
        &image::open(image_path).unwrap().to_rgb8(),
    ));
    let read = |with_classifier: bool| {
        let mut builder = OcrEngineBuilder::new()
            .det_model_dir(&det)
            .rec_model_dir(&rec);
        if with_classifier {
            builder = builder.textline_orientation_model_dir(&textline_ori);
        }
        builder
            .build()
            .unwrap()
            .run_from_image(&flipped)
            .unwrap()
            .into_iter()
            .map(|r| r.text)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let without = read(false);
    let with = read(true);
    eprintln!("--- without ---\n{}\n--- with ---\n{}", without, with);
    // Short all-uppercase crops such as `TAIYUAN` are nearly point-symmetric
    // and are not always recognised as upside down, so check mixed lines.
    for expected in ["BOARDING", "ZHANGQIWEI", "张祺伟", "登机牌", "GATES CLOSE"] {
        assert!(
            with.contains(expected),
            "expected `{}` in:\n{}",
            expected,
            with
        );
        assert!(!without.contains(expected));
    }
}

#[test]
fn in_memory_models_match_file_based_engine() {
    let (Some(det), Some(rec), Some(image_path)) = (
        model_dir("tiny", "det"),
        model_dir("tiny", "rec"),
        sample_image(),
    ) else {
        return;
    };
    let read = |p: PathBuf| std::fs::read(p).unwrap();
    let read_text = |p: PathBuf| std::fs::read_to_string(p).unwrap();

    let from_files = OcrEngineBuilder::new()
        .det_model_dir(&det)
        .rec_model_dir(&rec)
        .build()
        .unwrap();
    let from_memory = OcrEngineBuilder::new()
        .det_model_bytes(read(det.join("inference.onnx")))
        .det_config_yaml(read_text(det.join("inference.yml")))
        .rec_model_bytes(read(rec.join("inference.onnx")))
        .rec_config_yaml(read_text(rec.join("inference.yml")))
        .build()
        .unwrap();
    assert!(from_memory.det_model_path().is_none());
    assert!(from_memory.rec_model_path().is_none());
    assert!(from_memory.dictionary_path().is_none());

    let expected: Vec<String> = from_files
        .run_from_path(&image_path)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    let actual: Vec<String> = from_memory
        .run_from_bytes(&read(image_path))
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn engine_can_be_shared_between_threads() {
    let (Some(det), Some(rec), Some(image_path)) = (
        model_dir("tiny", "det"),
        model_dir("tiny", "rec"),
        sample_image(),
    ) else {
        return;
    };
    let engine = std::sync::Arc::new(
        OcrEngineBuilder::new()
            .det_model_dir(&det)
            .rec_model_dir(&rec)
            .build()
            .unwrap(),
    );
    let image = std::sync::Arc::new(image::open(image_path).unwrap());
    let expected: Vec<String> = engine
        .run_from_image(&image)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();

    let handles: Vec<_> = (0..3)
        .map(|_| {
            let engine = std::sync::Arc::clone(&engine);
            let image = std::sync::Arc::clone(&image);
            std::thread::spawn(move || {
                engine
                    .run_from_image(&image)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.text)
                    .collect::<Vec<String>>()
            })
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn thread_count_does_not_change_results() {
    let (Some(det), Some(rec), Some(image_path)) =
        (model_dir("tiny", "det"), model_dir("tiny", "rec"), sample_image())
    else {
        return;
    };
    let read = |threads: usize| {
        OcrEngineBuilder::new()
            .det_model_dir(&det)
            .rec_model_dir(&rec)
            .inference_threads(threads)
            .build()
            .unwrap()
            .run_from_path(&image_path)
            .unwrap()
            .into_iter()
            .map(|r| (r.text, (r.confidence * 1e4).round() as i64))
            .collect::<Vec<_>>()
    };
    let single = read(1);
    assert!(!single.is_empty());
    assert_eq!(read(4), single);
}
