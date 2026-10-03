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
        joined.contains("GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME"),
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
