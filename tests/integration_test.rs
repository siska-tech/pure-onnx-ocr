use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use once_cell::sync::Lazy;
use pure_onnx_ocr::{OcrEngine, OcrEngineBuilder, OcrError};

#[derive(Debug)]
struct FixturePaths {
    det_model: PathBuf,
    rec_model: PathBuf,
    dictionary: PathBuf,
    image: PathBuf,
}

static FIXTURES: OnceLock<Option<FixturePaths>> = OnceLock::new();
static ENGINE_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn locate_fixture_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("PURE_ONNX_OCR_FIXTURE_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

fn fixtures() -> Option<&'static FixturePaths> {
    FIXTURES
        .get_or_init(|| {
            let base = locate_fixture_dir()?;
            let det_model = base.join("models/ppocrv5/det.onnx");
            let rec_model = base.join("models/ppocrv5/rec.onnx");
            let dictionary = base.join("models/ppocrv5/ppocrv5_dict.txt");
            // Downloaded by scripts/fetch_fixtures.sh.
            let image = base.join("images/general_ocr_002.jpg");

            if det_model.exists() && rec_model.exists() && dictionary.exists() && image.exists() {
                Some(FixturePaths {
                    det_model,
                    rec_model,
                    dictionary,
                    image,
                })
            } else {
                None
            }
        })
        .as_ref()
}

fn build_engine(fixtures: &FixturePaths) -> Result<OcrEngine, OcrError> {
    OcrEngineBuilder::new()
        .det_model_path(&fixtures.det_model)
        .rec_model_path(&fixtures.rec_model)
        .dictionary_path(&fixtures.dictionary)
        .build()
}

#[test]
fn ocr_pipeline_smoke_test() -> Result<(), OcrError> {
    let Some(fixtures) = fixtures() else {
        eprintln!("Skipping OCR pipeline smoke test: integration fixtures not present.");
        return Ok(());
    };

    let _guard = ENGINE_MUTEX.lock().expect("integration mutex poisoned");
    let engine = build_engine(fixtures)?;
    let results = engine.run_from_path(&fixtures.image)?;
    assert!(
        !results.is_empty(),
        "expected OCR pipeline to detect at least one text region"
    );
    let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("BOARDING")),
        "expected PP-OCRv5 to read the boarding pass header, got {texts:?}"
    );

    Ok(())
}

#[test]
fn ocr_pipeline_reports_missing_image() {
    let Some(fixtures) = fixtures() else {
        eprintln!("Skipping OCR missing image test: integration fixtures not present.");
        return;
    };

    let _guard = ENGINE_MUTEX.lock().expect("integration mutex poisoned");
    let engine = build_engine(fixtures).expect("engine should initialise with fixtures");
    let err = engine
        .run_from_path("tests/fixtures/images/does_not_exist.png")
        .expect_err("missing image should produce an error");

    match err {
        OcrError::Io { .. } | OcrError::ImageDecode { .. } => { /* expected */ }
        other => panic!("expected IO or image decode error, got {other:?}"),
    }
}

#[test]
fn ocr_builder_rejects_missing_models() {
    let result = OcrEngineBuilder::new()
        .det_model_path("tests/fixtures/models/missing_det.onnx")
        .rec_model_path("tests/fixtures/models/missing_rec.onnx")
        .dictionary_path("tests/fixtures/models/missing_dict.txt")
        .build();

    assert!(
        result.is_err(),
        "builder should return error when models are unavailable"
    );
}
