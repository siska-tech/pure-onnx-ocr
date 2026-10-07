//! Prints OCR results as JSON in the same shape as the WebAssembly bindings'
//! `OcrEngine.run` (`[{ text, confidence, box, polygon }, ...]`), so browser
//! output can be compared with the native build (`examples/web/bench.mjs`).
//!
//! ```text
//! cargo run --release --example ocr_json -- --det DIR|FILE.onnx --rec DIR|FILE.onnx
//!     [--dict DICT.txt] [--threads N] IMAGE > native.json
//! ```
//!
//! `DIR` is a PaddleOCR 3.x model directory (`inference.onnx` +
//! `inference.yml`); a bare `.onnx` file is the legacy single-file layout,
//! which needs `--dict` for recognition.

use std::env;
use std::path::{Path, PathBuf};

use pure_onnx_ocr::{min_area_quad, OcrEngineBuilder};
use serde_json::{json, Value};

fn model_files(path: &Path) -> (PathBuf, Option<PathBuf>) {
    if path.is_dir() {
        (
            path.join("inference.onnx"),
            Some(path.join("inference.yml")),
        )
    } else {
        (path.to_path_buf(), None)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut det = None;
    let mut rec = None;
    let mut dict = None;
    let mut threads = None;
    let mut image = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--det" => det = args.next().map(PathBuf::from),
            "--rec" => rec = args.next().map(PathBuf::from),
            "--dict" => dict = args.next().map(PathBuf::from),
            "--threads" => threads = args.next().map(|n| n.parse::<usize>()).transpose()?,
            _ => image = Some(PathBuf::from(arg)),
        }
    }
    let (Some(det), Some(rec), Some(image)) = (det, rec, image) else {
        eprintln!(
            "usage: ocr_json --det DIR|FILE --rec DIR|FILE [--dict FILE] [--threads N] IMAGE"
        );
        std::process::exit(2);
    };

    let (det_model, det_config) = model_files(&det);
    let (rec_model, rec_config) = model_files(&rec);
    let mut builder = OcrEngineBuilder::new()
        .det_model_path(det_model)
        .rec_model_path(rec_model);
    if let Some(config) = det_config {
        builder = builder.det_config_path(config);
    }
    if let Some(config) = rec_config {
        builder = builder.rec_config_path(config);
    }
    if let Some(dict) = dict {
        builder = builder.dictionary_path(dict);
    }
    if let Some(threads) = threads {
        builder = builder.inference_threads(threads);
    }
    let engine = builder.build()?;

    let point = |x: f64, y: f64| json!([x, y]);
    let results: Vec<Value> = engine
        .run_from_path(&image)?
        .iter()
        .map(|result| {
            let mut points: Vec<_> = result.bounding_box.exterior().points().collect();
            if points.len() > 1 && points.first() == points.last() {
                points.pop();
            }
            let quad: Vec<Value> = min_area_quad(&result.bounding_box)
                .map(|corners| corners.iter().map(|&(x, y)| point(x, y)).collect())
                .unwrap_or_default();
            json!({
                "text": result.text,
                "confidence": result.confidence,
                "box": quad,
                "polygon": points.iter().map(|p| point(p.x(), p.y())).collect::<Vec<_>>(),
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
