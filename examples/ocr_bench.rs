//! Compares PaddleOCR model generations (PP-OCRv5 / PP-OCRv6) on this crate's
//! tract-based CPU runtime.
//!
//! ```text
//! cargo run --release --example ocr_bench -- [--fixtures DIR] [--runs N]
//!     [--models v5-mobile,v5-server,v6-tiny,v6-small,v6-medium]
//!     [--images general_ocr_002.jpg,ja.jpg] [--pipeline-models v5-mobile,v6-tiny,...]
//! ```
//!
//! Expected layout under the fixtures directory (default `tests/fixtures`):
//! `models/ppocrv5/{mobile,server}_{det,rec}/` and
//! `models/ppocrv6/{tiny,small,medium}_{det,rec}/`, each holding
//! `inference.onnx` + `inference.yml`, and the images under `images/`.
//!
//! Three measurements per model:
//! 1. **Model only**: detection on the preprocessed sample image and
//!    recognition on a fixed `[8, 3, 48, 320]` batch, so both generations see
//!    identical inputs regardless of how many regions they detect.
//! 2. **Pipeline**: `OcrEngine::run_with_metrics_from_image` per image. One
//!    warm-up run (which compiles the inference plans) is reported separately
//!    and excluded from the median.
//! 3. **Load**: `OcrEngineBuilder::build` (parsing + graph preparation).

use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use image::{DynamicImage, Rgb, RgbImage};
use pure_onnx_ocr::{
    DetInferenceSession, DetPreProcessor, DetPreProcessorConfig, OcrEngineBuilder,
    RecInferenceSession, RecPreProcessor, RecPreProcessorConfig,
};

struct ModelSpec {
    name: &'static str,
    det: &'static str,
    rec: &'static str,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "v5-mobile",
        det: "ppocrv5/mobile_det",
        rec: "ppocrv5/mobile_rec",
    },
    ModelSpec {
        name: "v5-server",
        det: "ppocrv5/server_det",
        rec: "ppocrv5/server_rec",
    },
    ModelSpec {
        name: "v6-tiny",
        det: "ppocrv6/tiny_det",
        rec: "ppocrv6/tiny_rec",
    },
    ModelSpec {
        name: "v6-small",
        det: "ppocrv6/small_det",
        rec: "ppocrv6/small_rec",
    },
    ModelSpec {
        name: "v6-medium",
        det: "ppocrv6/medium_det",
        rec: "ppocrv6/medium_rec",
    },
];

fn median(values: &mut [Duration]) -> Duration {
    values.sort();
    values[values.len() / 2]
}

fn ms(d: Duration) -> String {
    format!("{:.0}", d.as_secs_f64() * 1000.0)
}

fn file_mb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1e6)
        .unwrap_or(0.0)
}

/// Opts this process out of Windows power throttling (EcoQoS).
///
/// On hybrid CPUs (P-cores + E-cores), Windows may treat a process without a
/// foreground window as background work and run it on efficiency cores at
/// reduced clocks, which made benchmark timings swing by 5-10x.
#[cfg(windows)]
fn disable_power_throttling() -> bool {
    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn SetProcessInformation(
            process: *mut core::ffi::c_void,
            class: i32,
            information: *const core::ffi::c_void,
            size: u32,
        ) -> i32;
    }
    const PROCESS_POWER_THROTTLING: i32 = 4; // ProcessPowerThrottling
    const EXECUTION_SPEED: u32 = 0x1; // PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    let state = ProcessPowerThrottlingState {
        version: 1,
        control_mask: EXECUTION_SPEED,
        state_mask: 0, // control bit set, state bit clear => never throttle
    };
    // SAFETY: plain Win32 call on the current process with a correctly sized,
    // initialised PROCESS_POWER_THROTTLING_STATE.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            &state as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<ProcessPowerThrottlingState>() as u32,
        ) != 0
    }
}

#[cfg(not(windows))]
fn disable_power_throttling() -> bool {
    false
}

fn main() {
    if cfg!(windows) {
        println!(
            "power throttling opt-out: {}",
            if disable_power_throttling() {
                "ok"
            } else {
                "failed"
            }
        );
    }
    let mut fixtures = PathBuf::from("tests/fixtures");
    let mut runs = 5usize;
    let mut models: Vec<String> = MODELS.iter().map(|m| m.name.to_string()).collect();
    let mut images = vec!["general_ocr_002.jpg".to_string(), "ja.jpg".to_string()];
    let mut pipeline_models: Option<Vec<String>> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("missing option value");
        match arg.as_str() {
            "--fixtures" => fixtures = PathBuf::from(value()),
            "--runs" => runs = value().parse().expect("--runs expects a number"),
            "--models" => models = value().split(',').map(str::to_string).collect(),
            "--images" => images = value().split(',').map(str::to_string).collect(),
            "--pipeline-models" => {
                pipeline_models = Some(value().split(',').map(str::to_string).collect())
            }
            other => panic!("unknown option {other}"),
        }
    }
    let runs = runs.max(1);

    // Fixed recognition batch: 8 synthetic 320x48 text-like crops.
    let crops: Vec<RgbImage> = (0..8)
        .map(|i| {
            RgbImage::from_fn(320, 48, |x, y| {
                let on = (x / 6 + y / 8 + i) % 3 == 0 && (8..40).contains(&y);
                if on {
                    Rgb([20, 20, 20])
                } else {
                    Rgb([240, 240, 240])
                }
            })
        })
        .collect();
    let rec_batch = RecPreProcessor::new(RecPreProcessorConfig::default())
        .process_images(&crops)
        .expect("synthetic crops should preprocess");

    let loaded_images: Vec<(String, DynamicImage)> = images
        .iter()
        .map(|name| {
            let path = fixtures.join("images").join(name);
            (
                name.clone(),
                image::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
            )
        })
        .collect();
    let det_input = DetPreProcessor::new(DetPreProcessorConfig::default())
        .process(&loaded_images[0].1)
        .expect("sample image should preprocess");

    println!(
        "fixed inputs: detection {:?}, recognition {:?}",
        det_input.tensor.shape(),
        rec_batch.tensor.shape()
    );
    println!(
        "{runs} measured rounds; models are interleaved round-robin inside each round so that \
         CPU frequency drift (thermal throttling) affects every model alike. Median reported."
    );

    let specs: Vec<&ModelSpec> = models
        .iter()
        .map(|name| {
            MODELS
                .iter()
                .find(|m| m.name == name)
                .unwrap_or_else(|| panic!("unknown model {name}"))
        })
        .filter(|spec| {
            let ok = fixtures
                .join("models")
                .join(spec.det)
                .join("inference.onnx")
                .exists()
                && fixtures
                    .join("models")
                    .join(spec.rec)
                    .join("inference.onnx")
                    .exists();
            if !ok {
                eprintln!("skipping {}: model files missing", spec.name);
            }
            ok
        })
        .collect();
    let dirs = |spec: &ModelSpec| {
        (
            fixtures.join("models").join(spec.det),
            fixtures.join("models").join(spec.rec),
        )
    };

    // ---- 1. Model-only, identical inputs -------------------------------
    let sessions: Vec<(DetInferenceSession, RecInferenceSession)> = specs
        .iter()
        .map(|spec| {
            let (det_dir, rec_dir) = dirs(spec);
            let det = DetInferenceSession::load(det_dir.join("inference.onnx")).unwrap();
            let rec = RecInferenceSession::load(rec_dir.join("inference.onnx")).unwrap();
            // Warm-up: compiles the plans for these shapes.
            det.run(&det_input).unwrap();
            rec.run(&rec_batch).unwrap();
            (det, rec)
        })
        .collect();
    let mut det_times = vec![Vec::new(); specs.len()];
    let mut rec_times = vec![Vec::new(); specs.len()];
    for _ in 0..runs {
        for (i, (det, rec)) in sessions.iter().enumerate() {
            let start = Instant::now();
            det.run(&det_input).unwrap();
            det_times[i].push(start.elapsed());
            let start = Instant::now();
            rec.run(&rec_batch).unwrap();
            rec_times[i].push(start.elapsed());
        }
    }
    drop(sessions);

    println!("\n| model | det MB | rec MB | load ms | det-only ms | rec-only ms (8x320) |");
    println!("| :--- | ---: | ---: | ---: | ---: | ---: |");
    let mut engines = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let (det_dir, rec_dir) = dirs(spec);
        let load_start = Instant::now();
        let engine = OcrEngineBuilder::new()
            .det_model_dir(&det_dir)
            .rec_model_dir(&rec_dir)
            .build()
            .unwrap();
        let load = load_start.elapsed();
        println!(
            "| {} | {:.1} | {:.1} | {} | {} | {} |",
            spec.name,
            file_mb(&det_dir.join("inference.onnx")),
            file_mb(&rec_dir.join("inference.onnx")),
            ms(load),
            ms(median(&mut det_times[i])),
            ms(median(&mut rec_times[i])),
        );
        if pipeline_models
            .as_ref()
            .map_or(true, |list| list.iter().any(|m| m == spec.name))
        {
            engines.push((spec.name, engine));
        }
    }

    // ---- 2. Full pipeline on real images --------------------------------
    // Warm-up run per (model, image): compiles plans, reported as "first run".
    let mut rows: Vec<(
        &str,
        &str,
        usize,
        Duration,
        Vec<Duration>,
        Vec<Duration>,
        Vec<Duration>,
    )> = Vec::new();
    for (name, engine) in &engines {
        for (image_name, image) in &loaded_images {
            let first = engine.run_with_metrics_from_image(image).unwrap();
            rows.push((
                name,
                image_name.as_str(),
                first.results.len(),
                first.timings.total,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
        }
    }
    for _ in 0..runs {
        let mut row = 0;
        for (_, engine) in &engines {
            for (_, image) in &loaded_images {
                let run = engine.run_with_metrics_from_image(image).unwrap();
                rows[row].4.push(run.timings.total);
                rows[row].5.push(run.timings.detection.inference);
                rows[row].6.push(run.timings.recognition.inference);
                row += 1;
            }
        }
    }

    println!("\n| model | image | regions | first run ms | total ms | det ms | rec ms |");
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |");
    for (name, image, regions, first, mut totals, mut dets, mut recs) in rows {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            name,
            image,
            regions,
            ms(first),
            ms(median(&mut totals)),
            ms(median(&mut dets)),
            ms(median(&mut recs)),
        );
    }
}
