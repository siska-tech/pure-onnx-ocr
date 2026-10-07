//! Throughput of `OcrEngine::run_many_from_images` against a `run_from_image`
//! loop over the same images.
//!
//! ```text
//! cargo run --release --example throughput_bench -- [--fixtures DIR]
//!     [--model v6-small] [--images general_ocr_002.jpg,ja.jpg] [--repeat 8]
//!     [--rounds 3] [--threads N] [--rec-batch-size N]
//! ```
//!
//! The image list is repeated `--repeat` times. After one warm-up pass (which
//! compiles the plans), each round times both modes back to back, alternating
//! which goes first, and checks that they return identical results. Medians
//! over the rounds are reported.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[path = "common/power.rs"]
mod power;

use image::DynamicImage;
use pure_onnx_ocr::{OcrEngineBuilder, OcrResult};

fn main() {
    if cfg!(windows) {
        println!(
            "power throttling opt-out: {}",
            if power::disable_power_throttling() {
                "ok"
            } else {
                "failed"
            }
        );
    }
    let mut fixtures = PathBuf::from("tests/fixtures");
    let mut model = "v6-small".to_string();
    let mut images = vec!["general_ocr_002.jpg".to_string(), "ja.jpg".to_string()];
    let mut repeat = 8usize;
    let mut rounds = 3usize;
    let mut threads: Option<usize> = None;
    let mut rec_batch_size: Option<usize> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("missing option value");
        match arg.as_str() {
            "--fixtures" => fixtures = PathBuf::from(value()),
            "--model" => model = value(),
            "--images" => images = value().split(',').map(str::to_string).collect(),
            "--repeat" => repeat = value().parse().expect("number"),
            "--rounds" => rounds = value().parse().expect("number"),
            "--threads" => threads = Some(value().parse().expect("number")),
            "--rec-batch-size" => rec_batch_size = Some(value().parse().expect("number")),
            other => panic!("unknown option {other}"),
        }
    }

    let (generation, tier) = model.split_once('-').expect("model like v6-small");
    let base = fixtures.join("models").join(match generation {
        "v5" => "ppocrv5",
        "v6" => "ppocrv6",
        other => panic!("unknown generation {other}"),
    });
    let mut builder = OcrEngineBuilder::new()
        .det_model_dir(base.join(format!("{tier}_det")))
        .rec_model_dir(base.join(format!("{tier}_rec")));
    if let Some(threads) = threads {
        builder = builder.inference_threads(threads);
    }
    if let Some(size) = rec_batch_size {
        builder = builder.rec_batch_size(size);
    }
    let engine = builder.build().expect("engine should build");

    let loaded: Vec<DynamicImage> = images
        .iter()
        .map(|name| {
            let path = fixtures.join("images").join(name);
            image::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"))
        })
        .collect();
    let batch: Vec<DynamicImage> = (0..repeat).flat_map(|_| loaded.clone()).collect();
    println!(
        "{model}: {} images ({} x {repeat}), {} inference threads, rec batch {}",
        batch.len(),
        loaded.len(),
        engine.config().inference_threads,
        engine.config().rec_batch_size
    );

    let key = |results: Vec<OcrResult>| {
        results
            .into_iter()
            .map(|r| {
                (
                    r.text,
                    r.confidence.to_bits(),
                    format!("{:?}", r.bounding_box),
                )
            })
            .collect::<Vec<_>>()
    };
    let sequential = || {
        batch
            .iter()
            .map(|image| key(engine.run_from_image(image).expect("ocr should succeed")))
            .collect::<Vec<_>>()
    };
    let many = || {
        engine
            .run_many_from_images(&batch)
            .into_iter()
            .map(|result| key(result.expect("ocr should succeed")))
            .collect::<Vec<_>>()
    };

    // Warm-up: compiles every plan the measured rounds need.
    let reference = sequential();
    assert_eq!(many(), reference, "run_many must match run_from_image");

    let mut seq_times = Vec::new();
    let mut many_times = Vec::new();
    for round in 0..rounds.max(1) {
        let mut time = |many_first: bool| {
            for many_turn in [many_first, !many_first] {
                let start = Instant::now();
                let results = if many_turn { many() } else { sequential() };
                let elapsed = start.elapsed().as_secs_f64();
                assert_eq!(results, reference, "results changed between runs");
                if many_turn {
                    many_times.push(elapsed);
                } else {
                    seq_times.push(elapsed);
                }
            }
        };
        time(round % 2 == 1);
    }
    let median = |times: &mut Vec<f64>| {
        times.sort_by(|a, b| a.total_cmp(b));
        times[times.len() / 2]
    };
    let (seq, many) = (median(&mut seq_times), median(&mut many_times));
    let n = batch.len() as f64;
    println!("run_from_image loop: {:.2} s ({:.2} img/s)", seq, n / seq);
    println!(
        "run_many_from_images: {:.2} s ({:.2} img/s), {:.2}x",
        many,
        n / many,
        seq / many
    );
    println!("results identical in every round");
}
