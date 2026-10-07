//! pure-onnx-ocr (tract) vs OpenVINO Runtime on the same CPU, images, ONNX
//! files and pre/post-processing.
//!
//! One process measures one backend and one model, so that peak memory and
//! thread counts are not mixed up between backends. `run_bench.py` launches
//! these processes alternately (pure / OpenVINO, model by model) to spread
//! thermal and clock drift evenly, and aggregates the JSON reports.
//!
//! ```text
//! openvino-bench --backend pure|ov --model v6-small [--fixtures DIR]
//!     [--images general_ocr_002.jpg,ja.jpg] [--runs 5] [--model-only-runs 10]
//!     [--threads N]            pure: inference threads (default: crate default)
//!     [--rec-batch-size N]     both: recognition batch size (default 1 = crate default)
//!     [--ov-threads N]         ov: INFERENCE_NUM_THREADS (0 = auto)
//!     [--ov-det-hint LATENCY]  ov: PERFORMANCE_HINT for detection
//!     [--ov-rec-hint LATENCY]  ov: PERFORMANCE_HINT for recognition
//!     [--ov-rec-requests N]    ov: parallel recognition requests (0 = optimal)
//!     [--priority high|above_normal|normal] [--json OUT]
//! ```
//!
//! The OpenVINO pipeline is `OcrEngine::run_with_metrics_from_image_impl`
//! (no orientation classifiers) with `DetInferenceSession::run` and
//! `RecInferenceSession::run` replaced by OpenVINO infer requests. Every other
//! step is the crate's own public code: `DetPreProcessor`,
//! `DetPostProcessor::db_boxes`, `min_area_quad` + `crop_quad`,
//! `RecPreProcessor`, `RecPostProcessor` (CTC). The configuration is resolved
//! from `inference.yml` exactly like `OcrEngineBuilder::build`; the pure
//! backend checks that both resolutions are identical.

mod ov;
mod winproc;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use image::{imageops, DynamicImage, GenericImageView, Rgb, RgbImage};
use pure_onnx_ocr::{
    crop_quad, min_area_quad, DetInferenceSession, DetPolygonUnclipper, DetPostProcessor,
    DetPreProcessor, OcrEngineBuilder, OcrEngineConfig, OcrTimings, PaddleInferenceConfig, Polygon,
    RecCropMode, RecDictionary, RecInferenceSession, RecPostProcessor, RecPreProcessor,
    StageTimings,
};
use rayon::prelude::*;
use serde_json::{json, Value};

struct Args {
    backend: String,
    model: String,
    fixtures: PathBuf,
    images: Vec<String>,
    runs: usize,
    model_only_runs: usize,
    threads: Option<usize>,
    rec_batch_size: usize,
    ov_threads: usize,
    ov_det_hint: String,
    ov_rec_hint: String,
    ov_rec_requests: usize,
    priority: String,
    json: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        backend: "pure".into(),
        model: "v6-small".into(),
        fixtures: PathBuf::from("../../tests/fixtures"),
        images: vec!["general_ocr_002.jpg".into(), "ja.jpg".into()],
        runs: 5,
        model_only_runs: 10,
        threads: None,
        rec_batch_size: 1,
        ov_threads: 0,
        ov_det_hint: "LATENCY".into(),
        ov_rec_hint: "LATENCY".into(),
        ov_rec_requests: 1,
        priority: "high".into(),
        json: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("missing option value");
        match arg.as_str() {
            "--backend" => a.backend = value(),
            "--model" => a.model = value(),
            "--fixtures" => a.fixtures = PathBuf::from(value()),
            "--images" => a.images = value().split(',').map(str::to_string).collect(),
            "--runs" => a.runs = value().parse().expect("number"),
            "--model-only-runs" => a.model_only_runs = value().parse().expect("number"),
            "--threads" => a.threads = Some(value().parse().expect("number")),
            "--rec-batch-size" => a.rec_batch_size = value().parse().expect("number"),
            "--ov-threads" => a.ov_threads = value().parse().expect("number"),
            "--ov-det-hint" => a.ov_det_hint = value(),
            "--ov-rec-hint" => a.ov_rec_hint = value(),
            "--ov-rec-requests" => a.ov_rec_requests = value().parse().expect("number"),
            "--priority" => a.priority = value(),
            "--json" => a.json = Some(PathBuf::from(value())),
            other => panic!("unknown option {other}"),
        }
    }
    a.runs = a.runs.max(1);
    a
}

fn model_dirs(fixtures: &Path, model: &str) -> Result<(PathBuf, PathBuf)> {
    let (generation, tier) = model
        .split_once('-')
        .ok_or_else(|| anyhow!("model must look like v6-small"))?;
    let base = fixtures.join("models").join(match generation {
        "v5" => "ppocrv5",
        "v6" => "ppocrv6",
        other => bail!("unknown generation {other}"),
    });
    Ok((
        base.join(format!("{tier}_det")),
        base.join(format!("{tier}_rec")),
    ))
}

/// Same configuration `OcrEngineBuilder::new().det_model_dir(..).rec_model_dir(..)
/// .rec_batch_size(..).build()` produces (default options, no classifiers).
fn resolve_config(
    det_dir: &Path,
    rec_dir: &Path,
    rec_batch_size: usize,
    inference_threads: usize,
) -> Result<(OcrEngineConfig, RecDictionary)> {
    let mut config = OcrEngineConfig::default();
    config.rec_batch_size = rec_batch_size;
    config.inference_threads = inference_threads;
    let det = PaddleInferenceConfig::from_path(det_dir.join("inference.yml"))
        .map_err(|e| anyhow!("{e:?}"))?;
    if let Some(order) = det.color_order {
        config.det_preprocessor.color_order = order;
    }
    if let Some(mean) = det.normalize_mean {
        config.det_preprocessor.mean = mean;
    }
    if let Some(std) = det.normalize_std {
        config.det_preprocessor.std = std;
    }
    // det_postprocess_from_model_config is off by default: thresholds stay.
    let rec_yml = rec_dir.join("inference.yml");
    let rec = PaddleInferenceConfig::from_path(&rec_yml).map_err(|e| anyhow!("{e:?}"))?;
    if let Some(order) = rec.color_order {
        config.rec_preprocessor.color_order = order;
    }
    if let Some([_, height, width]) = rec.rec_image_shape {
        config.rec_preprocessor.target_height = height;
        config.rec_preprocessor.max_width = width;
        if config.rec_preprocessor.max_dynamic_width < width {
            config.rec_preprocessor.max_dynamic_width = width;
        }
    }
    let dictionary = RecDictionary::from_path(&rec_yml)
        .map_err(|e| anyhow!("{e:?}"))?
        .with_space_char();
    config.rec_postprocessor.blank_id = dictionary.blank_id();
    Ok((config, dictionary))
}

type Recognized = Vec<(String, f32, Polygon<f64>)>;

/// The engine pipeline with OpenVINO inference.
struct OvPipeline {
    config: OcrEngineConfig,
    det_pre: DetPreProcessor,
    det_post: DetPostProcessor,
    unclipper: DetPolygonUnclipper,
    rec_pre: RecPreProcessor,
    rec_post: RecPostProcessor,
    det: ov::OvDetSession,
    rec: ov::OvRecSession,
    /// Same size as the engine's pool; used for the same (pre-processing)
    /// steps the engine parallelises with `parallel_map`.
    pool: Option<rayon::ThreadPool>,
}

impl OvPipeline {
    fn run(&self, image: &DynamicImage) -> Result<(Recognized, OcrTimings)> {
        let pipeline_start = Instant::now();
        let mut timings = OcrTimings {
            total: Duration::ZERO,
            image_decode: Duration::ZERO,
            orientation: Duration::ZERO,
            detection: zero_stage(),
            recognition: zero_stage(),
        };
        let image_dims = image.dimensions();

        // ---- DetectionPipeline::detect_polygons_with_timings ----
        let t = Instant::now();
        let preprocessed = self.det_pre.process(image).map_err(|e| anyhow!("{e:?}"))?;
        timings.detection.preprocess = t.elapsed();
        let t = Instant::now();
        let inference = self.det.run(&preprocessed)?;
        timings.detection.inference = t.elapsed();
        let t = Instant::now();
        let boxes = self
            .det_post
            .db_boxes(&inference.probability_map, &self.unclipper)
            .map_err(|e| anyhow!("{e:?}"))?;
        let (scale_x, scale_y) = preprocessed.inverse_scale();
        let (width, height) = (image_dims.0 as f64, image_dims.1 as f64);
        let polygons: Vec<Polygon<f64>> = boxes
            .into_iter()
            .map(|det_box| {
                let mut coords: Vec<geo_types::Coord<f64>> = det_box
                    .quad
                    .iter()
                    .map(|&(x, y)| geo_types::Coord {
                        x: (x * scale_x).round().clamp(0.0, width),
                        y: (y * scale_y).round().clamp(0.0, height),
                    })
                    .collect();
                coords.push(coords[0]);
                Polygon::new(coords.into(), vec![])
            })
            .collect();
        timings.detection.postprocess = t.elapsed();

        if polygons.is_empty() {
            timings.total = pipeline_start.elapsed();
            return Ok((Vec::new(), timings));
        }

        // ---- OcrEngine::crop_regions ----
        let t = Instant::now();
        let rgb = image.to_rgb8();
        let crops: Vec<RgbImage> = polygons
            .iter()
            .map(|polygon| {
                if self.config.rec_crop_mode == RecCropMode::Rotated {
                    if let Some(crop) =
                        min_area_quad(polygon).and_then(|quad| crop_quad(&rgb, &quad))
                    {
                        return crop;
                    }
                }
                let (x, y, w, h) = text_region(polygon, image_dims);
                imageops::crop_imm(&rgb, x, y, w, h).to_image()
            })
            .collect();
        let crop_elapsed = t.elapsed();

        // ---- RecognitionPipeline::run_with_timings ----
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| aspect_ratio(&crops[a]).total_cmp(&aspect_ratio(&crops[b])));
        let chunks: Vec<&[usize]> = order.chunks(self.config.rec_batch_size.max(1)).collect();

        let t = Instant::now();
        let (rec_pre, crops) = (&self.rec_pre, &crops);
        let preprocess = |chunk: &&[usize]| {
            let batch: Vec<RgbImage> = chunk.iter().map(|&i| crops[i].clone()).collect();
            rec_pre.process_images(&batch)
        };
        let batches = match &self.pool {
            Some(pool) if chunks.len() > 1 => {
                pool.install(|| chunks.par_iter().map(preprocess).collect::<Vec<_>>())
            }
            _ => chunks.iter().map(preprocess).collect(),
        }
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("{e:?}"))?;
        timings.recognition.preprocess = t.elapsed() + crop_elapsed;

        let t = Instant::now();
        let inferences = self.rec.run_all(&batches)?;
        timings.recognition.inference = t.elapsed();

        let t = Instant::now();
        let decoded = inferences
            .iter()
            .map(|inference| self.rec_post.process(inference))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("{e:?}"))?;
        timings.recognition.postprocess = t.elapsed();

        let mut results = vec![None; crops.len()];
        for (chunk, sequences) in chunks.iter().zip(decoded) {
            for (&index, sequence) in chunk.iter().zip(sequences) {
                results[index] = Some(sequence);
            }
        }
        let recognized = polygons
            .into_iter()
            .zip(results)
            .map(|(polygon, sequence)| {
                let sequence = sequence.expect("every crop is decoded");
                (sequence.text, sequence.confidence, polygon)
            })
            .collect();
        timings.total = pipeline_start.elapsed();
        Ok((recognized, timings))
    }
}

fn zero_stage() -> StageTimings {
    StageTimings {
        preprocess: Duration::ZERO,
        inference: Duration::ZERO,
        postprocess: Duration::ZERO,
    }
}

fn aspect_ratio(crop: &RgbImage) -> f64 {
    crop.width() as f64 / crop.height().max(1) as f64
}

/// Copy of the engine's `polygon_to_text_region` (axis-aligned fallback).
fn text_region(polygon: &Polygon<f64>, image_dims: (u32, u32)) -> (u32, u32, u32, u32) {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon.exterior().points() {
        min_x = min_x.min(p.x());
        max_x = max_x.max(p.x());
        min_y = min_y.min(p.y());
        max_y = max_y.max(p.y());
    }
    let wl = image_dims.0.max(1) as f64;
    let hl = image_dims.1.max(1) as f64;
    let mut x1 = min_x.floor().max(0.0);
    let mut y1 = min_y.floor().max(0.0);
    let mut x2 = max_x.ceil().min(wl);
    let mut y2 = max_y.ceil().min(hl);
    if x2 <= x1 {
        x2 = (x1 + 1.0).min(wl);
    }
    if y2 <= y1 {
        y2 = (y1 + 1.0).min(hl);
    }
    if x2 <= x1 {
        x1 = (wl - 1.0).max(0.0);
        x2 = wl;
    }
    if y2 <= y1 {
        y1 = (hl - 1.0).max(0.0);
        y2 = hl;
    }
    let (x, y) = (x1.floor() as u32, y1.floor() as u32);
    let w = ((x2 - x1).ceil() as u32)
        .max(1)
        .min(image_dims.0.max(1) - x);
    let h = ((y2 - y1).ceil() as u32)
        .max(1)
        .min(image_dims.1.max(1) - y);
    (x, y, w, h)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn timings_json(t: &OcrTimings) -> Value {
    json!({
        "total": ms(t.total),
        "det_pre": ms(t.detection.preprocess),
        "det_inf": ms(t.detection.inference),
        "det_post": ms(t.detection.postprocess),
        "rec_pre": ms(t.recognition.preprocess),
        "rec_inf": ms(t.recognition.inference),
        "rec_post": ms(t.recognition.postprocess),
    })
}

fn results_json(results: &Recognized) -> Value {
    Value::Array(
        results
            .iter()
            .map(|(text, confidence, polygon)| {
                let quad: Vec<[f64; 2]> = polygon
                    .exterior()
                    .points()
                    .take(4)
                    .map(|p| [p.x(), p.y()])
                    .collect();
                json!({ "text": text, "confidence": confidence, "box": quad })
            })
            .collect(),
    )
}

fn mem_json(m: winproc::MemoryInfo) -> Value {
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    json!({
        "working_set_mb": mb(m.working_set),
        "peak_working_set_mb": mb(m.peak_working_set),
        "private_mb": mb(m.private_bytes),
        "peak_private_mb": mb(m.peak_private_bytes),
    })
}

fn median_ms(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Fixed recognition inputs, identical to `examples/ocr_bench.rs`:
/// synthetic 320x48 text-like crops.
fn synthetic_rec_crops(count: u32) -> Vec<RgbImage> {
    (0..count)
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
        .collect()
}

/// Times `f` `runs` times after one warm-up call; returns all samples in ms.
fn time_runs(runs: usize, mut f: impl FnMut() -> Result<()>) -> Result<Vec<f64>> {
    f()?;
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f()?;
            Ok(ms(t.elapsed()))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = parse_args();
    let throttling = winproc::disable_power_throttling();
    let priority = winproc::set_priority(&args.priority);
    let (det_dir, rec_dir) = model_dirs(&args.fixtures, &args.model)?;
    let det_onnx = det_dir.join("inference.onnx");
    let rec_onnx = rec_dir.join("inference.onnx");
    let images: Vec<(String, DynamicImage)> = args
        .images
        .iter()
        .map(|name| {
            let path = args.fixtures.join("images").join(name);
            image::open(&path)
                .with_context(|| format!("{path:?}"))
                .map(|img| (name.clone(), img))
        })
        .collect::<Result<_>>()?;

    let mut report = json!({
        "backend": args.backend,
        "model": args.model,
        "det_onnx": det_onnx.canonicalize()?,
        "rec_onnx": rec_onnx.canonicalize()?,
        "power_throttling_opt_out": throttling,
        "priority": if priority { args.priority.clone() } else { "unchanged".into() },
        "rec_batch_size": args.rec_batch_size,
        "runs": args.runs,
    });

    // ---- Load ----
    let cpu_start = winproc::cpu_seconds();
    let load_start = Instant::now();
    let run_fn: Box<dyn Fn(&DynamicImage) -> Result<(Recognized, OcrTimings)>>;
    let threads_used: usize;
    match args.backend.as_str() {
        "pure" => {
            let mut builder = OcrEngineBuilder::new()
                .det_model_dir(&det_dir)
                .rec_model_dir(&rec_dir)
                .rec_batch_size(args.rec_batch_size);
            if let Some(threads) = args.threads {
                builder = builder.inference_threads(threads);
            }
            let engine = builder.build()?;
            report["load_ms"] = json!(ms(load_start.elapsed()));
            threads_used = engine.config().inference_threads;
            let (resolved, _) =
                resolve_config(&det_dir, &rec_dir, args.rec_batch_size, threads_used)?;
            let matches = format!("{resolved:?}") == format!("{:?}", engine.config());
            if !matches {
                eprintln!(
                    "WARNING: OpenVINO-side config differs from the engine's\n  engine: {:?}\n  ours:   {resolved:?}",
                    engine.config()
                );
            }
            report["config_matches_engine"] = json!(matches);
            report["pure"] = json!({
                "inference_threads": threads_used,
                "multithread_supported": pure_onnx_ocr::MULTITHREAD_SUPPORTED,
                "default_inference_threads": pure_onnx_ocr::default_inference_threads(),
            });
            run_fn = Box::new(move |image| {
                let run = engine.run_with_metrics_from_image(image)?;
                let results = run
                    .results
                    .into_iter()
                    .map(|r| (r.text, r.confidence, r.bounding_box))
                    .collect();
                Ok((results, run.timings))
            });
        }
        "ov" => {
            threads_used = args
                .threads
                .unwrap_or_else(pure_onnx_ocr::default_inference_threads);
            let mut core = ov::new_core()?;
            report["openvino_version"] = json!(ov::version_string(&core));
            let det_settings = ov::OvSettings {
                hint: args.ov_det_hint.clone(),
                threads: args.ov_threads,
                streams: None,
            };
            let rec_settings = ov::OvSettings {
                hint: args.ov_rec_hint.clone(),
                threads: args.ov_threads,
                streams: None,
            };
            let det_compiled = ov::compile(&mut core, &det_onnx, &det_settings)?;
            let rec_compiled = ov::compile(&mut core, &rec_onnx, &rec_settings)?;
            let requests = if args.ov_rec_requests == 0 {
                rec_compiled
                    .get_property(&openvino::PropertyKey::OptimalNumberOfInferRequests)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            } else {
                args.ov_rec_requests
            };
            let (config, dictionary) =
                resolve_config(&det_dir, &rec_dir, args.rec_batch_size, threads_used)?;
            let pipeline = OvPipeline {
                det_pre: DetPreProcessor::new(config.det_preprocessor),
                det_post: DetPostProcessor::new(config.det_postprocessor),
                unclipper: DetPolygonUnclipper::new(config.det_unclipper),
                rec_pre: RecPreProcessor::new(config.rec_preprocessor.clone()),
                rec_post: RecPostProcessor::new(
                    std::sync::Arc::new(dictionary),
                    config.rec_postprocessor.clone(),
                ),
                det: ov::OvDetSession::new(det_compiled)?,
                rec: ov::OvRecSession::new(rec_compiled, requests)?,
                pool: (threads_used > 1).then(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads_used)
                        .build()
                        .expect("rayon pool")
                }),
                config,
            };
            report["load_ms"] = json!(ms(load_start.elapsed()));
            report["openvino"] = json!({
                "det": ov::describe(&pipeline.det.compiled),
                "rec": ov::describe(&pipeline.rec.compiled),
                "rec_requests": pipeline.rec.request_count(),
                "preprocess_threads": threads_used,
            });
            run_fn = Box::new(move |image| pipeline.run(image));
        }
        other => bail!("unknown backend {other}"),
    }
    report["load_cpu_s"] = json!(winproc::cpu_seconds() - cpu_start);
    report["after_load"] = json!({
        "memory": mem_json(winproc::memory()),
        "threads": winproc::thread_count(),
    });

    // ---- Pipeline: first (cold) run per image, then warm runs ----
    let mut per_image = serde_json::Map::new();
    let mut warm: Vec<Vec<Value>> = vec![Vec::new(); images.len()];
    for (name, image) in &images {
        let t = Instant::now();
        let (results, timings) = run_fn(image)?;
        let wall = ms(t.elapsed());
        per_image.insert(
            name.clone(),
            json!({
                "regions": results.len(),
                "first": timings_json(&timings),
                "first_wall_ms": wall,
                "results": results_json(&results),
            }),
        );
    }
    let after_first = json!({
        "memory": mem_json(winproc::memory()),
        "threads": winproc::thread_count(),
    });
    let cpu0 = winproc::cpu_seconds();
    let wall0 = Instant::now();
    let mut images_done = 0usize;
    for _ in 0..args.runs {
        for (i, (_, image)) in images.iter().enumerate() {
            let (_, timings) = run_fn(image)?;
            warm[i].push(timings_json(&timings));
            images_done += 1;
        }
    }
    let warm_wall = wall0.elapsed().as_secs_f64();
    let warm_cpu = winproc::cpu_seconds() - cpu0;
    for (i, (name, _)) in images.iter().enumerate() {
        let entry = per_image.get_mut(name).unwrap();
        let totals: Vec<f64> = warm[i]
            .iter()
            .map(|t| t["total"].as_f64().unwrap())
            .collect();
        entry["warm"] = Value::Array(warm[i].clone());
        entry["warm_total_median_ms"] = json!(median_ms(totals));
    }
    report["images"] = Value::Object(per_image);
    report["after_first_run"] = after_first;
    report["warm"] = json!({
        "wall_s": warm_wall,
        "cpu_s": warm_cpu,
        "avg_cores_busy": warm_cpu / warm_wall,
        "images": images_done,
        "images_per_s": images_done as f64 / warm_wall,
    });
    report["after_pipeline"] = json!({
        "memory": mem_json(winproc::memory()),
        "threads": winproc::thread_count(),
    });
    drop(run_fn);

    // ---- A. Model only: identical tensors for both backends ----
    if args.model_only_runs > 0 {
        let (config, _) = resolve_config(&det_dir, &rec_dir, 1, threads_used)?;
        let det_input = DetPreProcessor::new(config.det_preprocessor)
            .process(&images[0].1)
            .map_err(|e| anyhow!("{e:?}"))?;
        let rec_pre = RecPreProcessor::new(config.rec_preprocessor.clone());
        let rec8 = rec_pre
            .process_images(&synthetic_rec_crops(8))
            .map_err(|e| anyhow!("{e:?}"))?;
        let rec1 = rec_pre
            .process_images(&synthetic_rec_crops(1))
            .map_err(|e| anyhow!("{e:?}"))?;
        let runs = args.model_only_runs;
        let (det_ms, rec8_ms, rec1_ms, extra) = match args.backend.as_str() {
            "pure" => {
                let mut det = DetInferenceSession::load(&det_onnx)?;
                let mut rec = RecInferenceSession::load_with_input_height(
                    &rec_onnx,
                    config.rec_preprocessor.target_height,
                )?;
                det.set_inference_threads(threads_used);
                rec.set_inference_threads(threads_used);
                (
                    time_runs(runs, || {
                        det.run(&det_input).map(drop).map_err(|e| anyhow!("{e}"))
                    })?,
                    time_runs(runs, || {
                        rec.run(&rec8).map(drop).map_err(|e| anyhow!("{e}"))
                    })?,
                    time_runs(runs, || {
                        rec.run(&rec1).map(drop).map_err(|e| anyhow!("{e}"))
                    })?,
                    json!({ "inference_threads": threads_used }),
                )
            }
            _ => {
                let mut core = ov::new_core()?;
                let settings = ov::OvSettings {
                    hint: "LATENCY".into(),
                    threads: args.ov_threads,
                    streams: None,
                };
                let det = ov::OvDetSession::new(ov::compile(&mut core, &det_onnx, &settings)?)?;
                let rec = ov::OvRecSession::new(ov::compile(&mut core, &rec_onnx, &settings)?, 1)?;
                let extra = json!({
                    "det": ov::describe(&det.compiled),
                    "rec": ov::describe(&rec.compiled),
                });
                (
                    time_runs(runs, || det.run(&det_input).map(drop))?,
                    time_runs(runs, || rec.run(&rec8).map(drop))?,
                    time_runs(runs, || rec.run(&rec1).map(drop))?,
                    extra,
                )
            }
        };
        report["model_only"] = json!({
            "det_input": det_input.tensor.shape(),
            "rec8_input": rec8.tensor.shape(),
            "rec1_input": rec1.tensor.shape(),
            "det_ms": det_ms, "det_median_ms": median_ms(det_ms.clone()),
            "rec8_ms": rec8_ms, "rec8_median_ms": median_ms(rec8_ms.clone()),
            "rec1_ms": rec1_ms, "rec1_median_ms": median_ms(rec1_ms.clone()),
            "settings": extra,
        });
    }
    report["final"] = json!({
        "memory": mem_json(winproc::memory()),
        "threads": winproc::thread_count(),
    });

    // ---- Output ----
    eprintln!(
        "[{} {}] load {:.0} ms | {}",
        args.backend,
        args.model,
        report["load_ms"].as_f64().unwrap_or(0.0),
        images
            .iter()
            .map(|(name, _)| format!(
                "{name}: first {:.0} ms, warm median {:.0} ms",
                report["images"][name]["first"]["total"]
                    .as_f64()
                    .unwrap_or(0.0),
                report["images"][name]["warm_total_median_ms"]
                    .as_f64()
                    .unwrap_or(0.0)
            ))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let text = serde_json::to_string_pretty(&report)?;
    match &args.json {
        Some(path) => std::fs::write(path, text)?,
        None => println!("{text}"),
    }
    Ok(())
}
