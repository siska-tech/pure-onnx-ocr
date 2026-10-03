use std::env;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use pure_onnx_ocr::{
    DetLimitType, OcrEngineBuilder, OcrError, OcrResult, OcrRunWithMetrics, RecCropMode,
    StageTimings,
};

const DEFAULT_DET_MODEL: &str = "models/ppocrv5/det.onnx";
const DEFAULT_REC_MODEL: &str = "models/ppocrv5/rec.onnx";
const DEFAULT_DICTIONARY: &str = "models/ppocrv5/ppocrv5_dict.txt";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {}", err);
        if let Some(source) = err.source() {
            eprintln!("    caused by: {}", source);
        }
        process::exit(1);
    }
}

fn run() -> Result<(), RunError> {
    let cli = Cli::parse(env::args())?;
    init_logger(cli.verbose);

    if cli.show_help {
        println!("{}", Cli::usage());
        return Ok(());
    }

    let mut builder = OcrEngineBuilder::new();
    builder = match &cli.det_model_dir {
        Some(dir) => builder.det_model_dir(dir),
        None => builder.det_model_path(&cli.det_model),
    };
    builder = match &cli.rec_model_dir {
        Some(dir) => builder.rec_model_dir(dir),
        None => builder.rec_model_path(&cli.rec_model),
    };
    match (&cli.dictionary, &cli.rec_model_dir) {
        (Some(dictionary), _) => builder = builder.dictionary_path(dictionary),
        (None, Some(_)) => {}
        (None, None) => builder = builder.dictionary_path(DEFAULT_DICTIONARY),
    }
    if let Some(threshold) = cli.det_thresh {
        builder = builder.det_threshold(threshold);
    }
    if let Some(threshold) = cli.det_box_thresh {
        builder = builder.det_box_threshold(threshold);
    }
    if cli.no_space_char {
        builder = builder.rec_use_space_char(false);
    }
    if let Some(limit_type) = cli.det_limit_type {
        builder = builder.det_limit_type(limit_type);
    }
    if let Some(limit) = cli.det_max_side_limit {
        builder = builder.det_max_side_limit(limit);
    }
    if cli.det_params_from_config {
        builder = builder.det_postprocess_from_model_config(true);
    }
    if let Some(mode) = cli.crop_mode {
        builder = builder.rec_crop_mode(mode);
    }
    if let Some(dir) = &cli.doc_ori_model_dir {
        builder = builder.doc_orientation_model_dir(dir);
    }
    if let Some(dir) = &cli.textline_ori_model_dir {
        builder = builder.textline_orientation_model_dir(dir);
    }

    if let Some(limit) = cli.det_limit_side_len {
        builder = builder.det_limit_side_len(limit);
    }
    if let Some(unclip) = cli.det_unclip_ratio {
        builder = builder.det_unclip_ratio(unclip);
    }
    if let Some(batch_size) = cli.rec_batch_size {
        builder = builder.rec_batch_size(batch_size);
    }
    if let Some(threads) = cli.threads {
        builder = builder.inference_threads(threads);
    }

    let engine = builder.build().map_err(RunError::from)?;

    let image_path = cli
        .image_path
        .as_ref()
        .expect("image path should be present when help is not requested");

    let start = Instant::now();
    let run = engine
        .run_with_metrics_from_path(image_path)
        .map_err(RunError::from)?;
    let total_duration = start.elapsed();
    if cli.benchmark {
        print_benchmark_report(image_path, &run);
    }
    let doc_orientation_angle = run.doc_orientation_angle;
    let results = run.results;

    println!("Input image: {}", image_path.display());
    println!(
        "Detection model: {}",
        display_source(engine.det_model_path())
    );
    println!(
        "Recognition model: {}",
        display_source(engine.rec_model_path())
    );
    println!("Dictionary: {}", display_source(engine.dictionary_path()));
    println!("Recognition batch size: {}", engine.rec_batch_size());
    println!("Inference threads: {}", engine.config().inference_threads);
    if let Some(angle) = doc_orientation_angle {
        println!(
            "Document orientation: {} degrees (rotated upright before detection)",
            angle
        );
    }
    println!("Total time: {:.3} seconds", total_duration.as_secs_f64());

    if results.is_empty() {
        println!("No text regions detected.");
    } else {
        println!("Detected {} text regions:", results.len());
        for (index, result) in results.iter().enumerate() {
            print_result(index, result);
        }
    }

    Ok(())
}

fn print_result(index: usize, result: &OcrResult) {
    println!("--- Region {} ---", index + 1);
    println!("Text: {}", result.text);
    println!("Confidence: {:.3}", result.confidence);
    println!(
        "Polygon: {}",
        format_polygon(result.bounding_box.exterior())
    );
}

fn format_polygon(line_string: &geo_types::LineString<f64>) -> String {
    let mut points = line_string
        .points()
        .map(|point| format!("({:.1}, {:.1})", point.x(), point.y()))
        .collect::<Vec<_>>();

    // Avoid printing duplicate last point if polygon is closed.
    if points.len() >= 2 && points.first() == points.last() {
        points.pop();
    }

    points.join(" -> ")
}

#[derive(Debug)]
struct Cli {
    image_path: Option<PathBuf>,
    det_model: PathBuf,
    rec_model: PathBuf,
    dictionary: Option<PathBuf>,
    det_model_dir: Option<PathBuf>,
    rec_model_dir: Option<PathBuf>,
    det_thresh: Option<f32>,
    det_box_thresh: Option<f32>,
    no_space_char: bool,
    det_limit_type: Option<DetLimitType>,
    det_max_side_limit: Option<u32>,
    det_params_from_config: bool,
    crop_mode: Option<RecCropMode>,
    threads: Option<usize>,
    doc_ori_model_dir: Option<PathBuf>,
    textline_ori_model_dir: Option<PathBuf>,
    det_limit_side_len: Option<u32>,
    det_unclip_ratio: Option<f64>,
    rec_batch_size: Option<usize>,
    benchmark: bool,
    verbose: bool,
    show_help: bool,
}

impl Cli {
    fn parse<I, S>(args: I) -> Result<Self, RunError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut iter = args.into_iter();
        let _program_name = iter.next();

        let mut cli = Cli {
            image_path: None,
            det_model: PathBuf::from(DEFAULT_DET_MODEL),
            rec_model: PathBuf::from(DEFAULT_REC_MODEL),
            dictionary: None,
            det_model_dir: None,
            rec_model_dir: None,
            det_thresh: None,
            det_box_thresh: None,
            no_space_char: false,
            det_limit_type: None,
            det_max_side_limit: None,
            det_params_from_config: false,
            crop_mode: None,
            threads: None,
            doc_ori_model_dir: None,
            textline_ori_model_dir: None,
            det_limit_side_len: None,
            det_unclip_ratio: None,
            rec_batch_size: None,
            benchmark: false,
            verbose: false,
            show_help: false,
        };

        while let Some(arg) = iter.next() {
            let arg = arg.into();
            match arg.as_str() {
                "--help" | "-h" => {
                    cli.show_help = true;
                    return Ok(cli);
                }
                "--image" => {
                    let value = next_value("--image", &mut iter)?;
                    cli.image_path = Some(PathBuf::from(value));
                }
                "--det-model" => {
                    let value = next_value("--det-model", &mut iter)?;
                    cli.det_model = PathBuf::from(value);
                }
                "--rec-model" => {
                    let value = next_value("--rec-model", &mut iter)?;
                    cli.rec_model = PathBuf::from(value);
                }
                "--dictionary" => {
                    let value = next_value("--dictionary", &mut iter)?;
                    cli.dictionary = Some(PathBuf::from(value));
                }
                "--det-model-dir" => {
                    let value = next_value("--det-model-dir", &mut iter)?;
                    cli.det_model_dir = Some(PathBuf::from(value));
                }
                "--rec-model-dir" => {
                    let value = next_value("--rec-model-dir", &mut iter)?;
                    cli.rec_model_dir = Some(PathBuf::from(value));
                }
                "--det-thresh" => {
                    let value = next_value("--det-thresh", &mut iter)?;
                    cli.det_thresh = Some(parse_unit_interval("--det-thresh", &value)?);
                }
                "--det-box-thresh" => {
                    let value = next_value("--det-box-thresh", &mut iter)?;
                    cli.det_box_thresh = Some(parse_unit_interval("--det-box-thresh", &value)?);
                }
                "--no-space-char" => {
                    cli.no_space_char = true;
                }
                "--det-limit-type" => {
                    let value = next_value("--det-limit-type", &mut iter)?;
                    cli.det_limit_type = Some(match value.as_str() {
                        "max" => DetLimitType::Max,
                        "min" => DetLimitType::Min,
                        other => {
                            return Err(RunError::cli(format!(
                            "invalid value for --det-limit-type: `{}` (expected `max` or `min`)",
                            other
                        )))
                        }
                    });
                }
                "--det-max-side-limit" => {
                    let value = next_value("--det-max-side-limit", &mut iter)?;
                    let parsed = value.parse::<u32>().map_err(|_| {
                        RunError::cli(format!(
                            "invalid value for --det-max-side-limit: `{}`",
                            value
                        ))
                    })?;
                    cli.det_max_side_limit = Some(parsed);
                }
                "--det-params-from-config" => {
                    cli.det_params_from_config = true;
                }
                "--doc-ori-model-dir" => {
                    let value = next_value("--doc-ori-model-dir", &mut iter)?;
                    cli.doc_ori_model_dir = Some(PathBuf::from(value));
                }
                "--textline-ori-model-dir" => {
                    let value = next_value("--textline-ori-model-dir", &mut iter)?;
                    cli.textline_ori_model_dir = Some(PathBuf::from(value));
                }
                "--threads" => {
                    let value = next_value("--threads", &mut iter)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        RunError::cli(format!("invalid value for --threads: `{}`", value))
                    })?;
                    if parsed == 0 {
                        return Err(RunError::cli(
                            "--threads must be greater than zero".to_string(),
                        ));
                    }
                    cli.threads = Some(parsed);
                }
                "--crop-mode" => {
                    let value = next_value("--crop-mode", &mut iter)?;
                    cli.crop_mode = Some(match value.as_str() {
                        "rotated" => RecCropMode::Rotated,
                        "axis" => RecCropMode::AxisAligned,
                        other => {
                            return Err(RunError::cli(format!(
                            "invalid value for --crop-mode: `{}` (expected `rotated` or `axis`)",
                            other
                        )))
                        }
                    });
                }
                "--det-limit-side-len" => {
                    let value = next_value("--det-limit-side-len", &mut iter)?;
                    let parsed = value.parse::<u32>().map_err(|_| {
                        RunError::cli(format!(
                            "invalid value for --det-limit-side-len: `{}`",
                            value
                        ))
                    })?;
                    cli.det_limit_side_len = Some(parsed);
                }
                "--det-unclip-ratio" => {
                    let value = next_value("--det-unclip-ratio", &mut iter)?;
                    let parsed = value.parse::<f64>().map_err(|_| {
                        RunError::cli(format!("invalid value for --det-unclip-ratio: `{}`", value))
                    })?;
                    cli.det_unclip_ratio = Some(parsed);
                }
                "--rec-batch-size" => {
                    let value = next_value("--rec-batch-size", &mut iter)?;
                    let parsed = value.parse::<usize>().map_err(|_| {
                        RunError::cli(format!("invalid value for --rec-batch-size: `{}`", value))
                    })?;
                    if parsed == 0 {
                        return Err(RunError::cli(
                            "--rec-batch-size must be greater than zero".to_string(),
                        ));
                    }
                    cli.rec_batch_size = Some(parsed);
                }
                "--benchmark" => {
                    cli.benchmark = true;
                }
                "--verbose" | "-v" => {
                    cli.verbose = true;
                }
                other if other.starts_with('-') => {
                    return Err(RunError::cli(format!("unknown option `{}`", other)));
                }
                positional => match cli.image_path {
                    None => cli.image_path = Some(PathBuf::from(positional)),
                    Some(_) => {
                        return Err(RunError::cli(format!(
                            "unexpected positional argument `{}`",
                            positional
                        )));
                    }
                },
            }
        }

        if cli.image_path.is_none() {
            return Err(RunError::cli(
                "missing input image path. provide an image via positional argument or `--image`."
                    .to_string(),
            ));
        }

        Ok(cli)
    }

    fn usage() -> String {
        let mut text = String::new();
        text.push_str("Usage:\n");
        text.push_str("  ocr_smoke <IMAGE_PATH> [options]\n");
        text.push_str("  ocr_smoke --image <IMAGE_PATH> [options]\n\n");
        text.push_str("Options:\n");
        text.push_str("  -h, --help                    Show this help message and exit\n");
        text.push_str(&format!(
            "      --det-model PATH          Detection model path (default: {})\n",
            DEFAULT_DET_MODEL
        ));
        text.push_str(&format!(
            "      --rec-model PATH          Recognition model path (default: {})\n",
            DEFAULT_REC_MODEL
        ));
        text.push_str(&format!(
            "      --dictionary PATH         Dictionary path (default: {})\n",
            DEFAULT_DICTIONARY
        ));
        text.push_str(
            "      --det-limit-side-len N    Override detection preprocessing limit side length\n",
        );
        text.push_str("      --det-unclip-ratio R      Override detection polygon unclip ratio\n");
        text.push_str("      --rec-batch-size N        Override recognition batch size (> 0)\n");
        text.push_str(
            "      --det-model-dir DIR       PaddleOCR detection model directory (inference.onnx + inference.yml)\n",
        );
        text.push_str(
            "      --rec-model-dir DIR       PaddleOCR recognition model directory; its inference.yml supplies the dictionary\n",
        );
        text.push_str(
            "      --det-thresh T            DBNet binarisation threshold (default: 0.3)\n",
        );
        text.push_str(
            "      --det-box-thresh T        Minimum mean score per detected box (default: 0.6)\n",
        );
        text.push_str(
            "      --det-limit-type max|min  Bound the longest (max, default) or shortest (min) side by --det-limit-side-len\n",
        );
        text.push_str(
            "      --det-max-side-limit N    Upper bound for the longest detection side (default: 4000)\n",
        );
        text.push_str(
            "      --det-params-from-config  Use thresh/box_thresh/unclip_ratio from the detection inference.yml\n",
        );
        text.push_str(
            "      --doc-ori-model-dir DIR   Document orientation classifier (PP-LCNet_x1_0_doc_ori) directory
",
        );
        text.push_str(
            "      --textline-ori-model-dir DIR  Text-line orientation classifier (PP-LCNet_x*_textline_ori) directory
",
        );
        text.push_str(
            "      --no-space-char           Do not append the space class to the dictionary\n",
        );
        text.push_str(
            "      --threads N               Inference threads (default: logical CPUs, at most 8)\n",
        );
        text.push_str("      --benchmark               Emit timing diagnostics for benchmarking\n");
        text.push_str(
            "  -v, --verbose                 Print model loading and inference logs to stderr\n",
        );
        text
    }
}

fn parse_unit_interval(flag: &str, value: &str) -> Result<f32, RunError> {
    match value.parse::<f32>() {
        Ok(parsed) if (0.0..=1.0).contains(&parsed) => Ok(parsed),
        _ => Err(RunError::cli(format!(
            "invalid value for {}: `{}` (expected a number between 0 and 1)",
            flag, value
        ))),
    }
}

fn next_value<I, S>(flag: &str, iter: &mut I) -> Result<String, RunError>
where
    I: Iterator<Item = S>,
    S: Into<String>,
{
    iter.next()
        .map(Into::into)
        .ok_or_else(|| RunError::cli(format!("expected value after `{}`", flag)))
}

#[derive(Debug)]
enum RunError {
    Cli(String),
    Ocr(OcrError),
}

impl RunError {
    fn cli(message: String) -> Self {
        Self::Cli(message)
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Cli(message) => write!(f, "{}", message),
            RunError::Ocr(error) => write!(f, "{}", error),
        }
    }
}

impl From<OcrError> for RunError {
    fn from(value: OcrError) -> Self {
        Self::Ocr(value)
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunError::Cli(_) => None,
            RunError::Ocr(_) => None,
        }
    }
}

fn print_benchmark_report(image_path: &std::path::Path, run: &OcrRunWithMetrics) {
    println!("[INFO] benchmark.image={}", image_path.display());
    print_timing_line("benchmark.total_seconds", run.timings.total);
    print_timing_line("benchmark.image_decode_seconds", run.timings.image_decode);
    print_timing_line("benchmark.orientation_seconds", run.timings.orientation);
    print_stage_timings("benchmark.det", &run.timings.detection);
    print_stage_timings("benchmark.rec", &run.timings.recognition);
}

fn print_timing_line(label: &str, duration: std::time::Duration) {
    println!("[INFO] {}={:.6}", label, duration.as_secs_f64());
}

fn print_stage_timings(prefix: &str, stage: &StageTimings) {
    print_timing_line(&format!("{}.preprocess_seconds", prefix), stage.preprocess);
    print_timing_line(&format!("{}.inference_seconds", prefix), stage.inference);
    print_timing_line(
        &format!("{}.postprocess_seconds", prefix),
        stage.postprocess,
    );
}

/// Minimal stderr logger so the CLI can surface the library's `log` output
/// without pulling in a logging framework.
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

fn init_logger(verbose: bool) {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(if verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Warn
        });
    }
}

fn display_source(path: Option<&std::path::Path>) -> String {
    path.map(|p| p.display().to_string())
        .unwrap_or_else(|| "<memory>".to_string())
}
