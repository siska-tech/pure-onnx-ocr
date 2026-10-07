"""Runs pure-onnx-ocr vs OpenVINO benchmark rounds and aggregates the results.

Each (backend config, model) pair runs in its own process (so peak memory and
thread counts are per backend). Within a round, every model is measured with
every backend config one after another, and the order of the configs rotates
from round to round, so thermal / clock drift hits every config alike.

    python run_bench.py [--models v6-small,v6-medium,v6-tiny] [--rounds 3]
        [--runs 5] [--model-only-runs 5] [--images general_ocr_002.jpg,ja.jpg]
        [--out results] [--no-single-thread] [--configs pure,ov,...]
        [--baseline-exe PATH] [--add-config NAME="ARGS"]...

Requires: a release build of this crate (`cargo build --release`) and, when
an OpenVINO config is measured, the `openvino` pip package in the running
interpreter (its `libs/` directory provides openvino_c.dll; no IR conversion
is done, the ONNX files are read directly by OpenVINO).

A/B mode (comparing two builds or two settings of pure-onnx-ocr):

    --baseline-exe PATH   adds the config `base`: the `pure` config run with
                          another build of this tool (e.g. one built in a git
                          worktree of the baseline commit).
    --add-config NAME="ARGS"
                          adds a config with the given CLI arguments, e.g.
                          --add-config t16="--backend pure --threads 16".

When either is given, the default `--configs` becomes the reference (`base`
if given, otherwise `pure`), `pure` and the added configs, without OpenVINO,
and the summary gains an "A/B" table (each config relative to the
reference). Output parity is always checked against the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRATE = HERE.parent.parent
EXE = HERE / "target" / "release" / ("openvino-bench.exe" if os.name == "nt" else "openvino-bench")

# name -> extra CLI arguments
CONFIGS = {
    # pure-onnx-ocr defaults: min(logical CPUs, 16) threads, rec batch 1,
    # recognition batches run in parallel.
    "pure": ["--backend", "pure"],
    # OpenVINO, recognition with the THROUGHPUT hint and the plugin's optimal
    # number of infer requests running batch-1 crops concurrently (same
    # structure as pure-onnx-ocr). Detection: LATENCY hint. Best e2e setting
    # in our sweep -> the primary OpenVINO number.
    "ov": ["--backend", "ov", "--ov-rec-hint", "THROUGHPUT", "--ov-rec-requests", "0"],
    # OpenVINO out of the box: LATENCY hint, one request, crops one by one.
    "ov-latency": ["--backend", "ov", "--ov-rec-hint", "LATENCY", "--ov-rec-requests", "1"],
}

# name -> executable, for configs that do not use this tree's build (A/B).
EXES: dict[str, Path] = {}

STAGES = ["total", "det_pre", "det_inf", "det_post", "rec_pre", "rec_inf", "rec_post"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def uses_openvino(config: str) -> bool:
    args = CONFIGS[config]
    return "--backend" in args and args[args.index("--backend") + 1] == "ov"


def openvino_env() -> dict:
    import openvino  # noqa: F401  (the pip package ships the C API DLLs)

    libs = Path(openvino.__file__).parent / "libs"
    env = dict(os.environ)
    env["PATH"] = str(libs) + os.pathsep + env.get("PATH", "")
    return env


def openvino_version() -> str:
    try:
        import openvino
    except ImportError:
        return "-"
    return openvino.__version__


def run_one(config: str, model: str, args, out_dir: Path, tag: str, extra=()) -> dict:
    out = out_dir / "raw" / f"{tag}_{config}_{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EXES.get(config, EXE)),
        *CONFIGS[config],
        "--model", model,
        "--fixtures", str(args.fixtures),
        "--images", args.images,
        "--runs", str(args.runs),
        "--model-only-runs", str(args.model_only_runs),
        "--json", str(out),
        *extra,
    ]
    t = time.time()
    proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")
    print(f"  {tag} {config:<10} {model:<10} {time.time() - t:6.1f}s  {proc.stderr.strip()}", flush=True)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def med(values):
    return statistics.median(values) if values else float("nan")


def environment(args, configs: list[str]) -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    models = {}
    for model in args.models:
        gen, tier = model.split("-")
        base = Path(args.fixtures) / "models" / ("ppocrv6" if gen == "v6" else "ppocrv5")
        for kind in ("det", "rec"):
            p = base / f"{tier}_{kind}" / "inference.onnx"
            models[f"{model}/{kind}"] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "cpu": sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"'),
        "power_plan": sh("powercfg /getactivescheme"),
        "battery_status": sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus"'),
        "openvino_python": openvino_version(),
        "rustc": sh("rustc --version"),
        "crate_commit": sh(f'git -C "{CRATE}" rev-parse --short HEAD'),
        "exes": {c: str(EXES.get(c, EXE)) for c in configs},
        "models": models,
        "images": args.images.split(","),
        "rounds": args.rounds,
        "warm_runs_per_process": args.runs,
        "configs": {c: CONFIGS[c] for c in configs},
        "reference": args.reference,
    }


def aggregate(reports: list[dict]) -> dict:
    """reports: all rounds of one (config, model)."""
    images = list(reports[0]["images"].keys())
    out = {"images": {}}
    for image in images:
        warm = [w for r in reports for w in r["images"][image]["warm"]]
        firsts = [r["images"][image]["first"] for r in reports]
        out["images"][image] = {
            "regions": reports[0]["images"][image]["regions"],
            "warm_n": len(warm),
            "warm": {s: med([w[s] for w in warm]) for s in STAGES},
            "warm_min_total": min(w["total"] for w in warm),
            "warm_max_total": max(w["total"] for w in warm),
            "first": {s: med([f[s] for f in firsts]) for s in STAGES},
        }
    out["load_ms"] = med([r["load_ms"] for r in reports])
    out["images_per_s"] = med([r["warm"]["images_per_s"] for r in reports])
    out["avg_cores_busy"] = med([r["warm"]["avg_cores_busy"] for r in reports])
    out["threads_after_load"] = med([r["after_load"]["threads"] for r in reports])
    out["threads_after_pipeline"] = med([r["after_pipeline"]["threads"] for r in reports])
    for key in ("peak_working_set_mb", "peak_private_mb"):
        out[key] = med([r["after_pipeline"]["memory"][key] for r in reports])
        out[key + "_max"] = max(r["after_pipeline"]["memory"][key] for r in reports)
    out["ws_after_load_mb"] = med([r["after_load"]["memory"]["working_set_mb"] for r in reports])
    if "model_only" in reports[0]:
        for key in ("det_ms", "rec8_ms", "rec1_ms"):
            out["model_only_" + key] = med([v for r in reports for v in r["model_only"][key]])
        out["model_only_shapes"] = {
            k: reports[0]["model_only"][k] for k in ("det_input", "rec8_input", "rec1_input")
        }
    out["settings"] = {
        k: reports[0].get(k)
        for k in ("pure", "openvino", "config_matches_engine", "power_throttling_opt_out", "priority")
        if k in reports[0]
    }
    return out


def parity(a: dict, b: dict) -> dict:
    """Compares OCR output of two single reports (first run)."""
    out = {}
    for image in a["images"]:
        ra, rb = a["images"][image]["results"], b["images"][image]["results"]
        out[image] = {
            "regions": [len(ra), len(rb)],
            "same_text": sum(x["text"] == y["text"] for x, y in zip(ra, rb)),
            "same_box": sum(x["box"] == y["box"] for x, y in zip(ra, rb)),
        }
    return out


def fmt_s(ms):
    return f"{ms / 1000:.2f} s"


def markdown(summary: dict) -> str:
    agg, env = summary["aggregate"], summary["environment"]
    ref = env["reference"]
    models = [m for m in summary["models"] if f"{ref}/{m}" in agg]
    images = env["images"]
    has_ov = all(f"pure/{m}" in agg and f"ov/{m}" in agg for m in models)
    L = []
    L.append("# pure-onnx-ocr vs OpenVINO (measured)\n")
    L.append(f"- {env['date']} / {env['cpu']} / {env['platform']}")
    L.append(f"- OpenVINO {env['openvino_python']} / {env['rustc']} / pure-onnx-ocr {env['crate_commit']}")
    for c, exe in env["exes"].items():
        if Path(exe) != EXE:
            L.append(f"- `{c}` runs `{exe}`")
    L.append(f"- rounds {env['rounds']} x warm runs {env['warm_runs_per_process']} per process "
             f"(warm medians over {env['rounds'] * env['warm_runs_per_process']} samples)\n")

    ab = [c for c in env["configs"] if c != ref and not uses_openvino(c)]
    if ab:
        L.append(f"## A/B: relative to `{ref}` (warm median, ratio < 1 is faster)\n")
        L.append("| Model | Image | Config | total ms | ratio | det inf ms | ratio "
                 "| rec inf ms | ratio | first run ms | ratio |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in models:
            for img in images:
                base = agg[f"{ref}/{m}"]["images"][img]
                for c in [ref, *ab]:
                    d = agg[f"{c}/{m}"]["images"][img]
                    cells = []
                    for value, base_value in (
                        (d["warm"]["total"], base["warm"]["total"]),
                        (d["warm"]["det_inf"], base["warm"]["det_inf"]),
                        (d["warm"]["rec_inf"], base["warm"]["rec_inf"]),
                        (d["first"]["total"], base["first"]["total"]),
                    ):
                        cells += [f"{value:.0f}", f"{value / base_value:.2f}"]
                    L.append(f"| {m} | {img} | {c} | " + " | ".join(cells) + " |")
        L.append("")

    if not has_ov:
        return "\n".join(L + markdown_tail(summary, models, images)) + "\n"

    L.append("## Headline: end-to-end warm latency (median), OpenVINO = best config (`ov`)\n")
    L.append("| Model | Image | pure-onnx-ocr | OpenVINO | Ratio (pure / OV) |")
    L.append("|---|---|---:|---:|---:|")
    for m in models:
        for img in images:
            p = agg[f"pure/{m}"]["images"][img]["warm"]["total"]
            o = agg[f"ov/{m}"]["images"][img]["warm"]["total"]
            L.append(f"| {m} | {img} | {fmt_s(p)} | {fmt_s(o)} | {p / o:.2f} |")
    L.append("")
    L.append("| Model | pure-onnx-ocr (mean of images) | OpenVINO | Ratio | OpenVINO `ov-latency` | Ratio |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m in models:
        p = statistics.mean(agg[f"pure/{m}"]["images"][i]["warm"]["total"] for i in images)
        o = statistics.mean(agg[f"ov/{m}"]["images"][i]["warm"]["total"] for i in images)
        ol = statistics.mean(agg[f"ov-latency/{m}"]["images"][i]["warm"]["total"] for i in images) \
            if f"ov-latency/{m}" in agg else float("nan")
        L.append(f"| {m} | {fmt_s(p)} | {fmt_s(o)} | {p / o:.2f} | {fmt_s(ol)} | {p / ol:.2f} |")
    L.append("")

    return "\n".join(L + markdown_tail(summary, models, images)) + "\n"


def markdown_tail(summary: dict, models: list[str], images: list[str]) -> list[str]:
    """Sections shared by the OpenVINO comparison and A/B mode."""
    agg, env = summary["aggregate"], summary["environment"]
    has_ov = all(f"pure/{m}" in agg and f"ov/{m}" in agg for m in models)
    L = []
    L.append("## Stage breakdown (warm median, ms)\n")
    L.append("| Model | Image | Config | regions | first run | total | det pre | det inf | det post | rec pre (+crop) | rec inf | rec post |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for img in images:
            for c in CONFIGS:
                key = f"{c}/{m}"
                if key not in agg:
                    continue
                d = agg[key]["images"][img]
                w = d["warm"]
                L.append(
                    f"| {m} | {img} | {c} | {d['regions']} | {d['first']['total']:.0f} | {w['total']:.0f} | "
                    + " | ".join(f"{w[s]:.1f}" for s in STAGES[1:]) + " |"
                )
    L.append("")

    L.append("## Load, cold start, resources\n")
    L.append("| Model | Config | load ms | first run ms (img1) | load+first ms | throughput img/s | avg cores busy | threads (after run) | peak WS MB | peak private MB |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for c in CONFIGS:
            key = f"{c}/{m}"
            if key not in agg:
                continue
            a = agg[key]
            first = a["images"][images[0]]["first"]["total"]
            L.append(
                f"| {m} | {c} | {a['load_ms']:.0f} | {first:.0f} | {a['load_ms'] + first:.0f} | "
                f"{a['images_per_s']:.2f} | {a['avg_cores_busy']:.1f} | {a['threads_after_pipeline']:.0f} | "
                f"{a['peak_working_set_mb']:.0f} | {a['peak_private_mb']:.0f} |"
            )
    L.append("")

    if has_ov:
        L += model_only_tables(summary, models)

    L.append(f"## Output parity (first run, relative to `{env['reference']}`)\n")
    L.append("| Model | Image | Config | regions (ref / config) | identical text | identical box |")
    L.append("|---|---|---|---:|---:|---:|")
    for c, per_model in summary["parity"].items():
        for m, per in per_model.items():
            for img, d in per.items():
                L.append(f"| {m} | {img} | {c} | {d['regions'][0]} / {d['regions'][1]} | "
                         f"{d['same_text']} | {d['same_box']} |")
    L.append("")

    L.append("## Models (same ONNX file for every config)\n")
    L.append("| Model | bytes | sha256 |")
    L.append("|---|---:|---|")
    for k, v in env["models"].items():
        L.append(f"| {k} | {v['bytes']} | `{v['sha256'][:16]}…` |")
    return L


def model_only_tables(summary: dict, models: list[str]) -> list[str]:
    agg = summary["aggregate"]
    L = []
    L.append("## A. Model only (identical tensors, median ms)\n")
    L.append("| Model | det pure | det OV | ratio | rec 8x320 pure | rec 8x320 OV | ratio | rec 1x320 pure | rec 1x320 OV | ratio |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        p, o = agg[f"pure/{m}"], agg[f"ov/{m}"]
        if "model_only_det_ms" not in p:
            continue
        cells = []
        for k in ("det_ms", "rec8_ms", "rec1_ms"):
            a, b = p["model_only_" + k], o["model_only_" + k]
            cells += [f"{a:.1f}", f"{b:.1f}", f"{a / b:.2f}"]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    shapes = agg[f"pure/{models[0]}"].get("model_only_shapes")
    if shapes:
        L.append(f"\nShapes: det {shapes['det_input']}, rec {shapes['rec8_input']} / {shapes['rec1_input']}. "
                 "pure: crate default threads; OV: LATENCY hint, 1 request, automatic threads.\n")

    st = summary.get("single_thread")
    if st:
        L.append("## A'. Model only, 1 thread each (kernel efficiency, median ms)\n")
        L.append("| Model | det pure | det OV | ratio | rec 8x320 pure | rec 8x320 OV | ratio | rec 1x320 pure | rec 1x320 OV | ratio |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m, (p, o) in st.items():
            cells = []
            for k in ("det_ms", "rec8_ms", "rec1_ms"):
                a, b = p["model_only"][k.replace("_ms", "_median_ms")], o["model_only"][k.replace("_ms", "_median_ms")]
                cells += [f"{a:.1f}", f"{b:.1f}", f"{a / b:.2f}"]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def main():
    global ENV
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v6-small,v6-medium,v6-tiny")
    ap.add_argument("--configs", default=None,
                    help="comma-separated configs (default: the built-in configs, or the A/B set)")
    ap.add_argument("--baseline-exe", default=None,
                    help="adds the config `base`: `pure` run with this build of openvino-bench")
    ap.add_argument("--add-config", action="append", default=[], metavar='NAME="ARGS"',
                    help="adds a config with these openvino-bench arguments (repeatable)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model-only-runs", type=int, default=5)
    ap.add_argument("--images", default="general_ocr_002.jpg,ja.jpg")
    ap.add_argument("--fixtures", default=str(CRATE / "tests" / "fixtures"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-single-thread", action="store_true")
    args = ap.parse_args()
    args.models = args.models.split(",")
    added = []
    if args.baseline_exe:
        exe = Path(args.baseline_exe).resolve()
        if not exe.exists():
            raise SystemExit(f"baseline exe not found: {exe}")
        CONFIGS["base"] = list(CONFIGS["pure"])
        EXES["base"] = exe
        added.append("base")
    for spec in args.add_config:
        name, sep, cli = spec.partition("=")
        if not sep or not name or name in CONFIGS:
            raise SystemExit(f"--add-config expects a new NAME=ARGS, got {spec!r}")
        CONFIGS[name] = shlex.split(cli)
        added.append(name)
    args.reference = "base" if args.baseline_exe else "pure"
    if args.configs:
        configs = args.configs.split(",")
    elif added:
        configs = list(dict.fromkeys([args.reference, "pure", *added]))
    else:
        configs = list(CONFIGS)
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}")
    if args.reference not in configs:
        raise SystemExit(f"the reference config `{args.reference}` must be measured")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not EXE.exists():
        raise SystemExit("build first: cargo build --release")
    ENV = openvino_env() if any(uses_openvino(c) for c in configs) else dict(os.environ)

    env = environment(args, configs)
    print(json.dumps({k: env[k] for k in ("cpu", "power_plan", "battery_status", "openvino_python")}, ensure_ascii=False))
    reports: dict[str, list[dict]] = {}
    for rnd in range(args.rounds):
        print(f"round {rnd + 1}/{args.rounds}", flush=True)
        order = configs[rnd % len(configs):] + configs[: rnd % len(configs)]
        for model in args.models:
            for config in order:
                r = run_one(config, model, args, out_dir, f"r{rnd}")
                reports.setdefault(f"{config}/{model}", []).append(r)

    single_thread = {}
    if not args.no_single_thread and "pure" in configs and any(uses_openvino(c) for c in configs):
        print("single-thread model-only", flush=True)
        st_args = argparse.Namespace(**{**vars(args), "runs": 1, "images": args.images.split(",")[0]})
        for model in args.models:
            p = run_one("pure", model, st_args, out_dir, "st", ["--threads", "1"])
            o = run_one("ov-latency", model, st_args, out_dir, "st", ["--ov-threads", "1", "--threads", "1"])
            single_thread[model] = (p, o)

    summary = {
        "environment": env,
        "models": args.models,
        "aggregate": {k: aggregate(v) for k, v in reports.items()},
        "parity": {
            c: {m: parity(reports[f"{args.reference}/{m}"][0], reports[f"{c}/{m}"][0]) for m in args.models}
            for c in configs
            if c != args.reference
        },
        "single_thread": single_thread,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "single_thread"}
                  | {"single_thread": {m: [p["model_only"], o["model_only"]] for m, (p, o) in single_thread.items()}},
                  f, ensure_ascii=False, indent=1)
    md = markdown(summary)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


ENV: dict = {}

if __name__ == "__main__":
    main()
