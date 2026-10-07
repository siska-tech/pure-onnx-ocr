// Measures the web demo's builds in headless Chrome and checks their output
// against the native build.
//
//   node examples/web/bench.mjs [--models v6-tiny,v6-small,v6-medium,v5-mobile]
//       [--threads 1,2,4,8] [--rounds 4] [--image models/sample.png]
//       [--native-dir DIR] [--out results.json] [--playwright MODULE]
//
// Serves examples/web with COOP/COEP headers (serve.mjs --coi) and, for each
// model, runs the single-threaded build (pkg) and the multi-threaded build
// (pkg-threads) with each thread count, in a fresh Worker each. Reports the
// first run (which compiles the inference plans), the median of the later
// runs and the WebAssembly memory size at the end (memory never shrinks, so
// this is the peak).
//
// Output check: every configuration must return the same text and boxes as
// the single-threaded build; with --native-dir DIR, also as DIR/<model>.json
// written by `cargo run --release --example ocr_json` (text must be equal,
// coordinates within 0.01 px).
//
// Needs Playwright (`npm i -D playwright`, or --playwright with the path to
// its index.mjs) and its Chromium. Paths are relative to examples/web.
import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const presets = {
  "v6-tiny": { det: "models/ppocrv6/tiny_det", rec: "models/ppocrv6/tiny_rec" },
  "v6-small": { det: "models/ppocrv6/small_det", rec: "models/ppocrv6/small_rec" },
  "v6-medium": { det: "models/ppocrv6/medium_det", rec: "models/ppocrv6/medium_rec" },
  "v5-mobile": { det: "models/ppocrv5/mobile_det", rec: "models/ppocrv5/mobile_rec" },
  // Legacy single-file layout (det.onnx, rec.onnx, ppocrv5_dict.txt).
  "v5-legacy": {
    det: { onnx: "models/ppocrv5/det.onnx", yml: null },
    rec: { onnx: "models/ppocrv5/rec.onnx", yml: null },
    dictionary: "models/ppocrv5/ppocrv5_dict.txt",
  },
};

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
};
const models = option("--models", "v6-tiny,v6-small,v6-medium,v5-mobile").split(",");
const threadCounts = option("--threads", "1,2,4,8").split(",").map(Number);
const rounds = Number(option("--rounds", "4"));
const image = option("--image", "models/sample.png");
const nativeDir = option("--native-dir");
const outFile = option("--out");
const port = Number(option("--port", "8097"));
const { chromium } = await import(option("--playwright", "playwright"));

const webDir = fileURLToPath(new URL(".", import.meta.url));
const server = spawn(process.execPath, [`${webDir}serve.mjs`, String(port), "--coi"], {
  stdio: ["ignore", "pipe", "inherit"],
});
await new Promise((resolve) => server.stdout.once("data", resolve));

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const sameOutput = (a, b, tolerance) =>
  a.length === b.length &&
  a.every(
    (r, i) =>
      r.text === b[i].text &&
      r.box.length === b[i].box.length &&
      r.box.every(([x, y], j) => Math.abs(x - b[i].box[j][0]) <= tolerance && Math.abs(y - b[i].box[j][1]) <= tolerance)
  );

const browser = await chromium.launch();
const rows = [];
let failures = 0;
try {
  for (const model of models) {
    const engine = presets[model];
    if (!engine) throw new Error(`unknown model ${model}; known: ${Object.keys(presets).join(", ")}`);
    const native = nativeDir ? JSON.parse(await readFile(`${nativeDir}/${model}.json`, "utf8")) : null;
    let reference = null;
    const configs = [{ build: "single", threads: 1 }, ...threadCounts.map((threads) => ({ build: "threads", threads }))];
    for (const { build, threads } of configs) {
      const page = await browser.newPage();
      page.on("pageerror", (error) => console.error(`page error: ${error.message}`));
      await page.goto(`http://localhost:${port}/bench.html`);
      await page.waitForFunction(() => window.benchReady);
      if (!(await page.evaluate(() => crossOriginIsolated))) throw new Error("page is not cross-origin isolated");
      const result = await page.evaluate((config) => window.runBench(config), {
        image,
        rounds,
        engine: { ...engine, build, threads },
      });
      await page.close();

      reference ??= result.results;
      const matchesSingle = sameOutput(result.results, reference, 0);
      const matchesNative = native ? sameOutput(result.results, native, 0.01) : null;
      if (!matchesSingle || matchesNative === false) failures++;
      const later = result.runs.slice(1);
      const row = {
        model,
        build,
        threads: result.threads,
        loadMs: Math.round(result.loadMs),
        firstMs: Math.round(result.runs[0].wall),
        laterMs: later.length ? Math.round(median(later.map((r) => r.wall))) : null,
        detectionMs: later.length ? Math.round(median(later.map((r) => r.detection))) : null,
        recognitionMs: later.length ? Math.round(median(later.map((r) => r.recognition))) : null,
        memoryMiB: Math.round(result.memoryBytes / 2 ** 20),
        regions: result.results.length,
        matchesSingle,
        matchesNative,
      };
      rows.push(row);
      console.error(JSON.stringify(row));
    }
  }
} finally {
  await browser.close();
  server.kill();
}

console.log("| Model | Build | Threads | First run | Later runs (median) | Detection | Recognition | Wasm memory | Output |");
console.log("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |");
for (const r of rows) {
  const output = !r.matchesSingle ? "differs" : r.matchesNative === null ? "same" : r.matchesNative ? "= native" : "≠ native";
  const s = (ms) => (ms === null ? "-" : `${(ms / 1000).toFixed(2)} s`);
  console.log(
    `| ${r.model} | ${r.build} | ${r.threads ?? "-"} | ${s(r.firstMs)} | ${s(r.laterMs)} | ${s(r.detectionMs)} | ${s(r.recognitionMs)} | ${r.memoryMiB} MiB | ${output} |`
  );
}
if (outFile) await writeFile(outFile, JSON.stringify(rows, null, 2));
if (failures) {
  console.error(`${failures} configuration(s) returned different output`);
  process.exitCode = 1;
}
