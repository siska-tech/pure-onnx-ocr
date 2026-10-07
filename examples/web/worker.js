// Runs pure-onnx-ocr in a Web Worker so the page stays responsive while the
// (synchronous) WebAssembly inference runs.
//
// Cross-origin isolated pages (COOP/COEP headers, see README.md) load the
// multi-threaded build from ./pkg-threads: its rayon threads are nested
// Workers started from this Worker. Other pages, or a missing or failing
// ./pkg-threads, use the single-threaded build from ./pkg.

let wasm = null;
let engine = null;

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.text();
}

// A model is a PaddleOCR 3.x directory (`inference.onnx` + `inference.yml`)
// or `{ onnx, yml }` URLs; `yml` may be null for legacy single-file models.
async function loadModel(model) {
  if (typeof model === "string") {
    model = { onnx: `${model}/inference.onnx`, yml: `${model}/inference.yml` };
  }
  return [await fetchBytes(model.onnx), model.yml ? await fetchText(model.yml) : ""];
}

// `threads`: 0 = all logical CPUs, 1 = single-threaded build, n = n threads.
// `build` ("threads" or "single", for benchmarks) skips the automatic choice;
// "threads" fails instead of falling back.
async function loadWasm(threads, build) {
  if (build === "threads" || (build !== "single" && threads !== 1 && self.crossOriginIsolated)) {
    try {
      const module = await import("./pkg-threads/pure_onnx_ocr_wasm.js");
      const exports = await module.default();
      const poolSize = threads || navigator.hardwareConcurrency || 4;
      await module.initThreadPool(poolSize);
      return { module, exports, build: "threads", poolSize };
    } catch (error) {
      if (build === "threads") throw error;
      console.warn("multi-threaded build unavailable, using the single-threaded one:", error);
    }
  }
  const module = await import("./pkg/pure_onnx_ocr_wasm.js");
  const exports = await module.default();
  return { module, exports, build: "single", poolSize: 1 };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const started = performance.now();
      // The thread pool can only be started once per Worker: the page starts
      // a new Worker to switch models or thread counts.
      wasm ??= await loadWasm(data.threads ?? 0, data.build);
      let builder = new wasm.module.OcrEngineBuilder()
        .detModel(...(await loadModel(data.det)))
        .recModel(...(await loadModel(data.rec)));
      if (data.dictionary) builder = builder.dictionaryText(await fetchText(data.dictionary));
      if (data.textlineOrientation) {
        builder = builder.textlineOrientationModel(...(await loadModel(data.textlineOrientation)));
      }
      if (data.docOrientation) {
        builder = builder.docOrientationModel(...(await loadModel(data.docOrientation)));
      }
      engine = builder.build();
      self.postMessage({
        type: "loaded",
        ms: performance.now() - started,
        build: wasm.build,
        threads: engine.inferenceThreads,
        crossOriginIsolated: self.crossOriginIsolated,
      });
    } else if (data.type === "run") {
      const output = engine.runWithMetrics(new Uint8Array(data.image));
      self.postMessage({
        type: "result",
        results: output.results,
        timings: output.timings,
        docOrientationAngle: output.docOrientationAngle,
        // WebAssembly memory never shrinks, so its size is the peak so far.
        memoryBytes: wasm.exports.memory.buffer.byteLength,
      });
    }
  } catch (error) {
    self.postMessage({ type: "error", message: String(error && error.message ? error.message : error) });
  }
};
