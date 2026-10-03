// Runs pure-onnx-ocr in a Web Worker so the page stays responsive while the
// (synchronous) WebAssembly inference runs.
import init, { OcrEngineBuilder } from "./pkg/pure_onnx_ocr_wasm.js";

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

async function loadModel(dir) {
  return [await fetchBytes(`${dir}/inference.onnx`), await fetchText(`${dir}/inference.yml`)];
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const started = performance.now();
      await init();
      let builder = new OcrEngineBuilder()
        .detModel(...(await loadModel(data.det)))
        .recModel(...(await loadModel(data.rec)));
      if (data.textlineOrientation) {
        builder = builder.textlineOrientationModel(...(await loadModel(data.textlineOrientation)));
      }
      if (data.docOrientation) {
        builder = builder.docOrientationModel(...(await loadModel(data.docOrientation)));
      }
      engine = builder.build();
      self.postMessage({ type: "loaded", ms: performance.now() - started });
    } else if (data.type === "run") {
      const output = engine.runWithMetrics(new Uint8Array(data.image));
      self.postMessage({
        type: "result",
        results: output.results,
        timings: output.timings,
        docOrientationAngle: output.docOrientationAngle,
      });
    }
  } catch (error) {
    self.postMessage({ type: "error", message: String(error && error.message ? error.message : error) });
  }
};
