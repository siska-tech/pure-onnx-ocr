// Minimal static file server for the web demo (no dependencies).
// Usage: node examples/web/serve.mjs [port] [--coi] [--root DIR]
// WebAssembly must be served as application/wasm, and ES module workers need
// an http(s) origin, so opening index.html from the file system does not work.
//
// --coi adds the headers that make the page cross-origin isolated
// (Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy:
// require-corp). The multi-threaded build (pkg-threads) needs them for
// SharedArrayBuffer; without them the demo uses the single-threaded build.
// --root serves another directory (default: this one).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const option = (name) => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args.splice(index, 2)[1];
};
const coi = args.includes("--coi");
if (coi) args.splice(args.indexOf("--coi"), 1);
const root = resolve(option("--root") ?? fileURLToPath(new URL(".", import.meta.url)));
const port = Number(args[0] ?? 8080);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".yml": "text/yaml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
};
const isolationHeaders = coi
  ? {
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
      // Every response is same-origin, so this only documents the intent.
      "cross-origin-resource-policy": "same-origin",
    }
  : {};

createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://localhost");
    let path = normalize(join(root, decodeURIComponent(url.pathname)));
    if (!path.startsWith(root)) throw Object.assign(new Error("forbidden"), { status: 403 });
    if ((await stat(path)).isDirectory()) path = join(path, "index.html");
    const body = await readFile(path);
    response.writeHead(200, {
      "content-type": types[extname(path)] ?? "application/octet-stream",
      ...isolationHeaders,
    });
    response.end(body);
  } catch (error) {
    response.writeHead(error.status ?? 404, isolationHeaders);
    response.end(String(error.message));
  }
}).listen(port, () =>
  console.log(`pure-onnx-ocr demo: http://localhost:${port}/${coi ? " (cross-origin isolated)" : ""}`)
);
