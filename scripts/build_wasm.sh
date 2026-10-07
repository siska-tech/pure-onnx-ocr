#!/usr/bin/env bash
# Builds the browser bindings (bindings/wasm) and runs wasm-bindgen.
#
#   scripts/build_wasm.sh [--threads] [--out-dir DIR]
#
# Without --threads: stable Rust, single-threaded, output in
#   examples/web/pkg (default).
# With --threads: the pinned nightly from bindings/wasm/threads (shared
#   memory, wasm-bindgen-rayon), output in examples/web/pkg-threads (default).
#   Pages that load it must be cross-origin isolated (COOP/COEP).
#
# Requires wasm-bindgen-cli at the version of the wasm-bindgen crate in
# Cargo.lock (cargo install wasm-bindgen-cli --version 0.2.105 --locked).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THREADS=0
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS=1 ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

LOCKED="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print }' "$ROOT/Cargo.lock")"
INSTALLED="$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' || true)"
if [[ "$INSTALLED" != "$LOCKED" ]]; then
  echo "wasm-bindgen-cli $LOCKED is required (found: ${INSTALLED:-none})." >&2
  echo "  cargo install wasm-bindgen-cli --version $LOCKED --locked" >&2
  exit 1
fi

if [[ "$THREADS" == 1 ]]; then
  OUT_DIR="${OUT_DIR:-$ROOT/examples/web/pkg-threads}"
  # rust-toolchain.toml and .cargo/config.toml in this directory select the
  # nightly toolchain, the atomics flags and target/wasm-threads.
  (cd "$ROOT/bindings/wasm/threads" && cargo build --release -p pure-onnx-ocr-wasm --features threads)
  WASM="$ROOT/target/wasm-threads/wasm32-unknown-unknown/release/pure_onnx_ocr_wasm.wasm"
else
  OUT_DIR="${OUT_DIR:-$ROOT/examples/web/pkg}"
  (cd "$ROOT" && cargo build --release -p pure-onnx-ocr-wasm --target wasm32-unknown-unknown)
  WASM="$ROOT/target/wasm32-unknown-unknown/release/pure_onnx_ocr_wasm.wasm"
fi

rm -rf "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" "$WASM"
echo "wrote $OUT_DIR:"
(cd "$OUT_DIR" && find . -type f | sort | sed 's|^\./|  |')
