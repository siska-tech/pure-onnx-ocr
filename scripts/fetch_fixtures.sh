#!/usr/bin/env bash
# Downloads the models and images used by the tests into tests/fixtures
# (or $PURE_ONNX_OCR_FIXTURE_DIR). Files that already exist are skipped.
#
#   scripts/fetch_fixtures.sh          # minimal set used by `cargo test`
#   scripts/fetch_fixtures.sh --all    # + PP-OCRv6 small/medium, PP-OCRv5
#                                      #   server, x1_0 text-line classifier
#                                      #   and ja.jpg (ignored tests, ocr_bench)
set -euo pipefail

ROOT="${PURE_ONNX_OCR_FIXTURE_DIR:-$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures}"
HF="https://huggingface.co/PaddlePaddle"
ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1

fetch() { # url dest
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then return; fi
  mkdir -p "$(dirname "$dest")"
  echo "downloading $dest"
  curl -fsSL --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

paddle_dir() { # hf-repo dest-dir
  fetch "$HF/$1/resolve/main/inference.onnx" "$2/inference.onnx"
  fetch "$HF/$1/resolve/main/inference.yml" "$2/inference.yml"
}

# PP-OCRv6 tiny (default pipeline tests)
paddle_dir PP-OCRv6_tiny_det_onnx "$ROOT/models/ppocrv6/tiny_det"
paddle_dir PP-OCRv6_tiny_rec_onnx "$ROOT/models/ppocrv6/tiny_rec"

# PP-OCRv5 mobile in the legacy single-file layout (engine unit tests)
fetch "$HF/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx" "$ROOT/models/ppocrv5/det.onnx"
fetch "$HF/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx" "$ROOT/models/ppocrv5/rec.onnx"
fetch "$HF/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml" "$ROOT/models/ppocrv5/rec.yml"
fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt" \
  "$ROOT/models/ppocrv5/ppocrv5_dict.txt"

# Orientation classifiers
paddle_dir PP-LCNet_x1_0_doc_ori_onnx "$ROOT/models/PP-LCNet_x1_0_doc_ori"
paddle_dir PP-LCNet_x0_25_textline_ori_onnx "$ROOT/models/PP-LCNet_x0_25_textline_ori"

# Sample image (served as .png but actually a JPEG)
fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png" \
  "$ROOT/images/general_ocr_002.jpg"

if [[ $ALL -eq 1 ]]; then
  for tier in small medium; do
    paddle_dir "PP-OCRv6_${tier}_det_onnx" "$ROOT/models/ppocrv6/${tier}_det"
    paddle_dir "PP-OCRv6_${tier}_rec_onnx" "$ROOT/models/ppocrv6/${tier}_rec"
  done
  for tier in mobile server; do
    paddle_dir "PP-OCRv5_${tier}_det_onnx" "$ROOT/models/ppocrv5/${tier}_det"
    paddle_dir "PP-OCRv5_${tier}_rec_onnx" "$ROOT/models/ppocrv5/${tier}_rec"
  done
  paddle_dir PP-LCNet_x1_0_textline_ori_onnx "$ROOT/models/PP-LCNet_x1_0_textline_ori"
  fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/japan_2.jpg" \
    "$ROOT/images/ja.jpg"
fi

echo "fixtures ready under $ROOT"
