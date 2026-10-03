"""Generate PaddleOCR reference outputs for the parity tests.

Runs the official PaddleOCR 3.x pipeline on the same ONNX models and images as
`tests/paddle_parity.rs` and writes one JSON file per (model, image) under
`tests/reference/`. The JSON files are committed, so the Rust tests do not need
Python.

Setup (uv):

    uv venv .venv --python 3.12
    uv pip install --python .venv -r scripts/requirements-reference.txt
    scripts/fetch_fixtures.sh --all

Run from the repository root:

    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      .venv/Scripts/python scripts/paddleocr_reference.py    # Windows
      .venv/bin/python     scripts/paddleocr_reference.py    # Linux / macOS

The pipeline uses PaddleOCR's defaults (detection at native resolution with
limit_type=min / limit_side_len=64 / max_side_limit=4000, thresh 0.3,
box_thresh 0.6, unclip_ratio 1.5) with document preprocessing and text-line
orientation disabled, and ONNX Runtime as the inference engine so that both
implementations run the very same model files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(os.environ.get("PURE_ONNX_OCR_FIXTURE_DIR", ROOT / "tests" / "fixtures"))
OUT_DIR = ROOT / "tests" / "reference"

# name -> (det model name, det dir, rec model name, rec dir), dirs relative to fixtures/models
MODELS = {
    "v6-tiny": ("PP-OCRv6_tiny_det", "ppocrv6/tiny_det", "PP-OCRv6_tiny_rec", "ppocrv6/tiny_rec"),
    "v6-small": ("PP-OCRv6_small_det", "ppocrv6/small_det", "PP-OCRv6_small_rec", "ppocrv6/small_rec"),
    "v6-medium": ("PP-OCRv6_medium_det", "ppocrv6/medium_det", "PP-OCRv6_medium_rec", "ppocrv6/medium_rec"),
    "v5-mobile": ("PP-OCRv5_mobile_det", "ppocrv5/mobile_det", "PP-OCRv5_mobile_rec", "ppocrv5/mobile_rec"),
}
IMAGES = ["general_ocr_002.jpg", "ja.jpg"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--images", default=",".join(IMAGES))
    args = parser.parse_args()

    import paddleocr
    from paddleocr import PaddleOCR

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models.split(","):
        det_name, det_dir, rec_name, rec_dir = MODELS[name]
        det_path = FIXTURES / "models" / det_dir
        rec_path = FIXTURES / "models" / rec_dir
        if not (det_path / "inference.onnx").exists() or not (rec_path / "inference.onnx").exists():
            print(f"skip {name}: models missing under {FIXTURES / 'models'}", file=sys.stderr)
            continue
        ocr = PaddleOCR(
            text_detection_model_name=det_name,
            text_detection_model_dir=str(det_path),
            text_recognition_model_name=rec_name,
            text_recognition_model_dir=str(rec_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="onnxruntime",
        )
        for image in args.images.split(","):
            image_path = FIXTURES / "images" / image
            if not image_path.exists():
                print(f"skip {image}: not found", file=sys.stderr)
                continue
            result = ocr.predict(str(image_path))[0].json["res"]
            regions = [
                {
                    "text": text,
                    "score": round(float(score), 6),
                    "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in poly],
                }
                for text, score, poly in zip(
                    result["rec_texts"], result["rec_scores"], result["rec_polys"]
                )
            ]
            payload = {
                "generator": f"paddleocr {paddleocr.__version__} (engine=onnxruntime)",
                "model": name,
                "image": image,
                "det_params": result["text_det_params"],
                "regions": regions,
            }
            out = OUT_DIR / f"{name}__{Path(image).stem}.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)} ({len(regions)} regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
