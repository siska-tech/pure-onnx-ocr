# Integration Test Fixtures

The quickest way to get everything the default test suite needs (about 35 MB):

```bash
scripts/fetch_fixtures.sh          # PP-OCRv6 tiny, PP-OCRv5 mobile, classifiers, sample image
scripts/fetch_fixtures.sh --all    # + small/medium, PP-OCRv5 server, ja.jpg (ignored tests, ocr_bench)
```

The integration tests expect the following assets to be available either via
the `PURE_ONNX_OCR_FIXTURE_DIR` environment variable or under this directory:

```
fixtures/
  models/
    ppocrv5/
      det.onnx
      rec.onnx
      ppocrv5_dict.txt
      rec.yml                 # optional: inference.yml of the v5 rec export
      {mobile,server}_{det,rec}/ inference.onnx  inference.yml   # optional, for examples/ocr_bench
    ppocrv6/
      tiny_det/   inference.onnx  inference.yml
      tiny_rec/   inference.onnx  inference.yml
      small_det/  inference.onnx  inference.yml
      small_rec/  inference.onnx  inference.yml
      medium_det/ inference.onnx  inference.yml
      medium_rec/ inference.onnx  inference.yml
    PP-LCNet_x1_0_doc_ori/        inference.onnx  inference.yml   # optional
    PP-LCNet_x1_0_textline_ori/   inference.onnx  inference.yml   # optional
    PP-LCNet_x0_25_textline_ori/  inference.onnx  inference.yml   # optional
  images/
    demo.png
    general_ocr_002.jpg
```

The `demo.png` image should contain readable text that the PP-OCRv5 models can
detect. The ONNX models are the standard PaddleOCR exports. They can be copied
from the `models/ppocrv5/` directory used during development, or downloaded
from Hugging Face (`PaddlePaddle/PP-OCRv5_mobile_{det,rec}_onnx`; rename
`inference.onnx` to `det.onnx` / `rec.onnx`).

The PP-OCRv6 models and the sample image can be downloaded as follows:

```bash
cd tests/fixtures
for tier in tiny small medium; do
  for kind in det rec; do
    mkdir -p models/ppocrv6/${tier}_${kind}
    for f in inference.onnx inference.yml; do
      curl -L -o models/ppocrv6/${tier}_${kind}/${f} \
        https://huggingface.co/PaddlePaddle/PP-OCRv6_${tier}_${kind}_onnx/resolve/main/${f}
    done
  done
done
for model in PP-LCNet_x1_0_doc_ori PP-LCNet_x1_0_textline_ori PP-LCNet_x0_25_textline_ori; do
  mkdir -p models/${model}
  for f in inference.onnx inference.yml; do
    curl -L -o models/${model}/${f} \
      https://huggingface.co/PaddlePaddle/${model}_onnx/resolve/main/${f}
  done
done
mkdir -p images
curl -L -o images/general_ocr_002.jpg \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
```

The sample image is served with a `.png` URL but is actually a JPEG, so it is
saved as `.jpg`.

Because these files are large and may be subject to licensing constraints, the
repository does not bundle them.  CI systems or local developers should place
the assets in a secure location and point `PURE_ONNX_OCR_FIXTURE_DIR` to it:

```
PURE_ONNX_OCR_FIXTURE_DIR=/path/to/fixtures cargo test -- --ignored
```

The tests will automatically skip when the fixtures are not present, emitting a
message to indicate that real assets are required.
