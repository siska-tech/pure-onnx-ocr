# Integration Test Fixtures

The integration tests expect the following assets to be available either via
the `PURE_ONNX_OCR_FIXTURE_DIR` environment variable or under this directory:

```
fixtures/
  models/
    ppocrv5/
      det.onnx
      rec.onnx
      ppocrv5_dict.txt
  images/
    demo.png
```

The `demo.png` image should contain readable text that the PP-OCRv5 models can
detect.  The ONNX models are the standard PaddleOCR exports and can be copied
from the `models/ppocrv5/` directory used during development.

Because these files are large and may be subject to licensing constraints, the
repository does not bundle them.  CI systems or local developers should place
the assets in a secure location and point `PURE_ONNX_OCR_FIXTURE_DIR` to it:

```
PURE_ONNX_OCR_FIXTURE_DIR=/path/to/fixtures cargo test -- --ignored
```

The tests will automatically skip when the fixtures are not present, emitting a
message to indicate that real assets are required.




