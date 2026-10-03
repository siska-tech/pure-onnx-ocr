# `ROADMAP_fix.md`

## 🛠 Fix & Issue Backlog

The main roadmap tracks feature delivery.  
This document captures follow-up fixes, regression hunts, and operational hardening tasks.

### F0: Smoke Runner & Result Quality

| Status | Task ID              | Summary                                                                 | Notes                                                  |
| :----- | :------------------- | :---------------------------------------------------------------------- | :----------------------------------------------------- |
| `[x]`  | `task-fix-000`       | Ship `ocr_smoke` CLI and document current limitations                   | Baseline utility is ready; OCR result quality unstable |
| `[ ]`  | `task-fix-001`        | Investigate noisy OCR outputs from `ocr_smoke` and stabilise detection | Branch `fix/001-ocr-smoke-quality`; blank token alignment implemented, awaiting smoke test re-run |
| `[x]`  | `task-fix-002`        | Replace provisional recognition confidence metric with softmax-based scoring | Detects pre-softmax probability outputs or applies log-sum-exp Softmax, then averages confidences; CLI now reports calibrated values |
| `[x]`  | `task-fix-003`        | Add benchmark-friendly timing output to `ocr_smoke` CLI                | `--benchmark` flag emits stage timings; JSON export tracked as follow-up |

### F1: Tooling & Diagnostics

| Status | Task ID        | Summary                                      | Notes |
| :----- | :------------- | :------------------------------------------- | :---- |
| `[ ]`  | _TBD_          | Add structured logging & tracing for engine | Draft once root-cause investigation starts |


