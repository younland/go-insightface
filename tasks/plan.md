# Implementation Plan

1. Establish the module, pinned runtime adapter, public contracts, and pure math/image primitives.
2. Implement and test SCRFD preprocessing/postprocessing.
3. Implement and test ArcFace 5-point alignment, preprocessing, and normalized output.
4. Compose the concurrent-safe engine and exact identity matcher.
5. Run ONNX Runtime fixture and external-model integration tests.
6. Add benchmarks, deployment/licensing docs, CI, release metadata, and publish to `lib-x/go-insightface`.

Risks are isolated behind `internal/ort`: the upstream runtime is untagged, supports only ONNX Runtime API 23, and declares its API unstable. Model weights remain external because InsightFace-provided weights have separate usage restrictions.
