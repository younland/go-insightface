# Validation record

This record documents technical compatibility testing. It is not a grant of
model rights and is not a claim that the tested weights are suitable for any
production, military, government, or commercial deployment.

## Environment

- Date: 2026-08-04
- Platform: Linux AMD64, Intel Core i5-12500H
- Go: 1.26.5 (module minimum: Go 1.25)
- ONNX Runtime: 1.23.2, CPU execution provider, API 23
- Runtime adapter:
  `github.com/shota3506/onnxruntime-purego@8db8bd7424b218de9ca0ebd894da8e86f5dc7983`

## Technical model fixtures

The external InsightFace `buffalo_l` v0.7 files below were used only as
non-committed technical compatibility fixtures:

| File | SHA-256 |
| --- | --- |
| `det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |

The release ZIP passed a full CRC test before the models were loaded. The
officially distributed weights are described by InsightFace as non-commercial
research models; they are not distributed by this repository.

## Runtime adapter migration

- Date: 2026-11-02
- Platform: Windows AMD64
- Go: 1.27.3 (module minimum: Go 1.27)
- Runtime adapter: `github.com/younland/onnxruntime`, pinned in `go.mod`

The ONNX Runtime binding was migrated from
`github.com/shota3506/onnxruntime-purego` (which did not compile on Windows)
to `github.com/younland/onnxruntime`. The new adapter negotiates the OrtApi
table with the loaded library at startup (supported range 1.23 through 1.31).

Re-validation results:

- `go build ./...`, `go vet ./...`, and the pure-Go unit suite pass on
  Windows AMD64.
- `TestRuntimeExecutesONNXModel` passed against the official CPU builds of
  both ONNX Runtime 1.23.2 (negotiated API 23, the minimum) and 1.31.1
  (negotiated API 31), loading the model through the reader/bytes path.
- The previous Linux validation results below were recorded against the old
  adapter and remain the reference for model-level accuracy.

## Original validation record (2026-08-04, old adapter)

- The upstream tiny ONNX fixture loaded and ran through the pure-Go adapter.
- SCRFD and ArcFace ran end-to-end on an external 512x512 face image.
- The first detection's box and five landmarks matched an independent
  Python/OpenCV/ONNX Runtime implementation within one pixel per coordinate.
- The Go and independent-reference normalized ArcFace embeddings had cosine
  similarity `0.99916426` (acceptance criterion: `>= 0.999`).
- The output embedding had 512 dimensions and L2 norm within `1e-5` of 1.
- The entire integration suite passed under the Go race detector.
- Exact matching over 3,000 identities × 3 samples × 512 dimensions measured
  approximately `5.11 ms/op`, `0 B/op`, and `0 allocs/op` on the machine above.
- Linux ARM64 with CGO disabled passed cross-compilation (`go build ./...`).
- `NewFromFS` materialized the real Linux runtime from a content-addressed
  cache, loaded the upstream tiny ONNX model directly from an `fs.FS` reader,
  and created and closed both sessions under the race detector.
- The build-stage downloader verified the exact official ONNX Runtime 1.23.2
  Linux AMD64 archive (`1fa4dcae...c5f9b`), extracted its declared shared
  library member, and downloaded both external model fixtures from a local
  HTTP server. All three output hashes matched, and the extracted runtime ran
  the upstream integration model successfully.
- Failure-injection tests confirmed that a bad final model leaves the prior
  three-asset output bundle intact; success replaces it without retaining
  stale files.

## Commands

```bash
go test ./...
go test -race ./...
go vet ./...
go test -run '^$' -bench . -benchmem ./...
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build ./...
```

Asset provisioning additionally ran the CLI against an HTTP-served copy of
the official runtime archive and the two external model fixtures, then ran
`TestRuntimeExecutesONNXModel` against the extracted output library.

The integration run supplied the runtime, model, image, and independent JSON
reference paths through environment variables as documented in the README.

## Remaining deployment validation

- Calibrate similarity and margin thresholds on legally usable, representative
  deployment data and report identity-search FPIR/FNIR at the target gallery
  size.
- Run end-to-end latency and concurrency tests on the target ARM64 hardware.
- Validate camera-specific blur, pose, lighting, occlusion, and compression.
- Complete legal review of the selected production weights and training data.
