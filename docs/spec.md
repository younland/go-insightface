# Spec: go-insightface

## Objective

Build a production-oriented Go inference library for InsightFace-compatible ONNX models without CGO or OpenCV. The library will load ONNX Runtime through `onnxruntime-purego`, detect and align faces, produce normalized embeddings, and provide exact identity-level search suitable for galleries of at least 3,000 identities.

The library is model-file agnostic: it does not redistribute InsightFace model weights. Applications are responsible for supplying models with suitable licenses.

## Tech Stack

- Go 1.25 or newer.
- `github.com/shota3506/onnxruntime-purego` pinned to commit `8db8bd7424b218de9ca0ebd894da8e86f5dc7983`.
- ONNX Runtime 1.23.x shared library, CPU execution provider in v0.1.
- Go standard image packages plus `golang.org/x/image/draw` for deterministic resizing.
- No CGO, OpenCV, database, HTTP server, or bundled model weights.

## Public Contract

```go
engine, err := insightface.New(insightface.Config{
    RuntimeLibrary: "/opt/onnxruntime/lib/libonnxruntime.so",
    DetectorModel:  "/opt/models/det_10g.onnx",
    RecognizerModel: "/opt/models/w600k_r50.onnx",
})
if err != nil { return err }
defer engine.Close()

faces, err := engine.Analyze(ctx, img)
embedding := faces[0].Embedding // L2-normalized []float32
```

The public API must:

- accept `context.Context` on inference calls;
- validate model outputs and image/landmark inputs at boundaries;
- return typed errors and never panic for caller-controlled input;
- return copied slices so callers cannot mutate engine state;
- make `Analyze`, `Detect`, and `Embed` safe for concurrent use;
- make `Close` idempotent and prevent close/run races.

## Commands

- Format: `gofmt -w .`
- Unit tests: `go test ./...`
- Race tests: `go test -race ./...`
- Static analysis: `go vet ./...`
- Benchmarks: `go test -run '^$' -bench . -benchmem ./...`
- Integration: `ORT_LIBRARY=/path/libonnxruntime.so DETECTOR_MODEL=/path/det_10g.onnx RECOGNIZER_MODEL=/path/w600k_r50.onnx go test -tags=integration ./...`

## Project Structure

- Root package: stable engine, face, embedding, and matcher API.
- `assets`: verified, cached, transactional build-stage asset acquisition.
- `cmd/insightface-assets`: dependency-free build tool over `assets.Download`.
- `runtimefs`: optional caller-asset filesystem and runtime materialization.
- `internal/imageutil`: resize, CHW conversion, affine alignment, bilinear sampling.
- `internal/scrfd`: SCRFD tensor decoding, anchors, NMS, landmark output.
- `internal/arcface`: ArcFace preprocessing and embedding normalization.
- `internal/ort`: narrow adapter around the unstable upstream runtime API.
- `testdata`: synthetic fixtures only; no restricted model weights or biometric data.
- `docs`: model contract, deployment, licensing, and migration notes.

## Code Style

```go
func (e *Engine) Analyze(ctx context.Context, src image.Image) ([]Face, error) {
    if err := ctx.Err(); err != nil {
        return nil, err
    }
    if src == nil {
        return nil, ErrInvalidImage
    }
    return e.analyze(ctx, src)
}
```

- Errors add context with `%w`; sentinel errors cover stable categories.
- No global mutable runtime/session state.
- Configuration is immutable after `New` returns.
- Prefer small pure functions for tensor transforms and ranking.

## Testing Strategy

- Unit tests first for normalization, NCHW conversion, affine alignment, SCRFD decoding, NMS, exact identity ranking, distance threshold, and Top1/Top2 margin.
- Runtime adapter tests use the upstream tiny ONNX fixture.
- Integration tests use caller-provided licensed models and sample images through environment variables.
- Concurrency tests run inference and close coordination under `-race`.
- Benchmarks cover 3,000 identities × 3 embeddings at 512 dimensions and report allocations.
- Golden comparison against an independent InsightFace implementation must keep cosine similarity of embeddings from the same aligned chip at `>= 0.999`.

## Boundaries

- Always: pin dependency commits; validate tensor names/shapes/types; L2-normalize embeddings; close every runtime tensor; document model fingerprints.
- Always: keep all runtime/model payloads outside the module; require model URLs and hashes from the caller.
- Ask first: add GPU providers, commit model weights, change the public API, or integrate into HoloFace.
- Never: label cosine similarity as probability; silently accept unknown tensor layouts; expose unrestricted biometric fixtures; claim military suitability without field FPIR/FNIR evaluation.

## Success Criteria

- Builds without CGO on Linux AMD64 and Linux ARM64.
- Loads ONNX Runtime 1.23.x by explicit path and runs the upstream fixture.
- Runs SCRFD + ArcFace end to end with licensed external models.
- Produces stable normalized embeddings (`abs(norm-1) <= 1e-5`).
- Same aligned chip matches independent reference output at cosine `>= 0.999`.
- Concurrent inference passes `go test -race`.
- Exact search over 9,000 embeddings (3,000 identities × 3) has no correctness loss and a documented benchmark.
- Repository contains license/dependency/model-license guidance and no model weights.
- Existing `New(Config)` remains compatible; optional `NewFromFS` loads models from caller-owned `fs.FS` assets.
- Build downloads verify SHA-256, stage all assets before publishing, and serialize publishers targeting the same output.

## Open Questions

- Production model license and exact detector/recognizer fingerprints must be selected by the deployer.
- ARM64 end-to-end latency requires target hardware; CI can only validate build and pure processing until hardware is available.
- `onnxruntime-purego` is currently unstable and untagged, so the adapter boundary may need a maintained `lib-x` fork if upstream changes or stalls.
