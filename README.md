# go-insightface

`go-insightface` runs SCRFD face detection and ArcFace feature extraction from
Go without CGO or OpenCV. It loads ONNX Runtime dynamically through
[`onnxruntime-purego`](https://github.com/shota3506/onnxruntime-purego), keeps
model weights outside the module, and includes an exact identity matcher for
galleries of several thousand people.

## Status

The initial backend targets Go 1.25+, ONNX Runtime 1.23.x (API 23), and CPU
inference on Linux or macOS. The upstream runtime wrapper is pre-release, so it
is pinned to a commit and isolated behind an internal adapter. Windows and GPU
execution are not supported in the first release.

No model weights or biometric samples are included.

## Install

```bash
go get github.com/lib-x/go-insightface
```

Download ONNX Runtime 1.23.x for the deployment platform and provide an
explicit path to its shared library. Provide separately licensed,
SCRFD/ArcFace-compatible ONNX models; see [the model contract](docs/model-contract.md).

## Detect and embed

```go
package main

import (
	"context"
	"image/jpeg"
	"os"

	insightface "github.com/lib-x/go-insightface"
)

func run() error {
	file, err := os.Open("face.jpg")
	if err != nil {
		return err
	}
	defer file.Close()

	img, err := jpeg.Decode(file)
	if err != nil {
		return err
	}

	engine, err := insightface.New(insightface.Config{
		RuntimeLibrary:  "/opt/onnxruntime/lib/libonnxruntime.so.1.23.2",
		DetectorModel:   "/opt/models/detector.onnx",
		RecognizerModel: "/opt/models/recognizer.onnx",
	})
	if err != nil {
		return err
	}
	defer engine.Close()

	faces, err := engine.Analyze(context.Background(), img)
	if err != nil {
		return err
	}
	for _, face := range faces {
		_ = face.Detection.Score
		_ = face.Embedding.Values() // copied, L2-normalized values
	}
	return nil
}
```

`Engine.Detect`, `Engine.Embed`, and `Engine.Analyze` are safe for concurrent
use. `Close` waits for in-flight calls and is idempotent. Context cancellation
is checked at Go boundaries; the pinned upstream wrapper cannot interrupt a
native inference that has already started.

## Identity matching

Keep multiple good enrollment images per identity. Matching selects the best
sample for each identity, then applies both a cosine-similarity threshold and a
Top-1/Top-2 different-identity margin:

```go
matcher, err := insightface.NewMatcher(insightface.MatcherConfig{
	MinSimilarity: 0.55, // example only: calibrate on deployment data
	MinMargin:     0.05,
}, gallery)
if err != nil {
	return err
}

result, err := matcher.Match(query)
if err != nil {
	return err
}
if result.Accepted {
	_ = result.Best.Identity
}
```

Do not treat cosine similarity as confidence or probability. Thresholds above
are illustrative; calibrate them with deployment-specific FPIR/FNIR tests.
The included benchmark covers 3,000 identities × 3 samples × 512 dimensions.

## Model changes and historical data

An embedding is valid only in the model space that produced it. Store the
model fingerprint/version next to every embedding. After changing the
recognizer, regenerate historical embeddings from retained source images via
an idempotent enrollment API, keep old/new galleries separate, and switch only
after the new gallery is complete. Existing vectors cannot be mathematically
converted to a different ArcFace model space.

## Licensing and sensitive deployments

The library code is MIT licensed. Model weights have independent terms.
InsightFace's officially distributed pretrained models are described as
non-commercial research models; do not use those weights in military,
government, commercial, or production environments without explicit licensing
for the intended use. Use appropriately licensed or internally trained weights.

Face images and embeddings are biometric data. Review
[deployment and operational guidance](docs/deployment.md) before integration.

## Verification

```bash
go test ./...
go test -race ./...
go vet ./...
go test -run '^$' -bench . -benchmem ./...
```

An end-to-end test accepts external models and a local face image:

```bash
ORT_LIBRARY=/path/libonnxruntime.so.1.23.2 \
DETECTOR_MODEL=/path/detector.onnx \
RECOGNIZER_MODEL=/path/recognizer.onnx \
FACE_IMAGE=/path/face.jpg \
go test -tags=integration -run TestEngineRunsLicensedExternalModels .
```

Set `REFERENCE_RESULT=/path/reference.json` to additionally compare the first
detection and embedding with an independent implementation. The JSON object
contains `box` (`[left,top,right,bottom]`), `landmarks` (`[5][2]`), and the raw
or normalized `embedding`. Bounding boxes and landmarks use a one-pixel
tolerance; embedding cosine must be at least 0.999. These external files are
never committed by the project.
