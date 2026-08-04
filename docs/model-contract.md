# Model contract

`go-insightface` does not bundle or download model weights. Applications must
provide their own licensed ONNX models and record their fingerprints in the
deployment configuration.

## Detector

The detector must be SCRFD-compatible:

- one RGB `NCHW` float32 input;
- input width and height configured by the caller;
- feature strides 8, 16, and 32;
- two anchors at each feature-map location;
- score, four-coordinate bounding-box distance, and five-point landmark
  distance outputs for every stride;
- float32 outputs with layouts recognized unambiguously from their shapes.

Outputs are classified by shape instead of model-specific output names. An
unknown or ambiguous layout is rejected rather than decoded heuristically.

## Recognizer

The recognizer must be ArcFace-compatible:

- one RGB float32 input shaped `[1, 3, 112, 112]`;
- the standard InsightFace five-point alignment template;
- pixel normalization `(channel - 127.5) / 127.5`;
- one float32 embedding output shaped `[1, D]` or `[D]`.

The output dimension is validated and the returned embedding is always L2
normalized. Similarity values are cosine similarities, not probabilities.

## Fingerprints

For a production rollout, pin and audit at least:

- model filename and SHA-256 digest;
- training data provenance and usage license;
- input/output tensor shapes;
- expected embedding dimension;
- ONNX Runtime version and execution provider;
- validation dataset, threshold, margin, FPIR, and FNIR results.

Changing a recognition model changes the embedding space. Existing gallery
embeddings must then be regenerated from source face images; old and new
embeddings must not be compared or mixed in one gallery.
