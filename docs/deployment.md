# Deployment and operational guidance

## Runtime

The current backend depends on
`github.com/younland/onnxruntime`, pinned in `go.mod`. It requires Go 1.27
or newer and negotiates the C API with ONNX Runtime 1.23 through 1.31 at
startup. Supply an explicit shared-library path at
startup, or opt into the build/download and caller-embedding mechanisms in
[asset provisioning](assets.md).

Execution providers are selected automatically: on every session load the
adapter queries the library's available providers and enables TensorRT
first, then CUDA (using the V2 provider-options API, default settings), so
GPU builds use maximum compute without configuration. If session creation
with GPU providers fails (for example a GPU build without CUDA drivers), the
adapter logs the failure and retries on CPU; CPU-only builds are unaffected.
Use a CPU build of ONNX Runtime to force CPU inference.

Sessions are created with `session.intra_op.allow_spinning=0` by default so
idle sessions do not keep the intra-op thread pool spinning on a CPU core.
`Config.SessionConfig` passes additional session configuration entries
through `AddSessionConfigEntry`; caller entries override defaults per key.

The upstream wrapper is untagged and its API is unstable. It is isolated
behind `internal/ort`; applications should use only this module's public API.
Request cancellation is checked before and after native inference, but ONNX
Runtime execution itself cannot currently be interrupted through the upstream
wrapper.

When embedding assets, allow the process to create its private user cache. The
runtime shared library must exist as a real file for the operating-system
dynamic loader; embedded models remain streams. Build and test a distinct
bundle for each target OS and architecture.

## Model licensing

This repository is MIT-licensed code only. Model weights have independent
licenses. In particular, the models distributed by the InsightFace project
are described as non-commercial research models. Do not use those weights in
military, government, commercial, or production deployments without an
explicit license that covers the intended use. Use separately licensed or
internally trained weights when necessary.

## Biometric data

Face images and embeddings are biometric data. Encrypt them at rest and in
transit, restrict access, audit enrollment and lookup operations, define a
retention/deletion policy, and avoid logging image bytes or embeddings.

## Threshold calibration

No universal similarity threshold represents a probability. Calibrate the
minimum cosine similarity and the Top-1/Top-2 identity margin on data matching
the deployed cameras, demographics, lighting, pose, compression, and expected
gallery size. For a 3,000-person gallery, report FPIR and FNIR at the
identity-search level, not only pairwise verification accuracy.

## Embedding migration

Give every stored embedding a model/version identifier. When the model
fingerprint changes, regenerate embeddings through an idempotent enrollment
API from retained source images. During migration, keep versioned galleries
separate and switch atomically after coverage and quality checks pass. If the
source image is unavailable, the old embedding cannot be converted into the
new model's embedding space and must be re-enrolled.
