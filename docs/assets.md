# Asset provisioning

`go-insightface` deliberately ships code only. It provides mechanisms for an
application to obtain or embed ONNX Runtime and model assets, but the module
itself contains no shared libraries, ONNX models, download side effects, or
embedded asset payloads.

## Choose a mode

### Ordinary file paths

Download and deploy the runtime and models with the application's existing
packaging system, then use the original constructor:

```go
engine, err := insightface.New(insightface.Config{
    RuntimeLibrary:  paths.RuntimeLibrary,
    DetectorModel:   paths.DetectorModel,
    RecognizerModel: paths.RecognizerModel,
})
```

This path is unchanged from v0.1.0 and has no dependency on `runtimefs`.

### Build-stage download

The `insightface-assets` command downloads each artifact into a SHA-256 keyed
cache, verifies it, extracts one declared ZIP or TAR.GZ member when requested,
and prepares a complete staging bundle. Only after all three artifacts succeed
does it replace the managed output directory. Publishers targeting the same
directory are serialized with an adjacent lock file. Ordinary publish errors
attempt to restore the previous bundle.

Create `insightface-assets.json`:

```json
{
  "detector": {
    "url": "https://models.example/detector.onnx",
    "sha256": "<detector sha256>"
  },
  "recognizer": {
    "url": "https://models.example/recognizer.onnx",
    "sha256": "<recognizer sha256>"
  }
}
```

The runtime entry is optional. If absent, the command uses its pinned ONNX
Runtime 1.23.2 manifest for `linux/amd64`, `linux/arm64`, `darwin/amd64`, or
`darwin/arm64`. To use an internal mirror or a different compatible build,
declare all runtime fields:

```json
{
  "runtime": {
    "url": "https://artifacts.example/onnxruntime-linux-x64.tgz",
    "sha256": "<archive sha256>",
    "archiveMember": "onnxruntime/lib/libonnxruntime.so.1.23.2"
  },
  "detector": {
    "url": "https://models.example/detector.onnx",
    "sha256": "<detector sha256>"
  },
  "recognizer": {
    "url": "https://models.example/recognizer.onnx",
    "sha256": "<recognizer sha256>"
  }
}
```

Run it directly or from a build step:

```bash
go run github.com/lib-x/go-insightface/cmd/insightface-assets@v0.2.0 \
  -manifest insightface-assets.json \
  -output ./faceassets/files \
  -cache /var/cache/my-product/face-assets \
  -timeout 30m
```

`-os` and `-arch` select a cross-build target. The command prints the three
resolved output paths as JSON. `-output` is an exclusively managed directory;
do not put unrelated files there. The cache must not be inside it. The tool
may leave an empty `<output>.lock` file beside the directory for future
cross-process locking; `<output>.previous` is reserved for interrupted-publish
recovery.

This is a build and packaging tool, not a live hot-update mechanism. Keeping
the stable v0.1-style paths requires a two-rename directory swap, so the output
name can be briefly unavailable. A process or machine crash in that window is
recovered on the next publish attempt. Do not run the command against assets
being opened by a live application.

Cross-process publish locking is implemented on Linux, macOS, and BSD hosts.
Other build hosts return `assets.ErrUnsupportedPlatform` instead of risking a
stale ad-hoc lock; use a supported build container when cross-packaging there.

The same operation is available as `assets.Download` for custom build tools.
The library API takes a caller-owned context and optional `http.Client`; it
does not impose its own timeout.

### Caller-owned embedding

First populate a directory during the build, then embed that directory in the
application:

```go
package faceassets

import "embed"

//go:generate go run github.com/lib-x/go-insightface/cmd/insightface-assets@v0.2.0 -manifest ../insightface-assets.json -output files

//go:embed files
var Files embed.FS
```

Wrap the embedded filesystem only where the application chooses this mode:

```go
source, err := runtimefs.New(faceassets.Files, "")
if err != nil {
    return err
}
engine, err := insightface.NewFromFS(ctx, insightface.Config{
    RuntimeLibrary:  "files/runtime/libonnxruntime.so.1.23.2",
    DetectorModel:   "files/models/detector.onnx",
    RecognizerModel: "files/models/recognizer.onnx",
}, source)
```

The model streams are read without persisting model files. The runtime library
is materialized at first use into
`os.UserCacheDir()/go-insightface/runtimefs/<hash-prefix>/<sha256>` with private
directory and file permissions. Existing cache bytes are hash-verified before
reuse.

Embedding commonly adds roughly 200–300 MB to the executable, depending on
the selected assets. Build one application binary per target platform and
embed the matching runtime library. A single embedded dynamic library is not
portable across operating systems or architectures.

## Integrity and licensing boundaries

- Downloads require HTTP or HTTPS and an exact SHA-256 supplied by the caller
  or the pinned official runtime manifest.
- URLs are redacted in errors so user information, queries, and fragments are
  not exposed.
- Archives extract one exact, safe member and reject traversal, duplicates,
  unsupported types, and artifacts over 1 GiB.
- Model URLs are never selected automatically. The application owner must
  choose weights licensed for the intended military, government, commercial,
  or production use.
- Download integrity is not a license grant and does not establish model
  accuracy. Calibrate FPIR/FNIR on representative deployment data.
