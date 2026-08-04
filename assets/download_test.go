package assets

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDownloadCachesArtifactsAndReusesSharedDownloads(t *testing.T) {
	cacheDir := t.TempDir()
	outputDir := t.TempDir()

	runtimeMember := "onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so.1.23.2"
	runtimeBytes := archiveZip(t, runtimeMember, []byte("runtime-bytes"))
	modelBytes := []byte("shared-model")

	var runtimeHits atomic.Int32
	var modelHits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime.zip":
			runtimeHits.Add(1)
			w.Header().Set("Content-Type", "application/zip")
			_, _ = w.Write(runtimeBytes)
		case "/model.onnx":
			modelHits.Add(1)
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = w.Write(modelBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	bundle := Bundle{
		Runtime: Artifact{
			URL:           server.URL + "/runtime.zip",
			SHA256:        sha256Hex(runtimeBytes),
			ArchiveMember: runtimeMember,
		},
		Detector: Artifact{
			URL:    server.URL + "/model.onnx",
			SHA256: sha256Hex(modelBytes),
		},
		Recognizer: Artifact{
			URL:    server.URL + "/model.onnx",
			SHA256: sha256Hex(modelBytes),
		},
	}
	options := Options{
		OutputDir: outputDir,
		CacheDir:  cacheDir,
		Client:    server.Client(),
	}

	paths, err := Download(context.Background(), bundle, options)
	if err != nil {
		t.Fatalf("Download() error = %v", err)
	}

	wantRuntime := filepath.Join(outputDir, "runtime", "libonnxruntime.so.1.23.2")
	wantDetector := filepath.Join(outputDir, "models", "detector.onnx")
	wantRecognizer := filepath.Join(outputDir, "models", "recognizer.onnx")
	if paths.RuntimeLibrary != wantRuntime {
		t.Fatalf("RuntimeLibrary = %q, want %q", paths.RuntimeLibrary, wantRuntime)
	}
	if paths.DetectorModel != wantDetector {
		t.Fatalf("DetectorModel = %q, want %q", paths.DetectorModel, wantDetector)
	}
	if paths.RecognizerModel != wantRecognizer {
		t.Fatalf("RecognizerModel = %q, want %q", paths.RecognizerModel, wantRecognizer)
	}
	assertFileBytes(t, wantRuntime, []byte("runtime-bytes"))
	assertFileBytes(t, wantDetector, modelBytes)
	assertFileBytes(t, wantRecognizer, modelBytes)

	if got := runtimeHits.Load(); got != 1 {
		t.Fatalf("runtime requests = %d, want 1", got)
	}
	if got := modelHits.Load(); got != 1 {
		t.Fatalf("model requests = %d, want 1", got)
	}

	if _, err := Download(context.Background(), bundle, options); err != nil {
		t.Fatalf("second Download() error = %v", err)
	}
	if got := runtimeHits.Load(); got != 1 {
		t.Fatalf("runtime requests after cache hit = %d, want 1", got)
	}
	if got := modelHits.Load(); got != 1 {
		t.Fatalf("model requests after cache hit = %d, want 1", got)
	}
}

func TestDownloadUsesDefaultCacheDirWhenEmpty(t *testing.T) {
	userCacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", userCacheDir)

	outputDir := t.TempDir()
	runtimeBytes := []byte("runtime-default-cache")
	detectorBytes := []byte("detector-default-cache")
	recognizerBytes := []byte("recognizer-default-cache")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime.bin":
			_, _ = w.Write(runtimeBytes)
		case "/detector.onnx":
			_, _ = w.Write(detectorBytes)
		case "/recognizer.onnx":
			_, _ = w.Write(recognizerBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	_, err := Download(context.Background(), Bundle{
		Runtime: Artifact{
			URL:    server.URL + "/runtime.bin",
			SHA256: sha256Hex(runtimeBytes),
		},
		Detector: Artifact{
			URL:    server.URL + "/detector.onnx",
			SHA256: sha256Hex(detectorBytes),
		},
		Recognizer: Artifact{
			URL:    server.URL + "/recognizer.onnx",
			SHA256: sha256Hex(recognizerBytes),
		},
	}, Options{
		OutputDir: outputDir,
		Client:    server.Client(),
	})
	if err != nil {
		t.Fatalf("Download() error = %v", err)
	}

	defaultCacheDir := filepath.Join(userCacheDir, "go-insightface", "downloads")
	if _, err := os.Stat(filepath.Join(defaultCacheDir, sha256Hex(runtimeBytes))); err != nil {
		t.Fatalf("runtime cache stat error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(defaultCacheDir, sha256Hex(detectorBytes))); err != nil {
		t.Fatalf("detector cache stat error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(defaultCacheDir, sha256Hex(recognizerBytes))); err != nil {
		t.Fatalf("recognizer cache stat error = %v", err)
	}
}

func TestDownloadUsesOfficialRuntimeManifestAndExtractsTarGz(t *testing.T) {
	outputDir := t.TempDir()
	cacheDir := t.TempDir()

	runtimeMember := "onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so.1.23.2"
	runtimeArchive := archiveTarGz(t, runtimeMember, []byte("official-runtime"))
	detectorBytes := []byte("detector")
	recognizerBytes := []byte("recognizer")

	var runtimeHits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime.tgz":
			runtimeHits.Add(1)
			w.Header().Set("Content-Type", "application/gzip")
			_, _ = w.Write(runtimeArchive)
		case "/detector.onnx":
			_, _ = w.Write(detectorBytes)
		case "/recognizer.onnx":
			_, _ = w.Write(recognizerBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	artifact, err := OfficialONNXRuntime("linux", "amd64")
	if err != nil {
		t.Fatalf("OfficialONNXRuntime() error = %v", err)
	}
	if artifact.ArchiveMember != runtimeMember {
		t.Fatalf("ArchiveMember = %q, want %q", artifact.ArchiveMember, runtimeMember)
	}
	if !strings.HasSuffix(artifact.URL, "/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz") {
		t.Fatalf("Official URL = %q, unexpected suffix", artifact.URL)
	}

	paths, err := Download(context.Background(), Bundle{
		Runtime: Artifact{
			URL:           server.URL + "/runtime.tgz",
			SHA256:        sha256Hex(runtimeArchive),
			ArchiveMember: runtimeMember,
		},
		Detector: Artifact{
			URL:    server.URL + "/detector.onnx",
			SHA256: sha256Hex(detectorBytes),
		},
		Recognizer: Artifact{
			URL:    server.URL + "/recognizer.onnx",
			SHA256: sha256Hex(recognizerBytes),
		},
	}, Options{
		OutputDir: outputDir,
		CacheDir:  cacheDir,
		Client:    server.Client(),
	})
	if err != nil {
		t.Fatalf("Download() error = %v", err)
	}

	assertFileBytes(t, paths.RuntimeLibrary, []byte("official-runtime"))
	assertFileBytes(t, paths.DetectorModel, detectorBytes)
	assertFileBytes(t, paths.RecognizerModel, recognizerBytes)
	if runtimeHits.Load() != 1 {
		t.Fatalf("official runtime requests = %d, want 1", runtimeHits.Load())
	}
}

func TestDownloadRejectsIllegalArchiveMember(t *testing.T) {
	outputDir := t.TempDir()
	cacheDir := t.TempDir()

	archiveBytes := archiveZip(t, "ok/file", []byte("payload"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(archiveBytes)
	}))
	defer server.Close()

	_, err := Download(context.Background(), Bundle{
		Runtime: Artifact{
			URL:           server.URL + "/runtime.zip",
			SHA256:        sha256Hex(archiveBytes),
			ArchiveMember: "../escape",
		},
		Detector: Artifact{
			URL:    server.URL + "/detector.onnx",
			SHA256: sha256Hex([]byte("detector")),
		},
		Recognizer: Artifact{
			URL:    server.URL + "/recognizer.onnx",
			SHA256: sha256Hex([]byte("recognizer")),
		},
	}, Options{
		OutputDir: outputDir,
		CacheDir:  cacheDir,
		Client:    server.Client(),
	})
	if err == nil {
		t.Fatal("Download() error = nil, want invalid archive member")
	}
	if !strings.Contains(err.Error(), "archive member") {
		t.Fatalf("Download() error = %v, want archive member error", err)
	}
}

func TestDownloadRejectsHashMismatch(t *testing.T) {
	outputDir := t.TempDir()
	cacheDir := t.TempDir()

	runtimeBytes := []byte("runtime")
	detectorBytes := []byte("detector")
	recognizerBytes := []byte("recognizer")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime.bin":
			_, _ = w.Write(runtimeBytes)
		case "/detector.onnx":
			_, _ = w.Write(detectorBytes)
		case "/recognizer.onnx":
			_, _ = w.Write(recognizerBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	_, err := Download(context.Background(), Bundle{
		Runtime: Artifact{
			URL:    server.URL + "/runtime.bin",
			SHA256: strings.Repeat("0", 64),
		},
		Detector: Artifact{
			URL:    server.URL + "/detector.onnx",
			SHA256: sha256Hex(detectorBytes),
		},
		Recognizer: Artifact{
			URL:    server.URL + "/recognizer.onnx",
			SHA256: sha256Hex(recognizerBytes),
		},
	}, Options{
		OutputDir: outputDir,
		CacheDir:  cacheDir,
		Client:    server.Client(),
	})
	if err == nil {
		t.Fatal("Download() error = nil, want hash mismatch")
	}
	if !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("Download() error = %v, want sha256 mismatch", err)
	}

	assertNoRegularFiles(t, outputDir)
}

func TestDownloadFailurePreservesPreviousBundle(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "bundle")
	cacheDir := filepath.Join(root, "cache")
	oldFiles := map[string][]byte{
		filepath.Join(outputDir, "runtime", "old-runtime.so"): []byte("old-runtime"),
		filepath.Join(outputDir, "models", "detector.onnx"):   []byte("old-detector"),
		filepath.Join(outputDir, "models", "recognizer.onnx"): []byte("old-recognizer"),
	}
	for name, content := range oldFiles {
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	runtimeBytes := []byte("new-runtime")
	detectorBytes := []byte("new-detector")
	recognizerBytes := []byte("new-recognizer")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime":
			_, _ = w.Write(runtimeBytes)
		case "/detector":
			_, _ = w.Write(detectorBytes)
		case "/recognizer":
			_, _ = w.Write(recognizerBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	_, err := Download(t.Context(), Bundle{
		Runtime:    Artifact{URL: server.URL + "/runtime", SHA256: sha256Hex(runtimeBytes)},
		Detector:   Artifact{URL: server.URL + "/detector", SHA256: sha256Hex(detectorBytes)},
		Recognizer: Artifact{URL: server.URL + "/recognizer", SHA256: strings.Repeat("0", 64)},
	}, Options{OutputDir: outputDir, CacheDir: cacheDir, Client: server.Client()})
	if !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("Download() error = %v, want ErrHashMismatch", err)
	}
	for name, content := range oldFiles {
		assertFileBytes(t, name, content)
	}
	assertNoWorkDirs(t, root)
}

func TestDownloadSuccessReplacesPreviousBundle(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "bundle")
	cacheDir := filepath.Join(root, "cache")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stalePath := filepath.Join(outputDir, "stale.txt")
	if err := os.WriteFile(stalePath, []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}

	runtimeBytes := []byte("runtime")
	detectorBytes := []byte("detector")
	recognizerBytes := []byte("recognizer")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/runtime":
			_, _ = w.Write(runtimeBytes)
		case "/detector":
			_, _ = w.Write(detectorBytes)
		case "/recognizer":
			_, _ = w.Write(recognizerBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	paths, err := Download(t.Context(), Bundle{
		Runtime:    Artifact{URL: server.URL + "/runtime", SHA256: sha256Hex(runtimeBytes)},
		Detector:   Artifact{URL: server.URL + "/detector", SHA256: sha256Hex(detectorBytes)},
		Recognizer: Artifact{URL: server.URL + "/recognizer", SHA256: sha256Hex(recognizerBytes)},
	}, Options{OutputDir: outputDir, CacheDir: cacheDir, Client: server.Client()})
	if err != nil {
		t.Fatalf("Download() error = %v", err)
	}
	assertFileBytes(t, paths.RuntimeLibrary, runtimeBytes)
	assertFileBytes(t, paths.DetectorModel, detectorBytes)
	assertFileBytes(t, paths.RecognizerModel, recognizerBytes)
	if _, err := os.Stat(stalePath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale file stat error = %v, want not exist", err)
	}
	assertNoWorkDirs(t, root)
}

func TestDownloadRejectsCacheInsideOutput(t *testing.T) {
	outputDir := t.TempDir()
	_, err := Download(t.Context(), Bundle{}, Options{
		OutputDir: outputDir,
		CacheDir:  filepath.Join(outputDir, "cache"),
	})
	if !errors.Is(err, ErrCacheInsideOutput) {
		t.Fatalf("Download() error = %v, want ErrCacheInsideOutput", err)
	}
}

func TestNormalizeArtifactRequiresHTTPURL(t *testing.T) {
	_, err := normalizeArtifact(Artifact{
		URL:    "file:///tmp/model.onnx",
		SHA256: strings.Repeat("0", 64),
	})
	if !errors.Is(err, ErrInvalidArtifact) {
		t.Fatalf("normalizeArtifact() error = %v, want ErrInvalidArtifact", err)
	}
}

func TestExtractArchiveMemberAcceptsDotSlashPrefix(t *testing.T) {
	member := "bundle/lib/runtime.so"
	content := []byte("runtime")
	tests := map[string][]byte{
		"zip":    archiveZip(t, "./"+member, content),
		"tar.gz": archiveTarGz(t, "./"+member, content),
	}
	for name, archive := range tests {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			archivePath := filepath.Join(root, "archive")
			targetPath := filepath.Join(root, "runtime.so")
			if err := os.WriteFile(archivePath, archive, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := extractArchiveMemberAtomic(t.Context(), archivePath, member, targetPath); err != nil {
				t.Fatalf("extractArchiveMemberAtomic() error = %v", err)
			}
			assertFileBytes(t, targetPath, content)
		})
	}
}

func TestReplaceOutputDirSerializesConcurrentPublishers(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "bundle")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "marker"), []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}

	stagingDirs := make([]string, 2)
	for i, marker := range []string{"first", "second"} {
		stagingDir, err := os.MkdirTemp(root, ".bundle.staging-*")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(stagingDir, "marker"), []byte(marker), 0o644); err != nil {
			t.Fatal(err)
		}
		stagingDirs[i] = stagingDir
	}

	start := make(chan struct{})
	errs := make([]error, len(stagingDirs))
	var wg sync.WaitGroup
	for i, stagingDir := range stagingDirs {
		wg.Go(func() {
			<-start
			errs[i] = replaceOutputDir(t.Context(), stagingDir, outputDir)
		})
	}
	close(start)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("replaceOutputDir() publisher %d error = %v", i, err)
		}
	}
	marker, err := os.ReadFile(filepath.Join(outputDir, "marker"))
	if err != nil {
		t.Fatal(err)
	}
	if string(marker) != "first" && string(marker) != "second" {
		t.Fatalf("published marker = %q, want a complete staged bundle", marker)
	}
	assertNoWorkDirs(t, root)
}

func TestRecoverInterruptedPublishRestoresPreviousBundle(t *testing.T) {
	root := t.TempDir()
	outputDir := filepath.Join(root, "bundle")
	backupDir := outputDir + ".previous"
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(backupDir, "marker"), []byte("previous"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := recoverInterruptedPublish(t.Context(), outputDir); err != nil {
		t.Fatalf("recoverInterruptedPublish() error = %v", err)
	}
	assertFileBytes(t, filepath.Join(outputDir, "marker"), []byte("previous"))
	if _, err := os.Stat(backupDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("backup stat error = %v, want not exist", err)
	}
}

func TestDownloadRedactsTransportErrorURL(t *testing.T) {
	rawURL := "https://user:password@example.invalid/model.onnx?token=secret#fragment"
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		return nil, errors.New("transport exposed " + request.URL.String())
	})}
	dl := downloader{
		ctx:      t.Context(),
		client:   client,
		cacheDir: t.TempDir(),
		ensured:  make(map[string]string),
	}
	err := dl.downloadToCache(Artifact{
		URL:    rawURL,
		SHA256: strings.Repeat("0", 64),
	}, filepath.Join(t.TempDir(), "cache-entry"))
	if !errors.Is(err, ErrHTTPTransport) {
		t.Fatalf("downloadToCache() error = %v, want ErrHTTPTransport", err)
	}
	for _, secret := range []string{"user", "password", "token", "secret", "fragment"} {
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("downloadToCache() error leaked %q: %v", secret, err)
		}
	}
}

func TestOfficialONNXRuntimeRejectsUnsupportedPlatform(t *testing.T) {
	_, err := OfficialONNXRuntime("windows", "amd64")
	if err == nil {
		t.Fatal("OfficialONNXRuntime() error = nil, want unsupported platform")
	}
}

func TestDownloadRespectsCancellation(t *testing.T) {
	outputDir := t.TempDir()
	cacheDir := t.TempDir()

	started := make(chan struct{})
	var once sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() { close(started) })
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		chunk := bytes.Repeat([]byte("a"), 16*1024)
		for {
			if _, err := w.Write(chunk); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-started
		time.Sleep(15 * time.Millisecond)
		cancel()
	}()

	artifact := Artifact{
		URL:    server.URL + "/artifact.bin",
		SHA256: sha256Hex(bytes.Repeat([]byte("a"), 16*1024)),
	}
	_, err := Download(ctx, Bundle{
		Runtime:    artifact,
		Detector:   artifact,
		Recognizer: artifact,
	}, Options{
		OutputDir: outputDir,
		CacheDir:  cacheDir,
		Client:    server.Client(),
	})
	if err == nil {
		t.Fatal("Download() error = nil, want cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Download() error = %v, want context.Canceled", err)
	}

	assertNoTempFiles(t, cacheDir)
	assertNoTempFiles(t, outputDir)
	assertNoRegularFiles(t, outputDir)
}

func TestDownloadDefaultsPlatformFromRuntime(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("current platform is unsupported by official runtime manifest")
	}

	goos := runtime.GOOS
	goarch := runtime.GOARCH
	artifact, err := OfficialONNXRuntime(goos, goarch)
	if err != nil {
		t.Fatalf("OfficialONNXRuntime(%q, %q) error = %v", goos, goarch, err)
	}
	if artifact.URL == "" || artifact.SHA256 == "" || artifact.ArchiveMember == "" {
		t.Fatalf("OfficialONNXRuntime(%q, %q) returned incomplete artifact: %+v", goos, goarch, artifact)
	}
}

func archiveZip(t *testing.T, name string, content []byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create(name)
	if err != nil {
		t.Fatalf("zip.Create(%q) error = %v", name, err)
	}
	if _, err := w.Write(content); err != nil {
		t.Fatalf("zip.Write(%q) error = %v", name, err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip.Close() error = %v", err)
	}
	return buf.Bytes()
}

func archiveTarGz(t *testing.T, name string, content []byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(zw)
	header := &tar.Header{
		Name: name,
		Mode: 0o600,
		Size: int64(len(content)),
	}
	if err := tw.WriteHeader(header); err != nil {
		t.Fatalf("tar.WriteHeader(%q) error = %v", name, err)
	}
	if _, err := tw.Write(content); err != nil {
		t.Fatalf("tar.Write(%q) error = %v", name, err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar.Close() error = %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("gzip.Close() error = %v", err)
	}
	return buf.Bytes()
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func assertFileBytes(t *testing.T, path string, want []byte) {
	t.Helper()

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("file %q bytes = %q, want %q", path, string(got), string(want))
	}
}

func assertNoTempFiles(t *testing.T, root string) {
	t.Helper()

	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			t.Fatalf("WalkDir(%q) error = %v", path, err)
		}
		if !d.IsDir() && strings.HasPrefix(filepath.Base(path), ".tmp-") {
			t.Fatalf("temporary file leaked: %q", path)
		}
		return nil
	})
}

func assertNoRegularFiles(t *testing.T, root string) {
	t.Helper()

	var files []string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			t.Fatalf("WalkDir(%q) error = %v", path, err)
		}
		if path == root || d.IsDir() {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if len(files) != 0 {
		t.Fatalf("regular files under %q = %v, want none", root, files)
	}
}

func assertNoWorkDirs(t *testing.T, root string) {
	t.Helper()

	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.Contains(entry.Name(), ".staging-") || strings.HasSuffix(entry.Name(), ".previous") {
			t.Fatalf("work directory leaked: %q", entry.Name())
		}
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}
