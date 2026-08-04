package runtimefs

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	stdfs "io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"
)

func TestNewRejectsNilFS(t *testing.T) {
	_, err := New(nil, t.TempDir())
	if !errors.Is(err, ErrNilFS) {
		t.Fatalf("New(nil) error = %v, want %v", err, ErrNilFS)
	}
}

func TestMaterializeMapFSUsesContentAddressedCache(t *testing.T) {
	cacheDir := t.TempDir()
	source := fstest.MapFS{
		"models/detector.onnx": {Data: []byte("detector-bytes")},
	}

	wrapped, err := New(source, cacheDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	file, err := wrapped.Open("models/detector.onnx")
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer file.Close()

	path, err := wrapped.Materialize(context.Background(), "models/detector.onnx")
	if err != nil {
		t.Fatalf("Materialize() error = %v", err)
	}

	digest := sha256.Sum256(source["models/detector.onnx"].Data)
	wantPath := filepath.Join(cacheDir, hex.EncodeToString(digest[:])[:2], hex.EncodeToString(digest[:]))
	if path != wantPath {
		t.Fatalf("Materialize() path = %q, want %q", path, wantPath)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", path, err)
	}
	if string(got) != "detector-bytes" {
		t.Fatalf("materialized bytes = %q, want %q", string(got), "detector-bytes")
	}

	rootInfo, err := os.Stat(cacheDir)
	if err != nil {
		t.Fatalf("os.Stat(cacheDir) error = %v", err)
	}
	if rootInfo.Mode().Perm() != 0o700 {
		t.Fatalf("cacheDir permissions = %#o, want %#o", rootInfo.Mode().Perm(), 0o700)
	}

	subdirInfo, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatalf("os.Stat(subdir) error = %v", err)
	}
	if subdirInfo.Mode().Perm() != 0o700 {
		t.Fatalf("cache subdir permissions = %#o, want %#o", subdirInfo.Mode().Perm(), 0o700)
	}
}

func TestMaterializeUsesDefaultUserCacheDirWhenEmpty(t *testing.T) {
	userCacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", userCacheDir)

	wrapped, err := New(fstest.MapFS{
		"runtime/libonnxruntime.so": {Data: []byte("runtime-lib")},
	}, "")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	path, err := wrapped.Materialize(context.Background(), "runtime/libonnxruntime.so")
	if err != nil {
		t.Fatalf("Materialize() error = %v", err)
	}

	wantPrefix := filepath.Join(userCacheDir, "go-insightface", "runtimefs") + string(os.PathSeparator)
	if !strings.HasPrefix(path, wantPrefix) {
		t.Fatalf("Materialize() path = %q, want prefix %q", path, wantPrefix)
	}
}

func TestMaterializeReusesCachedPath(t *testing.T) {
	cacheDir := t.TempDir()
	wrapped, err := New(fstest.MapFS{
		"runtime/libonnxruntime.so": {Data: []byte("runtime-lib")},
	}, cacheDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	firstPath, err := wrapped.Materialize(context.Background(), "runtime/libonnxruntime.so")
	if err != nil {
		t.Fatalf("first Materialize() error = %v", err)
	}
	firstInfo, err := os.Stat(firstPath)
	if err != nil {
		t.Fatalf("os.Stat(firstPath) error = %v", err)
	}

	time.Sleep(10 * time.Millisecond)

	secondPath, err := wrapped.Materialize(context.Background(), "runtime/libonnxruntime.so")
	if err != nil {
		t.Fatalf("second Materialize() error = %v", err)
	}
	secondInfo, err := os.Stat(secondPath)
	if err != nil {
		t.Fatalf("os.Stat(secondPath) error = %v", err)
	}

	if firstPath != secondPath {
		t.Fatalf("Materialize() paths differ: %q vs %q", firstPath, secondPath)
	}
	if !firstInfo.ModTime().Equal(secondInfo.ModTime()) {
		t.Fatalf("cache file modtime changed: %v vs %v", firstInfo.ModTime(), secondInfo.ModTime())
	}

	files := collectRegularFiles(t, cacheDir)
	if len(files) != 1 {
		t.Fatalf("regular cache files = %v, want exactly 1", files)
	}
}

func TestMaterializeReplacesCorruptedCacheEntry(t *testing.T) {
	cacheDir := t.TempDir()
	want := []byte("trusted-runtime")
	wrapped, err := New(fstest.MapFS{
		"runtime/libonnxruntime.so": {Data: want},
	}, cacheDir)
	if err != nil {
		t.Fatal(err)
	}

	path, err := wrapped.Materialize(context.Background(), "runtime/libonnxruntime.so")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("corrupted"), 0o600); err != nil {
		t.Fatal(err)
	}

	gotPath, err := wrapped.Materialize(context.Background(), "runtime/libonnxruntime.so")
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != path {
		t.Fatalf("Materialize() path = %q, want %q", gotPath, path)
	}
	got, err := os.ReadFile(gotPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("materialized bytes = %q, want %q", got, want)
	}
}

func TestMaterializeConcurrentCallsDoNotCorruptCache(t *testing.T) {
	cacheDir := t.TempDir()
	wrapped, err := New(fstest.MapFS{
		"models/recognizer.onnx": {Data: []byte(strings.Repeat("recognizer", 4096))},
	}, cacheDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	const goroutines = 16
	results := make([]string, goroutines)
	errs := make([]error, goroutines)

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = wrapped.Materialize(context.Background(), "models/recognizer.onnx")
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("Materialize() goroutine %d error = %v", i, err)
		}
		if results[i] != results[0] {
			t.Fatalf("Materialize() path[%d] = %q, want %q", i, results[i], results[0])
		}
	}

	got, err := os.ReadFile(results[0])
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", results[0], err)
	}
	if string(got) != strings.Repeat("recognizer", 4096) {
		t.Fatalf("materialized bytes were corrupted")
	}

	_ = filepath.WalkDir(cacheDir, func(path string, d stdfs.DirEntry, err error) error {
		if err != nil {
			t.Fatalf("WalkDir(%q) error = %v", path, err)
		}
		if !d.IsDir() && strings.HasPrefix(filepath.Base(path), ".tmp-") {
			t.Fatalf("temporary file leaked: %q", path)
		}
		return nil
	})
}

func TestMaterializeRespectsCancellationAndCleansUp(t *testing.T) {
	cacheDir := t.TempDir()
	fs := &slowFS{
		name:        "runtime/libonnxruntime.so",
		data:        []byte(strings.Repeat("runtime", 2048)),
		chunkSize:   64,
		readDelay:   10 * time.Millisecond,
		readStarted: make(chan struct{}),
	}

	wrapped, err := New(fs, cacheDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		_, err := wrapped.Materialize(ctx, "runtime/libonnxruntime.so")
		errCh <- err
	}()

	<-fs.readStarted
	cancel()

	err = <-errCh
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Materialize() error = %v, want %v", err, context.Canceled)
	}

	files := collectRegularFiles(t, cacheDir)
	if len(files) != 0 {
		t.Fatalf("cache contains leftover files after cancellation: %v", files)
	}
}

func TestMaterializeRejectsInvalidPathAndNilContext(t *testing.T) {
	wrapped, err := New(fstest.MapFS{
		"runtime/libonnxruntime.so": {Data: []byte("runtime-lib")},
	}, t.TempDir())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = wrapped.Materialize(context.Background(), "../escape")
	if !errors.Is(err, ErrInvalidPath) {
		t.Fatalf("Materialize(invalid path) error = %v, want %v", err, ErrInvalidPath)
	}

	_, err = wrapped.Materialize(nil, "runtime/libonnxruntime.so")
	if !errors.Is(err, ErrNilContext) {
		t.Fatalf("Materialize(nil context) error = %v, want %v", err, ErrNilContext)
	}
}

func collectRegularFiles(t *testing.T, root string) []string {
	t.Helper()

	files := make([]string, 0)
	if err := filepath.WalkDir(root, func(path string, d stdfs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			files = append(files, path)
		}
		return nil
	}); err != nil {
		t.Fatalf("WalkDir(%q) error = %v", root, err)
	}
	return files
}

type slowFS struct {
	name        string
	data        []byte
	chunkSize   int
	readDelay   time.Duration
	readStarted chan struct{}
}

func (s *slowFS) Open(name string) (stdfs.File, error) {
	if name != s.name {
		return nil, stdfs.ErrNotExist
	}
	return &slowFile{
		reader:      strings.NewReader(string(s.data)),
		size:        int64(len(s.data)),
		chunkSize:   s.chunkSize,
		readDelay:   s.readDelay,
		readStarted: s.readStarted,
	}, nil
}

type slowFile struct {
	reader      *strings.Reader
	size        int64
	chunkSize   int
	readDelay   time.Duration
	readStarted chan struct{}
	once        sync.Once
}

func (s *slowFile) Stat() (stdfs.FileInfo, error) {
	return fileInfo{name: "runtime/libonnxruntime.so", size: s.size}, nil
}

func (s *slowFile) Read(p []byte) (int, error) {
	s.once.Do(func() {
		close(s.readStarted)
	})
	time.Sleep(s.readDelay)
	if len(p) > s.chunkSize {
		p = p[:s.chunkSize]
	}
	return s.reader.Read(p)
}

func (s *slowFile) Close() error {
	return nil
}

type fileInfo struct {
	name string
	size int64
}

func (f fileInfo) Name() string         { return f.name }
func (f fileInfo) Size() int64          { return f.size }
func (f fileInfo) Mode() stdfs.FileMode { return 0o644 }
func (f fileInfo) ModTime() time.Time   { return time.Unix(0, 0) }
func (f fileInfo) IsDir() bool          { return false }
func (f fileInfo) Sys() any             { return nil }
