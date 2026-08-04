package assets

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	cacheDirPerm      = 0o700
	cacheFilePerm     = 0o600
	outputDirPerm     = 0o755
	outputFilePerm    = 0o644
	copyBufferSize    = 32 * 1024
	maxArtifactBytes  = 1 << 30
	officialORTVer    = "1.23.2"
	officialORTPrefix = "https://github.com/microsoft/onnxruntime/releases/download/v" + officialORTVer + "/"
)

var (
	// ErrNilContext reports a nil context passed to Download.
	ErrNilContext = errors.New("assets: context is nil")
	// ErrOutputDirRequired reports an empty output directory.
	ErrOutputDirRequired = errors.New("assets: output dir is required")
	// ErrCacheInsideOutput reports a cache nested in the managed output directory.
	ErrCacheInsideOutput = errors.New("assets: cache dir must not be inside output dir")
	// ErrUnsupportedPlatform reports that no official runtime is known for a target.
	ErrUnsupportedPlatform = errors.New("assets: unsupported platform")
	// ErrInvalidArtifact reports an invalid URL, digest, or artifact definition.
	ErrInvalidArtifact = errors.New("assets: invalid artifact")
	// ErrInvalidArchiveMember reports an unsafe or unsupported archive member.
	ErrInvalidArchiveMember = errors.New("assets: invalid archive member")
	// ErrHashMismatch reports downloaded bytes that do not match the declared digest.
	ErrHashMismatch = errors.New("assets: sha256 mismatch")
	// ErrHTTPStatus reports a non-success response from an artifact server.
	ErrHTTPStatus = errors.New("assets: unexpected http status")
	// ErrHTTPTransport reports a failure before a valid HTTP response was received.
	ErrHTTPTransport = errors.New("assets: HTTP transport failed")
	// ErrArtifactTooLarge reports an artifact exceeding the safety limit.
	ErrArtifactTooLarge = errors.New("assets: artifact exceeds size limit")
	// ErrUnsupportedArchive reports an archive other than ZIP or TAR.GZ.
	ErrUnsupportedArchive = errors.New("assets: unsupported archive format")
)

type Artifact struct {
	URL           string `json:"url"`
	SHA256        string `json:"sha256"`
	ArchiveMember string `json:"archiveMember,omitempty"`
}

type Bundle struct {
	Runtime    Artifact `json:"runtime"`
	Detector   Artifact `json:"detector"`
	Recognizer Artifact `json:"recognizer"`
}

type Options struct {
	OutputDir string       `json:"outputDir"`
	CacheDir  string       `json:"cacheDir,omitempty"`
	GOOS      string       `json:"goos,omitempty"`
	GOARCH    string       `json:"goarch,omitempty"`
	Client    *http.Client `json:"-"`
}

type Paths struct {
	RuntimeLibrary  string `json:"runtimeLibrary"`
	DetectorModel   string `json:"detectorModel"`
	RecognizerModel string `json:"recognizerModel"`
}

// Download fetches and verifies a complete asset bundle, then replaces
// Options.OutputDir as one coherent unit. OutputDir must be exclusively
// managed by Download. CacheDir must not be OutputDir or one of its children.
func Download(ctx context.Context, bundle Bundle, options Options) (Paths, error) {
	if ctx == nil {
		return Paths{}, ErrNilContext
	}
	if err := ctx.Err(); err != nil {
		return Paths{}, err
	}

	outputDir, err := resolveOutputDir(options.OutputDir)
	if err != nil {
		return Paths{}, err
	}
	cacheDir, err := resolveCacheDir(options.CacheDir)
	if err != nil {
		return Paths{}, err
	}
	if pathWithin(outputDir, cacheDir) {
		return Paths{}, ErrCacheInsideOutput
	}
	if err := os.MkdirAll(filepath.Dir(outputDir), outputDirPerm); err != nil {
		return Paths{}, err
	}
	if err := recoverInterruptedPublish(ctx, outputDir); err != nil {
		return Paths{}, fmt.Errorf("recover asset bundle: %w", err)
	}

	goos := options.GOOS
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := options.GOARCH
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	if isZeroArtifact(bundle.Runtime) {
		bundle.Runtime, err = OfficialONNXRuntime(goos, goarch)
		if err != nil {
			return Paths{}, err
		}
	}

	detector, err := normalizeArtifact(bundle.Detector)
	if err != nil {
		return Paths{}, fmt.Errorf("detector: %w", err)
	}
	recognizer, err := normalizeArtifact(bundle.Recognizer)
	if err != nil {
		return Paths{}, fmt.Errorf("recognizer: %w", err)
	}
	runtimeArtifact, err := normalizeArtifact(bundle.Runtime)
	if err != nil {
		return Paths{}, fmt.Errorf("runtime: %w", err)
	}

	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}

	dl := downloader{
		ctx:      ctx,
		client:   client,
		cacheDir: cacheDir,
		ensured:  make(map[string]string),
	}

	runtimeBase, err := runtimeBasename(runtimeArtifact)
	if err != nil {
		return Paths{}, err
	}

	stagingDir, err := os.MkdirTemp(filepath.Dir(outputDir), "."+filepath.Base(outputDir)+".staging-*")
	if err != nil {
		return Paths{}, err
	}
	defer os.RemoveAll(stagingDir)

	stagingPaths := Paths{
		RuntimeLibrary:  filepath.Join(stagingDir, "runtime", runtimeBase),
		DetectorModel:   filepath.Join(stagingDir, "models", "detector.onnx"),
		RecognizerModel: filepath.Join(stagingDir, "models", "recognizer.onnx"),
	}

	if err := dl.materialize(runtimeArtifact, stagingPaths.RuntimeLibrary); err != nil {
		return Paths{}, fmt.Errorf("runtime: %w", err)
	}
	if err := dl.materialize(detector, stagingPaths.DetectorModel); err != nil {
		return Paths{}, fmt.Errorf("detector: %w", err)
	}
	if err := dl.materialize(recognizer, stagingPaths.RecognizerModel); err != nil {
		return Paths{}, fmt.Errorf("recognizer: %w", err)
	}
	if err := replaceOutputDir(ctx, stagingDir, outputDir); err != nil {
		return Paths{}, fmt.Errorf("commit asset bundle: %w", err)
	}
	stagingDir = ""

	return Paths{
		RuntimeLibrary:  filepath.Join(outputDir, "runtime", runtimeBase),
		DetectorModel:   filepath.Join(outputDir, "models", "detector.onnx"),
		RecognizerModel: filepath.Join(outputDir, "models", "recognizer.onnx"),
	}, nil
}

// OfficialONNXRuntime returns the pinned ONNX Runtime artifact for a platform.
func OfficialONNXRuntime(goos, goarch string) (Artifact, error) {
	key := goos + "/" + goarch
	artifact, ok := officialRuntimeArtifacts[key]
	if !ok {
		return Artifact{}, fmt.Errorf("%s/%s: %w", goos, goarch, ErrUnsupportedPlatform)
	}
	return artifact, nil
}

type downloader struct {
	ctx      context.Context
	client   *http.Client
	cacheDir string
	ensured  map[string]string
}

func (d *downloader) materialize(artifact Artifact, targetPath string) error {
	cachePath, err := d.ensureCached(artifact)
	if err != nil {
		return err
	}
	if artifact.ArchiveMember == "" {
		return copyFileAtomic(d.ctx, cachePath, targetPath, maxArtifactBytes)
	}
	return extractArchiveMemberAtomic(d.ctx, cachePath, artifact.ArchiveMember, targetPath)
}

func (d *downloader) ensureCached(artifact Artifact) (string, error) {
	if existing, ok := d.ensured[artifact.SHA256]; ok {
		return existing, nil
	}
	if err := ensureDir(d.cacheDir, cacheDirPerm); err != nil {
		return "", err
	}

	targetPath := filepath.Join(d.cacheDir, artifact.SHA256)
	ok, err := verifyCachedFile(d.ctx, targetPath, artifact.SHA256)
	if err == nil && ok {
		d.ensured[artifact.SHA256] = targetPath
		return targetPath, nil
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	if !ok && err == nil {
		if removeErr := os.Remove(targetPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			return "", removeErr
		}
	}

	if err := d.downloadToCache(artifact, targetPath); err != nil {
		return "", err
	}
	d.ensured[artifact.SHA256] = targetPath
	return targetPath, nil
}

func (d *downloader) downloadToCache(artifact Artifact, targetPath string) error {
	req, err := http.NewRequestWithContext(d.ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return err
	}

	resp, err := d.client.Do(req)
	if err != nil {
		if contextErr := d.ctx.Err(); contextErr != nil {
			return contextErr
		}
		return fmt.Errorf("GET %q: %w", safeURL(artifact.URL), ErrHTTPTransport)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s %q: %w", resp.Status, safeURL(artifact.URL), ErrHTTPStatus)
	}
	if resp.ContentLength > maxArtifactBytes {
		return fmt.Errorf("%q: %w", safeURL(artifact.URL), ErrArtifactTooLarge)
	}

	tmp, err := os.CreateTemp(d.cacheDir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := tmp.Chmod(cacheFilePerm); err != nil {
		return err
	}

	digest := sha256.New()
	written, err := copyWithLimit(d.ctx, io.MultiWriter(tmp, digest), resp.Body, maxArtifactBytes)
	if err != nil {
		return err
	}
	if resp.ContentLength >= 0 && written != resp.ContentLength {
		return fmt.Errorf("downloaded %d bytes, expected %d", written, resp.ContentLength)
	}

	got := hex.EncodeToString(digest.Sum(nil))
	if got != artifact.SHA256 {
		return fmt.Errorf("got %s want %s: %w", got, artifact.SHA256, ErrHashMismatch)
	}

	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, targetPath); err != nil {
		ok, verifyErr := verifyCachedFile(d.ctx, targetPath, artifact.SHA256)
		if verifyErr == nil && ok {
			return nil
		}
		return err
	}
	tmpPath = ""
	return nil
}

func copyFileAtomic(ctx context.Context, srcPath, dstPath string, limit int64) error {
	source, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer source.Close()

	info, err := source.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("%q: %w", srcPath, ErrInvalidArtifact)
	}

	return writeAtomic(dstPath, func(dst *os.File) error {
		_, err := copyWithLimit(ctx, dst, source, limit)
		return err
	})
}

func extractArchiveMemberAtomic(ctx context.Context, archivePath, member, targetPath string) error {
	if err := validateArchiveMember(member); err != nil {
		return err
	}

	file, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer file.Close()

	header := make([]byte, 4)
	n, err := io.ReadFull(file, header)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
		return err
	}
	header = header[:n]

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}

	switch {
	case isZipMagic(header):
		return extractZipMemberAtomic(ctx, file, member, targetPath)
	case isGzipMagic(header):
		return extractTarGzMemberAtomic(ctx, file, member, targetPath)
	default:
		return fmt.Errorf("%q: %w", archivePath, ErrUnsupportedArchive)
	}
}

func extractZipMemberAtomic(ctx context.Context, file *os.File, member, targetPath string) error {
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if info.Size() > maxArtifactBytes {
		return fmt.Errorf("%q: %w", file.Name(), ErrArtifactTooLarge)
	}

	reader, err := zip.NewReader(file, info.Size())
	if err != nil {
		return err
	}

	var match *zip.File
	for _, entry := range reader.File {
		if canonicalArchiveName(entry.Name) != member {
			continue
		}
		if match != nil {
			return fmt.Errorf("%q: duplicate archive member", member)
		}
		if entry.FileInfo().IsDir() || !entry.Mode().IsRegular() {
			return fmt.Errorf("%q: %w", member, ErrInvalidArchiveMember)
		}
		if entry.UncompressedSize64 > maxArtifactBytes {
			return fmt.Errorf("%q: %w", member, ErrArtifactTooLarge)
		}
		match = entry
	}
	if match == nil {
		return fmt.Errorf("%q: archive member not found", member)
	}

	rc, err := match.Open()
	if err != nil {
		return err
	}
	defer rc.Close()

	return writeAtomic(targetPath, func(dst *os.File) error {
		n, err := copyWithLimit(ctx, dst, rc, maxArtifactBytes)
		if err != nil {
			return err
		}
		if n != int64(match.UncompressedSize64) {
			return fmt.Errorf("%q: truncated archive member", member)
		}
		return nil
	})
}

func extractTarGzMemberAtomic(ctx context.Context, file *os.File, member, targetPath string) error {
	gz, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gz.Close()

	return writeAtomic(targetPath, func(dst *os.File) error {
		tr := tar.NewReader(&boundedContextReader{
			ctx:       ctx,
			reader:    gz,
			remaining: maxArtifactBytes,
		})
		var matched bool
		for {
			header, err := tr.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return err
			}
			if header.Size < 0 || header.Size > maxArtifactBytes {
				return fmt.Errorf("%q: %w", header.Name, ErrArtifactTooLarge)
			}
			if canonicalArchiveName(header.Name) != member {
				continue
			}
			if matched {
				return fmt.Errorf("%q: duplicate archive member", member)
			}
			if header.Typeflag != tar.TypeReg && header.Typeflag != tar.TypeRegA {
				return fmt.Errorf("%q: %w", member, ErrInvalidArchiveMember)
			}
			n, err := copyWithLimit(ctx, dst, io.LimitReader(tr, header.Size), maxArtifactBytes)
			if err != nil {
				return err
			}
			if n != header.Size {
				return fmt.Errorf("%q: truncated archive member", member)
			}
			matched = true
		}
		if !matched {
			return fmt.Errorf("%q: archive member not found", member)
		}
		return nil
	})
}

type boundedContextReader struct {
	ctx       context.Context
	reader    io.Reader
	remaining int64
}

func (r *boundedContextReader) Read(buffer []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	if r.remaining == 0 {
		var probe [1]byte
		count, err := r.reader.Read(probe[:])
		if count > 0 {
			return 0, ErrArtifactTooLarge
		}
		return 0, err
	}
	if int64(len(buffer)) > r.remaining {
		buffer = buffer[:r.remaining]
	}
	count, err := r.reader.Read(buffer)
	r.remaining -= int64(count)
	if contextErr := r.ctx.Err(); contextErr != nil {
		return count, contextErr
	}
	return count, err
}

func writeAtomic(targetPath string, writeFn func(*os.File) error) error {
	dir := filepath.Dir(targetPath)
	if err := ensureDir(dir, outputDirPerm); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := tmp.Chmod(outputFilePerm); err != nil {
		return err
	}
	if err := writeFn(tmp); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return err
	}
	tmpPath = ""
	return nil
}

func verifyCachedFile(ctx context.Context, pathValue, wantSHA string) (bool, error) {
	file, err := os.Open(pathValue)
	if err != nil {
		return false, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return false, err
	}
	if !info.Mode().IsRegular() {
		return false, fmt.Errorf("%q: %w", pathValue, ErrInvalidArtifact)
	}
	if info.Size() > maxArtifactBytes {
		return false, nil
	}

	digest := sha256.New()
	if _, err := copyWithLimit(ctx, digest, file, maxArtifactBytes); err != nil {
		return false, err
	}
	return hex.EncodeToString(digest.Sum(nil)) == wantSHA, nil
}

func copyWithLimit(ctx context.Context, dst io.Writer, src io.Reader, limit int64) (int64, error) {
	buffer := make([]byte, copyBufferSize)
	var total int64

	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}

		n, readErr := src.Read(buffer)
		if n > 0 {
			total += int64(n)
			if total > limit {
				return total, ErrArtifactTooLarge
			}
			if _, err := dst.Write(buffer[:n]); err != nil {
				return total, err
			}
			if err := ctx.Err(); err != nil {
				return total, err
			}
		}

		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return total, nil
			}
			return total, readErr
		}
	}
}

func normalizeArtifact(artifact Artifact) (Artifact, error) {
	artifact.URL = strings.TrimSpace(artifact.URL)
	artifact.SHA256 = strings.ToLower(strings.TrimSpace(artifact.SHA256))
	artifact.ArchiveMember = strings.TrimSpace(artifact.ArchiveMember)

	if artifact.URL == "" || artifact.SHA256 == "" {
		return Artifact{}, ErrInvalidArtifact
	}
	parsed, err := url.ParseRequestURI(artifact.URL)
	if err != nil || parsed.Scheme != "http" && parsed.Scheme != "https" || parsed.Host == "" {
		return Artifact{}, fmt.Errorf("%q: %w", safeURL(artifact.URL), ErrInvalidArtifact)
	}
	sum, err := hex.DecodeString(artifact.SHA256)
	if err != nil || len(sum) != sha256.Size {
		return Artifact{}, fmt.Errorf("%q: %w", artifact.SHA256, ErrInvalidArtifact)
	}
	if artifact.ArchiveMember != "" {
		if err := validateArchiveMember(artifact.ArchiveMember); err != nil {
			return Artifact{}, err
		}
	}
	return artifact, nil
}

func validateArchiveMember(member string) error {
	if !fs.ValidPath(member) {
		return fmt.Errorf("%q: %w", member, ErrInvalidArchiveMember)
	}
	if path.Base(member) == "." || path.Base(member) == "/" {
		return fmt.Errorf("%q: %w", member, ErrInvalidArchiveMember)
	}
	return nil
}

func runtimeBasename(artifact Artifact) (string, error) {
	if artifact.ArchiveMember != "" {
		base := path.Base(artifact.ArchiveMember)
		if base == "." || base == "/" || base == "" {
			return "", fmt.Errorf("%q: %w", artifact.ArchiveMember, ErrInvalidArchiveMember)
		}
		return base, nil
	}

	parsed, err := url.Parse(artifact.URL)
	if err != nil {
		return "", err
	}
	base := path.Base(parsed.Path)
	if base == "." || base == "/" || base == "" {
		return "", fmt.Errorf("%q: %w", safeURL(artifact.URL), ErrInvalidArtifact)
	}
	return base, nil
}

func resolveOutputDir(outputDir string) (string, error) {
	if strings.TrimSpace(outputDir) == "" {
		return "", ErrOutputDirRequired
	}
	return filepath.Abs(outputDir)
}

func resolveCacheDir(cacheDir string) (string, error) {
	if strings.TrimSpace(cacheDir) == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		cacheDir = filepath.Join(userCacheDir, "go-insightface", "downloads")
	}
	return filepath.Abs(cacheDir)
}

func ensureDir(dir string, perm os.FileMode) error {
	if err := os.MkdirAll(dir, perm); err != nil {
		return err
	}
	return os.Chmod(dir, perm)
}

func replaceOutputDir(ctx context.Context, stagingDir, outputDir string) error {
	release, err := acquireOutputLock(ctx, outputDir+".lock")
	if err != nil {
		return err
	}
	defer release()

	if err := recoverOutputDir(outputDir); err != nil {
		return err
	}
	backupDir := outputDir + ".previous"

	hadOutput := true
	if err := os.Rename(outputDir, backupDir); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		hadOutput = false
	}
	if err := os.Rename(stagingDir, outputDir); err != nil {
		if hadOutput {
			if rollbackErr := os.Rename(backupDir, outputDir); rollbackErr != nil {
				return errors.Join(err, fmt.Errorf("restore previous output: %w", rollbackErr))
			}
		}
		return err
	}
	if hadOutput {
		return os.RemoveAll(backupDir)
	}
	return nil
}

func recoverInterruptedPublish(ctx context.Context, outputDir string) error {
	release, err := acquireOutputLock(ctx, outputDir+".lock")
	if err != nil {
		return err
	}
	defer release()
	return recoverOutputDir(outputDir)
}

func recoverOutputDir(outputDir string) error {
	backupDir := outputDir + ".previous"
	_, outputErr := os.Stat(outputDir)
	_, backupErr := os.Stat(backupDir)

	switch {
	case outputErr == nil:
		if backupErr == nil {
			return os.RemoveAll(backupDir)
		}
		if !errors.Is(backupErr, os.ErrNotExist) {
			return backupErr
		}
		return nil
	case !errors.Is(outputErr, os.ErrNotExist):
		return outputErr
	case backupErr == nil:
		return os.Rename(backupDir, outputDir)
	case errors.Is(backupErr, os.ErrNotExist):
		return nil
	default:
		return backupErr
	}
}

func pathWithin(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	if err != nil {
		return false
	}
	return rel == "." || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func canonicalArchiveName(name string) string {
	if trimmed, ok := strings.CutPrefix(name, "./"); ok {
		return trimmed
	}
	return name
}

func isZeroArtifact(artifact Artifact) bool {
	return strings.TrimSpace(artifact.URL) == "" &&
		strings.TrimSpace(artifact.SHA256) == "" &&
		strings.TrimSpace(artifact.ArchiveMember) == ""
}

func isZipMagic(header []byte) bool {
	return len(header) >= 4 &&
		header[0] == 'P' &&
		header[1] == 'K' &&
		(header[2] == 3 && header[3] == 4 ||
			header[2] == 5 && header[3] == 6 ||
			header[2] == 7 && header[3] == 8)
}

func isGzipMagic(header []byte) bool {
	return len(header) >= 2 && header[0] == 0x1f && header[1] == 0x8b
}

func safeURL(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return strings.SplitN(raw, "?", 2)[0]
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}

var officialRuntimeArtifacts = map[string]Artifact{
	"linux/amd64": {
		URL:           officialORTPrefix + "onnxruntime-linux-x64-" + officialORTVer + ".tgz",
		SHA256:        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
		ArchiveMember: "onnxruntime-linux-x64-" + officialORTVer + "/lib/libonnxruntime.so." + officialORTVer,
	},
	"linux/arm64": {
		URL:           officialORTPrefix + "onnxruntime-linux-aarch64-" + officialORTVer + ".tgz",
		SHA256:        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
		ArchiveMember: "onnxruntime-linux-aarch64-" + officialORTVer + "/lib/libonnxruntime.so." + officialORTVer,
	},
	"darwin/arm64": {
		URL:           officialORTPrefix + "onnxruntime-osx-arm64-" + officialORTVer + ".tgz",
		SHA256:        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
		ArchiveMember: "onnxruntime-osx-arm64-" + officialORTVer + "/lib/libonnxruntime." + officialORTVer + ".dylib",
	},
	"darwin/amd64": {
		URL:           officialORTPrefix + "onnxruntime-osx-universal2-" + officialORTVer + ".tgz",
		SHA256:        "49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809",
		ArchiveMember: "onnxruntime-osx-universal2-" + officialORTVer + "/lib/libonnxruntime." + officialORTVer + ".dylib",
	},
}
