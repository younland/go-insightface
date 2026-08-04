package runtimefs

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	stdfs "io/fs"
	"os"
	"path/filepath"
	"reflect"
)

const (
	cacheDirPerm  = 0o700
	cacheFilePerm = 0o600
)

var (
	ErrNilFS       = errors.New("runtimefs: fs is nil")
	ErrNilContext  = errors.New("runtimefs: context is nil")
	ErrInvalidPath = errors.New("runtimefs: invalid path")
)

type FS interface {
	stdfs.FS
	Materialize(context.Context, string) (string, error)
}

func New(base stdfs.FS, cacheDir string) (FS, error) {
	if isNilFS(base) {
		return nil, ErrNilFS
	}

	resolvedCacheDir, err := resolveCacheDir(cacheDir)
	if err != nil {
		return nil, err
	}

	return &cachedFS{
		base:     base,
		cacheDir: resolvedCacheDir,
	}, nil
}

type cachedFS struct {
	base     stdfs.FS
	cacheDir string
}

func (c *cachedFS) Open(name string) (stdfs.File, error) {
	return c.base.Open(name)
}

func (c *cachedFS) Materialize(ctx context.Context, name string) (string, error) {
	if ctx == nil {
		return "", ErrNilContext
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if !stdfs.ValidPath(name) {
		return "", fmt.Errorf("%q: %w", name, ErrInvalidPath)
	}

	if err := ensureDir(c.cacheDir); err != nil {
		return "", err
	}

	source, err := c.base.Open(name)
	if err != nil {
		return "", err
	}
	defer source.Close()

	info, err := source.Stat()
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%q: %w", name, ErrInvalidPath)
	}

	tempFile, err := os.CreateTemp(c.cacheDir, ".tmp-*")
	if err != nil {
		return "", err
	}
	tempPath := tempFile.Name()
	defer func() {
		_ = tempFile.Close()
		_ = os.Remove(tempPath)
	}()

	if err := tempFile.Chmod(cacheFilePerm); err != nil {
		return "", err
	}

	digest := sha256.New()
	if err := copyWithContext(ctx, io.MultiWriter(tempFile, digest), source); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := tempFile.Close(); err != nil {
		return "", err
	}

	sum := hex.EncodeToString(digest.Sum(nil))
	targetDir := filepath.Join(c.cacheDir, sum[:2])
	if err := ensureDir(targetDir); err != nil {
		return "", err
	}
	targetPath := filepath.Join(targetDir, sum)

	matches, matchErr := fileMatchesDigest(ctx, targetPath, sum)
	if matchErr == nil && matches {
		return targetPath, nil
	}
	if matchErr != nil && !errors.Is(matchErr, os.ErrNotExist) {
		return "", matchErr
	}
	if matchErr == nil && !matches {
		if err := os.Remove(targetPath); err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}

	if err := os.Rename(tempPath, targetPath); err != nil {
		if matches, verifyErr := fileMatchesDigest(ctx, targetPath, sum); verifyErr == nil && matches {
			return targetPath, nil
		}
		return "", err
	}
	tempPath = ""
	return targetPath, nil
}

func fileMatchesDigest(ctx context.Context, path, expected string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	digest := sha256.New()
	if err := copyWithContext(ctx, digest, file); err != nil {
		return false, err
	}
	return hex.EncodeToString(digest.Sum(nil)) == expected, nil
}

func copyWithContext(ctx context.Context, dst io.Writer, src io.Reader) error {
	buffer := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		n, readErr := src.Read(buffer)
		if n > 0 {
			if _, err := dst.Write(buffer[:n]); err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			return readErr
		}
	}
}

func ensureDir(path string) error {
	if err := os.MkdirAll(path, cacheDirPerm); err != nil {
		return err
	}
	return os.Chmod(path, cacheDirPerm)
}

func resolveCacheDir(cacheDir string) (string, error) {
	if cacheDir == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		cacheDir = filepath.Join(userCacheDir, "go-insightface", "runtimefs")
	}
	return filepath.Abs(cacheDir)
}

func isNilFS(base stdfs.FS) bool {
	if base == nil {
		return true
	}

	value := reflect.ValueOf(base)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
