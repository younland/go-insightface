//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package assets

import (
	"fmt"
	"os"
	"runtime"
)

func tryLockFile(file *os.File) error {
	return fmt.Errorf("output locking on %s: %w", runtime.GOOS, ErrUnsupportedPlatform)
}

func unlockFile(file *os.File) error {
	return nil
}
