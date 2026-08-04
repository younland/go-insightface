package assets

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"
)

var errOutputLockBusy = errors.New("assets: output lock is busy")

func acquireOutputLock(ctx context.Context, lockPath string) (func(), error) {
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, cacheFilePerm)
	if err != nil {
		return nil, err
	}
	for {
		err := tryLockFile(file)
		if err == nil {
			return sync.OnceFunc(func() {
				_ = unlockFile(file)
				_ = file.Close()
			}), nil
		}
		if !errors.Is(err, errOutputLockBusy) {
			_ = file.Close()
			return nil, err
		}

		select {
		case <-ctx.Done():
			_ = file.Close()
			return nil, ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
}
