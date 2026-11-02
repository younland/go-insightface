// Command insightface-assets downloads verified ONNX Runtime and model assets
// during an application's build or packaging stage.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/younland/go-insightface/assets"
)

const (
	maxManifestBytes = 1 << 20
	defaultTimeout   = 30 * time.Minute
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	if ctx == nil {
		return errors.New("context is nil")
	}
	flags := flag.NewFlagSet("insightface-assets", flag.ContinueOnError)
	flags.SetOutput(stderr)
	manifestPath := flags.String("manifest", "", "path to the JSON asset manifest")
	outputDir := flags.String("output", "", "directory for verified runtime and model files")
	cacheDir := flags.String("cache", "", "download cache directory (optional)")
	goos := flags.String("os", runtime.GOOS, "target operating system")
	goarch := flags.String("arch", runtime.GOARCH, "target architecture")
	timeout := flags.Duration("timeout", defaultTimeout, "maximum download duration")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if *manifestPath == "" || *outputDir == "" {
		return errors.New("-manifest and -output are required")
	}
	if *timeout <= 0 {
		return errors.New("-timeout must be greater than zero")
	}

	manifest, err := readManifest(*manifestPath)
	if err != nil {
		return err
	}
	downloadCtx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	paths, err := assets.Download(downloadCtx, manifest, assets.Options{
		OutputDir: *outputDir,
		CacheDir:  *cacheDir,
		GOOS:      *goos,
		GOARCH:    *goarch,
	})
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(paths)
}

func readManifest(path string) (assets.Bundle, error) {
	file, err := os.Open(path)
	if err != nil {
		return assets.Bundle{}, fmt.Errorf("open manifest: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return assets.Bundle{}, fmt.Errorf("stat manifest: %w", err)
	}
	if info.Size() > maxManifestBytes {
		return assets.Bundle{}, fmt.Errorf("manifest exceeds %d bytes", maxManifestBytes)
	}

	decoder := json.NewDecoder(io.LimitReader(file, maxManifestBytes))
	decoder.DisallowUnknownFields()
	var manifest assets.Bundle
	if err := decoder.Decode(&manifest); err != nil {
		return assets.Bundle{}, fmt.Errorf("decode manifest: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return assets.Bundle{}, errors.New("decode manifest: multiple JSON values")
		}
		return assets.Bundle{}, fmt.Errorf("decode manifest: %w", err)
	}
	return manifest, nil
}
