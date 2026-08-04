package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunRequiresManifestAndOutput(t *testing.T) {
	err := run(context.Background(), nil, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("run() error = %v, want required flags error", err)
	}
}

func TestReadManifestRejectsUnknownAndTrailingFields(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{name: "unknown", json: `{"unknown":true}`},
		{name: "trailing", json: `{}` + "\n" + `{}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "manifest.json")
			if err := os.WriteFile(path, []byte(test.json), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := readManifest(path); err == nil {
				t.Fatal("readManifest() error = nil")
			}
		})
	}
}

func TestReadManifestRejectsOversizedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, bytes.Repeat([]byte("x"), maxManifestBytes+1), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readManifest(path); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("readManifest() error = %v, want size error", err)
	}
}

func TestRunRejectsNonPositiveTimeout(t *testing.T) {
	err := run(t.Context(), []string{
		"-manifest", "unused.json",
		"-output", t.TempDir(),
		"-timeout", "0s",
	}, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("run() error = %v, want timeout error", err)
	}
}
