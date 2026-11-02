//go:build integration

package ort

import (
	"context"
	"os"
	"testing"
)

func TestRuntimeExecutesONNXModel(t *testing.T) {
	library := os.Getenv("ORT_LIBRARY")
	model := os.Getenv("ORT_TEST_MODEL")
	if library == "" || model == "" {
		t.Skip("ORT_LIBRARY and ORT_TEST_MODEL are required")
	}
	runtime, err := Open(library)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()

	modelFile, err := os.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()

	session, err := runtime.LoadReader(model, modelFile, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("available providers: %v", runtime.AvailableProviders())
	t.Logf("session providers: %v", session.Providers())
	outputs, err := session.Run(context.Background(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []int64{1, 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 || len(outputs[0].Data) != 3 {
		t.Fatalf("outputs = %+v, want one 3-value tensor", outputs)
	}
}
