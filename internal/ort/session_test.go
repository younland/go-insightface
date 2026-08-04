package ort

import (
	"errors"
	"os"
	"testing"
)

func TestTensorValidate(t *testing.T) {
	tests := []struct {
		name    string
		tensor  Tensor
		wantErr bool
	}{
		{name: "valid", tensor: Tensor{Name: "output", Shape: []int64{1, 2, 3}, Data: make([]float32, 6)}},
		{name: "empty name", tensor: Tensor{Shape: []int64{1}, Data: []float32{1}}, wantErr: true},
		{name: "negative dimension", tensor: Tensor{Name: "x", Shape: []int64{1, -1}, Data: []float32{1}}, wantErr: true},
		{name: "wrong data length", tensor: Tensor{Name: "x", Shape: []int64{2, 2}, Data: []float32{1}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.tensor.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestElementCountRejectsOverflow(t *testing.T) {
	if _, err := elementCount([]int64{1 << 62, 8}); err == nil {
		t.Fatal("expected overflow error")
	}
}

func TestLoadReaderRejectsNilReader(t *testing.T) {
	var runtime Runtime

	_, err := runtime.LoadReader("fixture.onnx", nil, 1)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "model reader is nil" {
		t.Fatalf("LoadReader() error = %v, want model reader is nil", err)
	}
}

func TestLoadReaderReturnsClosedError(t *testing.T) {
	runtime := &Runtime{closed: true}
	_, err := runtime.LoadReader("fixture.onnx", os.Stdin, 1)
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("LoadReader() error = %v, want %v", err, ErrClosed)
	}
}
