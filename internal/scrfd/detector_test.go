package scrfd

import (
	"context"
	"errors"
	"image"
	"image/color"
	"testing"

	"github.com/lib-x/go-insightface/internal/ort"
)

type recordingSession struct {
	input   []float32
	shape   []int64
	outputs []ort.Tensor
	err     error
}

func (s *recordingSession) Run(_ context.Context, input []float32, shape []int64) ([]ort.Tensor, error) {
	s.input = append([]float32(nil), input...)
	s.shape = append([]int64(nil), shape...)
	return s.outputs, s.err
}

func TestDetectorPreprocessesRunsAndRescales(t *testing.T) {
	session := &recordingSession{outputs: detectorOutputs(32, 32)}
	detector, err := NewDetector(session, DecoderConfig{
		InputWidth: 32, InputHeight: 32, Strides: []int{8, 16, 32},
		ScoreThreshold: 0.5, NMSThreshold: 0.4,
	})
	if err != nil {
		t.Fatal(err)
	}
	src := image.NewRGBA(image.Rect(0, 0, 16, 8))
	for y := range 8 {
		for x := range 16 {
			src.SetRGBA(x, y, color.RGBA{R: 255, G: 127, A: 255})
		}
	}

	detections, err := detector.Detect(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	if len(detections) != 1 {
		t.Fatalf("len(detections) = %d, want 1", len(detections))
	}
	if got, want := session.shape, []int64{1, 3, 32, 32}; !equalInt64s(got, want) {
		t.Fatalf("input shape = %v, want %v", got, want)
	}
	if got := session.input[0]; got != (255-127.5)/128 {
		t.Fatalf("red input = %v", got)
	}
	if got := session.input[32*32]; got != (127-127.5)/128 {
		t.Fatalf("green input = %v", got)
	}
	if detections[0].Box != (Box{Left: 0, Top: 0, Right: 8, Bottom: 8}) {
		t.Fatalf("box = %+v", detections[0].Box)
	}
}

func TestDetectorRejectsInvalidImage(t *testing.T) {
	detector, err := NewDetector(&recordingSession{}, DecoderConfig{
		InputWidth: 32, InputHeight: 32, Strides: []int{8, 16, 32},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := detector.Detect(context.Background(), nil); !errors.Is(err, ErrInvalidImage) {
		t.Fatalf("Detect() error = %v, want %v", err, ErrInvalidImage)
	}
}

func detectorOutputs(width, height int) []ort.Tensor {
	outputs := make([]ort.Tensor, 0, 9)
	for _, channels := range []int{1, 4, 10} {
		for _, stride := range []int{8, 16, 32} {
			rows := (width / stride) * (height / stride) * 2
			outputs = append(outputs, ort.Tensor{
				Name: "opaque", Shape: []int64{int64(rows), int64(channels)}, Data: make([]float32, rows*channels),
			})
		}
	}
	// Stride 8, row 10 -> cell 5 -> center (8, 8). Distances are in stride units.
	outputs[0].Data[10] = 0.9
	copy(outputs[3].Data[10*4:], []float32{1, 1, 1, 1})
	return outputs
}

func equalInt64s(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
