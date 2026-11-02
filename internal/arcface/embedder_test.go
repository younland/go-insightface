package arcface

import (
	"context"
	"errors"
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/younland/go-insightface/internal/imageutil"
	"github.com/younland/go-insightface/internal/ort"
)

func TestEmbedderEmbedAlignsPreprocessesAndNormalizes(t *testing.T) {
	runner := &fakeRunner{
		outputs: []ort.Tensor{
			{
				Name:  "embedding",
				Shape: []int64{1, embeddingDim},
				Data:  rawEmbedding(embeddingDim),
			},
		},
	}

	embedder, err := NewEmbedder(runner)
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}

	src := image.NewRGBA(image.Rect(0, 0, 130, 130))
	for y := 0; y < 130; y++ {
		for x := 0; x < 130; x++ {
			src.SetRGBA(x, y, color.RGBA{
				R: uint8(x),
				G: uint8(y),
				B: uint8(x + y),
				A: 255,
			})
		}
	}

	landmarks := translatedTemplate(10, 15)
	values, err := embedder.Embed(context.Background(), src, landmarks)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if got, want := len(runner.gotShape), 4; got != want {
		t.Fatalf("runner shape rank = %d, want %d", got, want)
	}
	wantShape := []int64{1, 3, chipSize, chipSize}
	for i := range wantShape {
		if runner.gotShape[i] != wantShape[i] {
			t.Fatalf("runner shape[%d] = %d, want %d", i, runner.gotShape[i], wantShape[i])
		}
	}

	plane := chipSize * chipSize
	wantR := normPixel(10)
	wantG := normPixel(15)
	wantB := normPixel(25)
	if diff := math.Abs(float64(runner.gotInput[0] - wantR)); diff > 1e-6 {
		t.Fatalf("runner input R(0,0) = %f, want %f", runner.gotInput[0], wantR)
	}
	if diff := math.Abs(float64(runner.gotInput[plane] - wantG)); diff > 1e-6 {
		t.Fatalf("runner input G(0,0) = %f, want %f", runner.gotInput[plane], wantG)
	}
	if diff := math.Abs(float64(runner.gotInput[2*plane] - wantB)); diff > 1e-6 {
		t.Fatalf("runner input B(0,0) = %f, want %f", runner.gotInput[2*plane], wantB)
	}

	if got, want := len(values), embeddingDim; got != want {
		t.Fatalf("Embed() length = %d, want %d", got, want)
	}

	var sumSquares float64
	for _, value := range values {
		sumSquares += float64(value * value)
	}
	if diff := math.Abs(math.Sqrt(sumSquares) - 1); diff > 1e-6 {
		t.Fatalf("embedding norm = %f, want 1", math.Sqrt(sumSquares))
	}
	if diff := math.Abs(float64(values[0] - 0.6)); diff > 1e-6 {
		t.Fatalf("embedding[0] = %f, want 0.6", values[0])
	}
	if diff := math.Abs(float64(values[1] - 0.8)); diff > 1e-6 {
		t.Fatalf("embedding[1] = %f, want 0.8", values[1])
	}

	values[0] = 0
	if runner.outputs[0].Data[0] != 3 {
		t.Fatalf("Embed() returned a slice aliasing runner output")
	}
}

func TestEmbedderEmbedRejectsWrongOutputShape(t *testing.T) {
	runner := &fakeRunner{
		outputs: []ort.Tensor{
			{
				Name:  "embedding",
				Shape: []int64{1, embeddingDim - 1},
				Data:  make([]float32, embeddingDim-1),
			},
		},
	}

	embedder, err := NewEmbedder(runner)
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}

	src := image.NewRGBA(image.Rect(0, 0, chipSize, chipSize))
	landmarks := canonicalTemplate
	_, err = embedder.Embed(context.Background(), src, landmarks)
	if !errors.Is(err, ErrInvalidOutput) {
		t.Fatalf("Embed() error = %v, want %v", err, ErrInvalidOutput)
	}
}

func TestExtractEmbeddingAcceptsFlatOutput(t *testing.T) {
	raw := rawEmbedding(embeddingDim)
	values, err := extractEmbedding([]ort.Tensor{{
		Name: "embedding", Shape: []int64{embeddingDim}, Data: raw,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != embeddingDim || values[0] != 3 || values[1] != 4 {
		t.Fatalf("embedding = %v", values[:2])
	}
	values[0] = 0
	if raw[0] != 3 {
		t.Fatal("extractEmbedding returned an alias")
	}
}

func TestEmbedderEmbedRejectsNilImage(t *testing.T) {
	embedder, err := NewEmbedder(&fakeRunner{})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}

	_, err = embedder.Embed(context.Background(), nil, canonicalTemplate)
	if !errors.Is(err, imageutil.ErrInvalidDimensions) {
		t.Fatalf("Embed() error = %v, want %v", err, imageutil.ErrInvalidDimensions)
	}
}

func TestEmbedderEmbedRejectsInvalidLandmark(t *testing.T) {
	embedder, err := NewEmbedder(&fakeRunner{})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}

	landmarks := canonicalTemplate
	landmarks[0].X = math.NaN()

	src := image.NewRGBA(image.Rect(0, 0, chipSize, chipSize))
	_, err = embedder.Embed(context.Background(), src, landmarks)
	if !errors.Is(err, ErrInvalidLandmarks) {
		t.Fatalf("Embed() error = %v, want %v", err, ErrInvalidLandmarks)
	}
}

type fakeRunner struct {
	gotInput []float32
	gotShape []int64
	outputs  []ort.Tensor
	runErr   error
}

func (r *fakeRunner) Run(_ context.Context, input []float32, shape []int64) ([]ort.Tensor, error) {
	r.gotInput = append([]float32(nil), input...)
	r.gotShape = append([]int64(nil), shape...)
	if r.runErr != nil {
		return nil, r.runErr
	}
	return r.outputs, nil
}

func translatedTemplate(dx, dy float64) [5]imageutil.Point {
	var points [5]imageutil.Point
	for i, point := range canonicalTemplate {
		points[i] = imageutil.Point{X: point.X + dx, Y: point.Y + dy}
	}
	return points
}

func rawEmbedding(dim int) []float32 {
	data := make([]float32, dim)
	data[0] = 3
	data[1] = 4
	return data
}

func normPixel(value float32) float32 {
	return (value - 127.5) / 127.5
}
