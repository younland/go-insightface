package arcface

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/lib-x/go-insightface/internal/imageutil"
	"github.com/lib-x/go-insightface/internal/ort"
)

const (
	chipSize     = 112
	embeddingDim = 512
)

var (
	ErrNilRunner             = errors.New("arcface: runner is nil")
	ErrInvalidLandmarks      = errors.New("arcface: invalid landmarks")
	ErrInvalidOutput         = errors.New("arcface: invalid embedding output")
	ErrEmptyEmbedding        = errors.New("arcface: embedding is empty")
	ErrZeroNormEmbedding     = errors.New("arcface: embedding has zero L2 norm")
	ErrInvalidEmbeddingValue = errors.New("arcface: embedding contains non-finite value")
)

// Runner executes a single ONNX inference step for the ArcFace embedding model.
type Runner interface {
	Run(context.Context, []float32, []int64) ([]ort.Tensor, error)
}

// Embedder aligns a face chip, preprocesses it, runs the embedding model, and
// returns a copied, L2-normalized embedding vector.
type Embedder struct {
	runner Runner
}

// canonicalTemplate is the standard InsightFace 5-point template for 112x112
// ArcFace alignment.
var canonicalTemplate = [5]imageutil.Point{
	{X: 38.2946, Y: 51.6963},
	{X: 73.5318, Y: 51.5014},
	{X: 56.0252, Y: 71.7366},
	{X: 41.5493, Y: 92.3655},
	{X: 70.7299, Y: 92.2041},
}

// NewEmbedder wires an ArcFace runner into the preprocessing and normalization pipeline.
func NewEmbedder(runner Runner) (*Embedder, error) {
	if runner == nil {
		return nil, ErrNilRunner
	}
	return &Embedder{runner: runner}, nil
}

// Embed aligns the input face using the standard 5-point template, converts the
// aligned chip to RGB NCHW with (pixel-127.5)/127.5 normalization, runs the
// session, and returns a normalized embedding vector.
func (e *Embedder) Embed(ctx context.Context, src image.Image, landmarks [5]imageutil.Point) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e == nil || e.runner == nil {
		return nil, ErrNilRunner
	}

	aligned, err := align(src, landmarks)
	if err != nil {
		return nil, err
	}

	input, err := imageutil.ToNCHW(aligned, imageutil.PixelTransform{
		Mean:  [3]float32{127.5, 127.5, 127.5},
		Scale: [3]float32{1 / 127.5, 1 / 127.5, 1 / 127.5},
	})
	if err != nil {
		return nil, fmt.Errorf("preprocess face: %w", err)
	}

	outputs, err := e.runner.Run(ctx, input, []int64{1, 3, chipSize, chipSize})
	if err != nil {
		return nil, fmt.Errorf("run embedding model: %w", err)
	}

	raw, err := extractEmbedding(outputs)
	if err != nil {
		return nil, err
	}

	values, err := normalizeEmbedding(raw)
	if err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return values, nil
}

func align(src image.Image, landmarks [5]imageutil.Point) (*image.RGBA, error) {
	if src == nil {
		return nil, imageutil.ErrInvalidDimensions
	}
	if err := validateLandmarks(landmarks); err != nil {
		return nil, err
	}

	transform, err := imageutil.EstimateSimilarity(landmarks, canonicalTemplate)
	if err != nil {
		return nil, fmt.Errorf("%w: estimate similarity: %v", ErrInvalidLandmarks, err)
	}

	aligned, err := imageutil.WarpAffine(src, transform, chipSize, chipSize)
	if err != nil {
		return nil, fmt.Errorf("warp aligned face: %w", err)
	}

	return aligned, nil
}

func validateLandmarks(landmarks [5]imageutil.Point) error {
	for i, point := range landmarks {
		if !finite(point.X) || !finite(point.Y) {
			return fmt.Errorf("%w: landmark %d contains non-finite coordinates", ErrInvalidLandmarks, i)
		}
	}
	return nil
}

func extractEmbedding(outputs []ort.Tensor) ([]float32, error) {
	if len(outputs) != 1 {
		return nil, fmt.Errorf("%w: expected 1 output tensor, got %d", ErrInvalidOutput, len(outputs))
	}

	tensor := outputs[0]
	if err := tensor.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidOutput, err)
	}
	validShape := len(tensor.Shape) == 1 && tensor.Shape[0] == embeddingDim
	validShape = validShape || len(tensor.Shape) == 2 && tensor.Shape[0] == 1 && tensor.Shape[1] == embeddingDim
	if !validShape {
		return nil, fmt.Errorf("%w: expected shape [%d] or [1,%d], got %v", ErrInvalidOutput, embeddingDim, embeddingDim, tensor.Shape)
	}
	if len(tensor.Data) != embeddingDim {
		return nil, fmt.Errorf("%w: expected %d values, got %d", ErrInvalidOutput, embeddingDim, len(tensor.Data))
	}

	values := make([]float32, embeddingDim)
	copy(values, tensor.Data)
	return values, nil
}

func normalizeEmbedding(raw []float32) ([]float32, error) {
	if len(raw) == 0 {
		return nil, ErrEmptyEmbedding
	}

	values := make([]float32, len(raw))
	var sumSquares float64
	for i, value := range raw {
		if !finite(float64(value)) {
			return nil, fmt.Errorf("%w: value at index %d", ErrInvalidEmbeddingValue, i)
		}
		values[i] = value
		sumSquares += float64(value) * float64(value)
	}

	if sumSquares == 0 {
		return nil, ErrZeroNormEmbedding
	}

	invNorm := float32(1 / math.Sqrt(sumSquares))
	for i := range values {
		values[i] *= invNorm
	}

	return values, nil
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
