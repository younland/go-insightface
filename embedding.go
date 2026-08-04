package insightface

import (
	"errors"
	"fmt"
	"math"
)

var (
	ErrEmptyEmbedding             = errors.New("insightface: embedding is empty")
	ErrZeroNormEmbedding          = errors.New("insightface: embedding has zero L2 norm")
	ErrInvalidEmbeddingValue      = errors.New("insightface: embedding contains non-finite value")
	ErrEmbeddingDimensionMismatch = errors.New("insightface: embedding dimension mismatch")
)

// Embedding is an immutable, L2-normalized face embedding.
type Embedding struct {
	values []float32
}

// NewEmbedding validates, copies, and L2-normalizes a raw embedding vector.
func NewEmbedding(raw []float32) (Embedding, error) {
	if len(raw) == 0 {
		return Embedding{}, ErrEmptyEmbedding
	}

	values := make([]float32, len(raw))
	var sumSquares float64
	for i, value := range raw {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return Embedding{}, fmt.Errorf("embedding value at index %d: %w", i, ErrInvalidEmbeddingValue)
		}
		values[i] = value
		sumSquares += float64(value) * float64(value)
	}

	if sumSquares == 0 {
		return Embedding{}, ErrZeroNormEmbedding
	}

	invNorm := float32(1 / math.Sqrt(sumSquares))
	for i := range values {
		values[i] *= invNorm
	}

	return Embedding{values: values}, nil
}

// Dimension returns the embedding dimensionality.
func (e Embedding) Dimension() int {
	return len(e.values)
}

// Values returns a copy of the normalized embedding values.
func (e Embedding) Values() []float32 {
	if len(e.values) == 0 {
		return nil
	}

	values := make([]float32, len(e.values))
	copy(values, e.values)
	return values
}

// CosineSimilarity returns the cosine similarity between two normalized embeddings.
func (e Embedding) CosineSimilarity(other Embedding) (float32, error) {
	if len(e.values) == 0 || len(other.values) == 0 {
		return 0, ErrEmptyEmbedding
	}
	if len(e.values) != len(other.values) {
		return 0, ErrEmbeddingDimensionMismatch
	}

	return dot(e.values, other.values), nil
}

func (e Embedding) clone() Embedding {
	if len(e.values) == 0 {
		return Embedding{}
	}

	values := make([]float32, len(e.values))
	copy(values, e.values)
	return Embedding{values: values}
}

func dot(left, right []float32) float32 {
	var sum float32
	for i, value := range left {
		sum += value * right[i]
	}
	return sum
}
