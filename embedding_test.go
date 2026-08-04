package insightface

import (
	"errors"
	"math"
	"testing"
)

func TestNewEmbeddingNormalizesAndCopiesInput(t *testing.T) {
	raw := []float32{3, 4}

	embedding, err := NewEmbedding(raw)
	if err != nil {
		t.Fatalf("NewEmbedding() error = %v", err)
	}

	raw[0] = 30

	values := embedding.Values()
	if len(values) != 2 {
		t.Fatalf("len(Values()) = %d, want 2", len(values))
	}
	if diff := math.Abs(float64(values[0] - 0.6)); diff > 1e-6 {
		t.Fatalf("Values()[0] = %f, want 0.6", values[0])
	}
	if diff := math.Abs(float64(values[1] - 0.8)); diff > 1e-6 {
		t.Fatalf("Values()[1] = %f, want 0.8", values[1])
	}

	var norm float64
	for _, value := range values {
		norm += float64(value * value)
	}
	if diff := math.Abs(math.Sqrt(norm) - 1); diff > 1e-6 {
		t.Fatalf("L2 norm = %f, want 1", math.Sqrt(norm))
	}
}

func TestNewEmbeddingRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name string
		raw  []float32
		want error
	}{
		{
			name: "empty",
			raw:  nil,
			want: ErrEmptyEmbedding,
		},
		{
			name: "zero norm",
			raw:  []float32{0, 0, 0},
			want: ErrZeroNormEmbedding,
		},
		{
			name: "nan",
			raw:  []float32{1, float32(math.NaN())},
			want: ErrInvalidEmbeddingValue,
		},
		{
			name: "inf",
			raw:  []float32{1, float32(math.Inf(1))},
			want: ErrInvalidEmbeddingValue,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewEmbedding(tt.raw)
			if !errors.Is(err, tt.want) {
				t.Fatalf("NewEmbedding() error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestEmbeddingValuesReturnsCopy(t *testing.T) {
	embedding, err := NewEmbedding([]float32{1, 2, 3})
	if err != nil {
		t.Fatalf("NewEmbedding() error = %v", err)
	}

	values := embedding.Values()
	values[0] = -1

	again := embedding.Values()
	if again[0] < 0 {
		t.Fatalf("Values() exposed mutable backing storage: %v", again)
	}
}

func TestEmbeddingCosineSimilarity(t *testing.T) {
	first, err := NewEmbedding([]float32{3, 4})
	if err != nil {
		t.Fatalf("NewEmbedding(first) error = %v", err)
	}
	same, err := NewEmbedding([]float32{6, 8})
	if err != nil {
		t.Fatalf("NewEmbedding(same) error = %v", err)
	}
	orthogonal, err := NewEmbedding([]float32{-4, 3})
	if err != nil {
		t.Fatalf("NewEmbedding(orthogonal) error = %v", err)
	}
	wrongDim, err := NewEmbedding([]float32{1, 2, 3})
	if err != nil {
		t.Fatalf("NewEmbedding(wrongDim) error = %v", err)
	}

	got, err := first.CosineSimilarity(same)
	if err != nil {
		t.Fatalf("CosineSimilarity(same) error = %v", err)
	}
	if diff := math.Abs(float64(got - 1)); diff > 1e-6 {
		t.Fatalf("CosineSimilarity(same) = %f, want 1", got)
	}

	got, err = first.CosineSimilarity(orthogonal)
	if err != nil {
		t.Fatalf("CosineSimilarity(orthogonal) error = %v", err)
	}
	if diff := math.Abs(float64(got)); diff > 1e-6 {
		t.Fatalf("CosineSimilarity(orthogonal) = %f, want 0", got)
	}

	_, err = first.CosineSimilarity(wrongDim)
	if !errors.Is(err, ErrEmbeddingDimensionMismatch) {
		t.Fatalf("CosineSimilarity(wrongDim) error = %v, want %v", err, ErrEmbeddingDimensionMismatch)
	}
}
