package insightface

import (
	"errors"
	"math"
	"strconv"
	"sync"
	"testing"
)

func TestMatcherMatchReturnsBestAndSecondDifferentIdentity(t *testing.T) {
	matcher := newTestMatcher(t, MatcherConfig{MinSimilarity: 0.95, MinMargin: 0.03}, []GalleryIdentity{
		{
			ID: "alice",
			Samples: []Embedding{
				mustEmbedding(t, []float32{0.82, 0.57}),
				mustEmbedding(t, []float32{0.8, 0.6}),
			},
		},
		{
			ID: "bob",
			Samples: []Embedding{
				mustEmbedding(t, []float32{0.6, 0.8}),
			},
		},
		{
			ID: "carol",
			Samples: []Embedding{
				mustEmbedding(t, []float32{-1, 0}),
			},
		},
	})

	result, err := matcher.Match(mustEmbedding(t, []float32{0.8, 0.6}))
	if err != nil {
		t.Fatalf("Match() error = %v", err)
	}

	if !result.Accepted {
		t.Fatalf("Match().Accepted = false, want true")
	}
	if !result.Best.Found || result.Best.Identity != "alice" {
		t.Fatalf("Match().Best = %+v, want alice", result.Best)
	}
	if result.Best.SampleIndex != 1 {
		t.Fatalf("Match().Best.SampleIndex = %d, want 1", result.Best.SampleIndex)
	}
	if !result.SecondBestDifferent.Found || result.SecondBestDifferent.Identity != "bob" {
		t.Fatalf("Match().SecondBestDifferent = %+v, want bob", result.SecondBestDifferent)
	}
	if diff := math.Abs(float64(result.Best.Similarity - 1)); diff > 1e-6 {
		t.Fatalf("Match().Best.Similarity = %f, want 1", result.Best.Similarity)
	}
	if diff := math.Abs(float64(result.SecondBestDifferent.Similarity - 0.96)); diff > 1e-5 {
		t.Fatalf("Match().SecondBestDifferent.Similarity = %f, want 0.96", result.SecondBestDifferent.Similarity)
	}
	if diff := math.Abs(float64(result.Margin - 0.04)); diff > 1e-5 {
		t.Fatalf("Match().Margin = %f, want 0.04", result.Margin)
	}
}

func TestMatcherMatchAppliesSimilarityAndMarginThresholds(t *testing.T) {
	sharedGallery := []GalleryIdentity{
		{
			ID: "alice",
			Samples: []Embedding{
				mustEmbedding(t, []float32{0.82, 0.57}),
				mustEmbedding(t, []float32{0.8, 0.6}),
			},
		},
		{
			ID: "bob",
			Samples: []Embedding{
				mustEmbedding(t, []float32{0.6, 0.8}),
			},
		},
	}

	tests := []struct {
		name       string
		cfg        MatcherConfig
		gallery    []GalleryIdentity
		query      Embedding
		accepted   bool
		wantSecond bool
	}{
		{
			name:       "accepted",
			cfg:        MatcherConfig{MinSimilarity: 0.95, MinMargin: 0.03},
			gallery:    sharedGallery,
			query:      mustEmbedding(t, []float32{0.8, 0.6}),
			accepted:   true,
			wantSecond: true,
		},
		{
			name:       "similarity too low",
			cfg:        MatcherConfig{MinSimilarity: 0.95, MinMargin: 0.05},
			gallery:    []GalleryIdentity{{ID: "alice", Samples: []Embedding{mustEmbedding(t, []float32{1, 0})}}, {ID: "bob", Samples: []Embedding{mustEmbedding(t, []float32{0, 1})}}},
			query:      mustEmbedding(t, []float32{0.3, 0.7}),
			accepted:   false,
			wantSecond: true,
		},
		{
			name:       "margin too low",
			cfg:        MatcherConfig{MinSimilarity: 0.95, MinMargin: 0.05},
			gallery:    sharedGallery,
			query:      mustEmbedding(t, []float32{0.8, 0.6}),
			accepted:   false,
			wantSecond: true,
		},
		{
			name:       "missing second identity",
			cfg:        MatcherConfig{MinSimilarity: 0.95, MinMargin: 0},
			gallery:    []GalleryIdentity{{ID: "alice", Samples: []Embedding{mustEmbedding(t, []float32{0.8, 0.6})}}},
			query:      mustEmbedding(t, []float32{0.8, 0.6}),
			accepted:   false,
			wantSecond: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matcher := newTestMatcher(t, tt.cfg, tt.gallery)

			result, err := matcher.Match(tt.query)
			if err != nil {
				t.Fatalf("Match() error = %v", err)
			}
			if result.Accepted != tt.accepted {
				t.Fatalf("Match().Accepted = %v, want %v", result.Accepted, tt.accepted)
			}
			if result.SecondBestDifferent.Found != tt.wantSecond {
				t.Fatalf("Match().SecondBestDifferent.Found = %v, want %v", result.SecondBestDifferent.Found, tt.wantSecond)
			}
		})
	}
}

func TestMatcherReplaceGalleryCopiesAndRemainsImmutable(t *testing.T) {
	raw := []float32{3, 4}
	sample, err := NewEmbedding(raw)
	if err != nil {
		t.Fatalf("NewEmbedding() error = %v", err)
	}

	gallery := []GalleryIdentity{
		{ID: "alice", Samples: []Embedding{sample}},
		{ID: "bob", Samples: []Embedding{mustEmbedding(t, []float32{0, 1})}},
	}
	matcher := newTestMatcher(t, MatcherConfig{}, gallery)

	gallery[0].ID = "mutated"
	gallery[0].Samples[0] = mustEmbedding(t, []float32{-1, 0})
	raw[0] = -30

	result, err := matcher.Match(mustEmbedding(t, []float32{3, 4}))
	if err != nil {
		t.Fatalf("Match() error = %v", err)
	}
	if result.Best.Identity != "alice" {
		t.Fatalf("Match().Best.Identity = %q, want %q", result.Best.Identity, "alice")
	}
}

func TestMatcherMatchRejectsDimensionMismatchAndEmptyGallery(t *testing.T) {
	matcher := newTestMatcher(t, MatcherConfig{}, []GalleryIdentity{
		{ID: "alice", Samples: []Embedding{mustEmbedding(t, []float32{1, 0})}},
	})

	_, err := matcher.Match(mustEmbedding(t, []float32{1, 0, 0}))
	if !errors.Is(err, ErrEmbeddingDimensionMismatch) {
		t.Fatalf("Match() dimension mismatch error = %v, want %v", err, ErrEmbeddingDimensionMismatch)
	}

	if err := matcher.ReplaceGallery(nil); err != nil {
		t.Fatalf("ReplaceGallery(nil) error = %v", err)
	}
	_, err = matcher.Match(mustEmbedding(t, []float32{1, 0}))
	if !errors.Is(err, ErrEmptyGallery) {
		t.Fatalf("Match() empty gallery error = %v, want %v", err, ErrEmptyGallery)
	}
}

func TestMatcherRejectsNonFiniteConfigAndDuplicateIdentities(t *testing.T) {
	sample := mustEmbedding(t, []float32{1, 0})
	for _, config := range []MatcherConfig{
		{MinSimilarity: float32(math.NaN())},
		{MinMargin: float32(math.Inf(1))},
	} {
		if _, err := NewMatcher(config, nil); !errors.Is(err, ErrInvalidMatcherConfig) {
			t.Fatalf("NewMatcher(%+v) error = %v, want %v", config, err, ErrInvalidMatcherConfig)
		}
	}

	_, err := NewMatcher(MatcherConfig{}, []GalleryIdentity{
		{ID: "alice", Samples: []Embedding{sample}},
		{ID: "alice", Samples: []Embedding{sample}},
	})
	if !errors.Is(err, ErrDuplicateIdentity) {
		t.Fatalf("NewMatcher() error = %v, want %v", err, ErrDuplicateIdentity)
	}
}

func TestMatcherNilReceiverReturnsTypedError(t *testing.T) {
	var matcher *Matcher
	if err := matcher.ReplaceGallery(nil); !errors.Is(err, ErrNilMatcher) {
		t.Fatalf("ReplaceGallery() error = %v, want %v", err, ErrNilMatcher)
	}
	if _, err := matcher.Match(Embedding{}); !errors.Is(err, ErrNilMatcher) {
		t.Fatalf("Match() error = %v, want %v", err, ErrNilMatcher)
	}
	if _, err := matcher.MatchVector(nil); !errors.Is(err, ErrNilMatcher) {
		t.Fatalf("MatchVector() error = %v, want %v", err, ErrNilMatcher)
	}
}

func TestMatcherConcurrentReplaceAndMatch(t *testing.T) {
	matcher := newTestMatcher(t, MatcherConfig{MinSimilarity: 0.7, MinMargin: 0.05}, []GalleryIdentity{
		{ID: "alice", Samples: []Embedding{mustEmbedding(t, []float32{1, 0})}},
		{ID: "bob", Samples: []Embedding{mustEmbedding(t, []float32{0, 1})}},
	})

	query := mustEmbedding(t, []float32{1, 0})

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			for j := 0; j < 200; j++ {
				if _, err := matcher.Match(query); err != nil {
					t.Errorf("Match() error = %v", err)
					return
				}
				if j%25 == 0 {
					id := "alice-" + strconv.Itoa(i) + "-" + strconv.Itoa(j)
					err := matcher.ReplaceGallery([]GalleryIdentity{
						{ID: id, Samples: []Embedding{mustEmbedding(t, []float32{1, 0})}},
						{ID: "bob", Samples: []Embedding{mustEmbedding(t, []float32{0, 1})}},
					})
					if err != nil {
						t.Errorf("ReplaceGallery() error = %v", err)
						return
					}
				}
			}
		}(i)
	}

	wg.Wait()
}

func BenchmarkMatcherMatch3000Identities3Samples512Dims(b *testing.B) {
	const (
		identities = 3000
		samples    = 3
		dim        = 512
	)

	gallery := make([]GalleryIdentity, 0, identities)
	for identityIndex := 0; identityIndex < identities; identityIndex++ {
		identity := GalleryIdentity{
			ID:      "id-" + strconv.Itoa(identityIndex),
			Samples: make([]Embedding, 0, samples),
		}
		for sampleIndex := 0; sampleIndex < samples; sampleIndex++ {
			identity.Samples = append(identity.Samples, mustEmbedding(b, benchmarkVector(identityIndex, sampleIndex, dim)))
		}
		gallery = append(gallery, identity)
	}

	matcher, err := NewMatcher(MatcherConfig{MinSimilarity: 0.8, MinMargin: 0.01}, gallery)
	if err != nil {
		b.Fatalf("NewMatcher() error = %v", err)
	}

	query := mustEmbedding(b, benchmarkVector(1731, 2, dim))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := matcher.Match(query)
		if err != nil {
			b.Fatalf("Match() error = %v", err)
		}
		if !result.Best.Found || result.Best.Identity != "id-1731" {
			b.Fatalf("Match().Best = %+v, want id-1731", result.Best)
		}
	}
}

func mustEmbedding(tb testing.TB, raw []float32) Embedding {
	tb.Helper()

	embedding, err := NewEmbedding(raw)
	if err != nil {
		tb.Fatalf("NewEmbedding() error = %v", err)
	}
	return embedding
}

func newTestMatcher(tb testing.TB, cfg MatcherConfig, gallery []GalleryIdentity) *Matcher {
	tb.Helper()

	matcher, err := NewMatcher(cfg, gallery)
	if err != nil {
		tb.Fatalf("NewMatcher() error = %v", err)
	}
	return matcher
}

func benchmarkVector(identityIndex, sampleIndex, dim int) []float32 {
	vector := make([]float32, dim)
	seed := float64((identityIndex + 1) * (sampleIndex + 3))
	for i := range vector {
		x := float64(i + 1)
		vector[i] = float32(math.Sin(seed*x*0.013) + 0.25*math.Cos(seed*x*0.007))
	}
	return vector
}
