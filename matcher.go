package insightface

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

var (
	ErrEmptyGallery         = errors.New("insightface: gallery is empty")
	ErrEmptyIdentity        = errors.New("insightface: identity is empty")
	ErrIdentityHasNoSample  = errors.New("insightface: identity has no samples")
	ErrDuplicateIdentity    = errors.New("insightface: duplicate identity")
	ErrInvalidMatcherConfig = errors.New("insightface: invalid matcher config")
	ErrNilMatcher           = errors.New("insightface: matcher is nil")
)

// MatcherConfig controls similarity and margin acceptance for identity matching.
type MatcherConfig struct {
	MinSimilarity float32
	MinMargin     float32
}

// GalleryIdentity stores all enrolled samples for one identity.
type GalleryIdentity struct {
	ID      string
	Samples []Embedding
}

// MatchCandidate is an identity-level search candidate.
type MatchCandidate struct {
	Identity    string
	Similarity  float32
	SampleIndex int
	Found       bool
}

// MatchResult contains the best identity, second-best different identity, and acceptance decision.
type MatchResult struct {
	Best                MatchCandidate
	SecondBestDifferent MatchCandidate
	Margin              float32
	Accepted            bool
}

// Matcher performs exact identity-level search over an immutable gallery snapshot.
type Matcher struct {
	mu       sync.RWMutex
	config   MatcherConfig
	snapshot gallerySnapshot
}

type gallerySnapshot struct {
	dimension  int
	identities []galleryIdentity
}

type galleryIdentity struct {
	id      string
	samples []Embedding
}

// NewMatcher validates the configuration and installs a copied gallery snapshot.
func NewMatcher(config MatcherConfig, gallery []GalleryIdentity) (*Matcher, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}

	snapshot, err := buildSnapshot(gallery)
	if err != nil {
		return nil, err
	}

	return &Matcher{
		config:   config,
		snapshot: snapshot,
	}, nil
}

// ReplaceGallery swaps the matcher gallery with a copied immutable snapshot.
func (m *Matcher) ReplaceGallery(gallery []GalleryIdentity) error {
	if m == nil {
		return ErrNilMatcher
	}
	snapshot, err := buildSnapshot(gallery)
	if err != nil {
		return err
	}

	m.mu.Lock()
	m.snapshot = snapshot
	m.mu.Unlock()
	return nil
}

// Match searches the gallery using a normalized query embedding.
func (m *Matcher) Match(query Embedding) (MatchResult, error) {
	if m == nil {
		return MatchResult{}, ErrNilMatcher
	}
	if query.Dimension() == 0 {
		return MatchResult{}, ErrEmptyEmbedding
	}

	m.mu.RLock()
	snapshot := m.snapshot
	config := m.config
	m.mu.RUnlock()

	if len(snapshot.identities) == 0 {
		return MatchResult{}, ErrEmptyGallery
	}
	if query.Dimension() != snapshot.dimension {
		return MatchResult{}, ErrEmbeddingDimensionMismatch
	}

	best := MatchCandidate{}
	second := MatchCandidate{}
	for _, identity := range snapshot.identities {
		candidate := bestForIdentity(query, identity)
		if !best.Found || candidate.Similarity > best.Similarity {
			second = best
			best = candidate
			continue
		}
		if !second.Found || candidate.Similarity > second.Similarity {
			second = candidate
		}
	}

	result := MatchResult{
		Best:                best,
		SecondBestDifferent: second,
	}
	if second.Found {
		result.Margin = best.Similarity - second.Similarity
		result.Accepted = best.Similarity >= config.MinSimilarity && result.Margin >= config.MinMargin
	}

	return result, nil
}

// MatchVector validates and normalizes a raw query vector before matching.
func (m *Matcher) MatchVector(raw []float32) (MatchResult, error) {
	if m == nil {
		return MatchResult{}, ErrNilMatcher
	}
	query, err := NewEmbedding(raw)
	if err != nil {
		return MatchResult{}, err
	}
	return m.Match(query)
}

func (c MatcherConfig) validate() error {
	if math.IsNaN(float64(c.MinSimilarity)) || math.IsInf(float64(c.MinSimilarity), 0) || c.MinSimilarity < -1 || c.MinSimilarity > 1 {
		return fmt.Errorf("min similarity %.6f out of range [-1, 1]: %w", c.MinSimilarity, ErrInvalidMatcherConfig)
	}
	if math.IsNaN(float64(c.MinMargin)) || math.IsInf(float64(c.MinMargin), 0) || c.MinMargin < 0 || c.MinMargin > 2 {
		return fmt.Errorf("min margin %.6f out of range [0, 2]: %w", c.MinMargin, ErrInvalidMatcherConfig)
	}
	return nil
}

func buildSnapshot(gallery []GalleryIdentity) (gallerySnapshot, error) {
	if len(gallery) == 0 {
		return gallerySnapshot{}, nil
	}

	identities := make([]galleryIdentity, 0, len(gallery))
	dimension := 0
	seen := make(map[string]struct{}, len(gallery))
	for index, identity := range gallery {
		if identity.ID == "" {
			return gallerySnapshot{}, fmt.Errorf("gallery identity at index %d: %w", index, ErrEmptyIdentity)
		}
		if len(identity.Samples) == 0 {
			return gallerySnapshot{}, fmt.Errorf("gallery identity %q: %w", identity.ID, ErrIdentityHasNoSample)
		}
		if _, ok := seen[identity.ID]; ok {
			return gallerySnapshot{}, fmt.Errorf("gallery identity %q: %w", identity.ID, ErrDuplicateIdentity)
		}
		seen[identity.ID] = struct{}{}

		samples := make([]Embedding, len(identity.Samples))
		for sampleIndex, sample := range identity.Samples {
			if sample.Dimension() == 0 {
				return gallerySnapshot{}, fmt.Errorf("gallery identity %q sample %d: %w", identity.ID, sampleIndex, ErrEmptyEmbedding)
			}
			if dimension == 0 {
				dimension = sample.Dimension()
			}
			if sample.Dimension() != dimension {
				return gallerySnapshot{}, fmt.Errorf("gallery identity %q sample %d: %w", identity.ID, sampleIndex, ErrEmbeddingDimensionMismatch)
			}
			samples[sampleIndex] = sample.clone()
		}

		identities = append(identities, galleryIdentity{
			id:      identity.ID,
			samples: samples,
		})
	}

	return gallerySnapshot{
		dimension:  dimension,
		identities: identities,
	}, nil
}

func bestForIdentity(query Embedding, identity galleryIdentity) MatchCandidate {
	best := MatchCandidate{
		Identity: identity.id,
		Found:    true,
	}

	for sampleIndex, sample := range identity.samples {
		similarity := dot(query.values, sample.values)
		if sampleIndex == 0 || similarity > best.Similarity {
			best.Similarity = similarity
			best.SampleIndex = sampleIndex
		}
	}

	return best
}
