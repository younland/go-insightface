package insightface

import (
	"context"
	"errors"
	"image"
	"sync"
	"testing"
	"time"

	"github.com/lib-x/go-insightface/internal/imageutil"
	"github.com/lib-x/go-insightface/internal/scrfd"
)

func TestNormalizeConfigAppliesDefaults(t *testing.T) {
	resolved, decoder, err := normalizeConfig(Config{RuntimeLibrary: "runtime", DetectorModel: "detector", RecognizerModel: "recognizer"})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.DetectorInputWidth != 640 || resolved.DetectorInputHeight != 640 || resolved.DetectionThreshold != 0.5 || resolved.NMSThreshold != 0.4 {
		t.Fatalf("resolved config = %+v", resolved)
	}
	if decoder.ScoreThreshold != 0.5 || decoder.NMSThreshold != 0.4 {
		t.Fatalf("decoder config = %+v", decoder)
	}
}

func TestEngineAnalyzeComposesDetectionAndEmbedding(t *testing.T) {
	detection := scrfd.Detection{
		Score:     0.9,
		Box:       scrfd.Box{Left: 1, Top: 2, Right: 10, Bottom: 11},
		Landmarks: [5]scrfd.Point{{X: 1, Y: 2}, {X: 3, Y: 4}, {X: 5, Y: 6}, {X: 7, Y: 8}, {X: 9, Y: 10}},
	}
	engine := &Engine{
		runtime:  &fakeRuntime{},
		detector: &fakeDetector{detections: []scrfd.Detection{detection}},
		embedder: &fakeEmbedder{values: []float32{3, 4}},
	}

	faces, err := engine.Analyze(context.Background(), image.NewRGBA(image.Rect(0, 0, 20, 20)))
	if err != nil {
		t.Fatal(err)
	}
	if len(faces) != 1 || faces[0].Detection.Score != 0.9 {
		t.Fatalf("faces = %+v", faces)
	}
	if faces[0].Embedding.Dimension() != 2 {
		t.Fatalf("embedding dimension = %d", faces[0].Embedding.Dimension())
	}
	values := faces[0].Embedding.Values()
	if values[0] != 0.6 || values[1] != 0.8 {
		t.Fatalf("embedding = %v", values)
	}
}

func TestEngineCloseWaitsForInferenceAndIsIdempotent(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	runtime := &fakeRuntime{}
	engine := &Engine{
		runtime: runtime,
		detector: &fakeDetector{run: func() {
			close(started)
			<-release
		}},
		embedder: &fakeEmbedder{values: []float32{1}},
	}

	runDone := make(chan error, 1)
	go func() {
		_, err := engine.Detect(context.Background(), image.NewRGBA(image.Rect(0, 0, 1, 1)))
		runDone <- err
	}()
	<-started

	closeDone := make(chan error, 1)
	go func() { closeDone <- engine.Close() }()
	select {
	case <-closeDone:
		t.Fatal("Close returned before in-flight inference completed")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-runDone; err != nil {
		t.Fatal(err)
	}
	if err := <-closeDone; err != nil {
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if runtime.closeCalls != 1 {
		t.Fatalf("runtime close calls = %d, want 1", runtime.closeCalls)
	}
	if _, err := engine.Detect(context.Background(), image.NewRGBA(image.Rect(0, 0, 1, 1))); !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("Detect() error = %v, want %v", err, ErrEngineClosed)
	}
}

func TestEngineRejectsNilContextAndTypedNilImage(t *testing.T) {
	engine := &Engine{detector: &fakeDetector{}, embedder: &fakeEmbedder{}}
	if _, err := engine.Detect(nil, image.NewRGBA(image.Rect(0, 0, 1, 1))); !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("Detect(nil context) error = %v", err)
	}
	var src *image.RGBA
	if _, err := engine.Detect(context.Background(), src); !errors.Is(err, ErrInvalidImage) {
		t.Fatalf("Detect(typed nil image) error = %v", err)
	}
}

type fakeRuntime struct {
	mu         sync.Mutex
	closeCalls int
}

func (r *fakeRuntime) Close() error {
	r.mu.Lock()
	r.closeCalls++
	r.mu.Unlock()
	return nil
}

type fakeDetector struct {
	detections []scrfd.Detection
	err        error
	run        func()
}

func (d *fakeDetector) Detect(context.Context, image.Image) ([]scrfd.Detection, error) {
	if d.run != nil {
		d.run()
	}
	return d.detections, d.err
}

type fakeEmbedder struct {
	values []float32
	err    error
}

func (e *fakeEmbedder) Embed(context.Context, image.Image, [5]imageutil.Point) ([]float32, error) {
	return e.values, e.err
}
