//go:build integration

package insightface

import (
	"context"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lib-x/go-insightface/runtimefs"
)

func TestEngineRunsLicensedExternalModels(t *testing.T) {
	library := os.Getenv("ORT_LIBRARY")
	detectorModel := os.Getenv("DETECTOR_MODEL")
	recognizerModel := os.Getenv("RECOGNIZER_MODEL")
	faceImage := os.Getenv("FACE_IMAGE")
	if library == "" || detectorModel == "" || recognizerModel == "" || faceImage == "" {
		t.Skip("ORT_LIBRARY, DETECTOR_MODEL, RECOGNIZER_MODEL, and FACE_IMAGE are required")
	}

	file, err := os.Open(faceImage)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	src, _, err := image.Decode(file)
	if err != nil {
		t.Fatal(err)
	}

	engine, err := New(Config{
		RuntimeLibrary:  library,
		DetectorModel:   detectorModel,
		RecognizerModel: recognizerModel,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	faces, err := engine.Analyze(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	if len(faces) == 0 {
		t.Fatal("no face detected")
	}
	if faces[0].Embedding.Dimension() != 512 {
		t.Fatalf("embedding dimension = %d, want 512", faces[0].Embedding.Dimension())
	}
	var squaredNorm float64
	for _, value := range faces[0].Embedding.Values() {
		squaredNorm += float64(value) * float64(value)
	}
	if math.Abs(math.Sqrt(squaredNorm)-1) > 1e-5 {
		t.Fatalf("embedding norm = %.8f, want 1", math.Sqrt(squaredNorm))
	}

	if referencePath := os.Getenv("REFERENCE_RESULT"); referencePath != "" {
		compareReferenceResult(t, referencePath, faces[0])
	}
}

func TestEngineLoadsRuntimeAndModelsFromFS(t *testing.T) {
	library := os.Getenv("ORT_LIBRARY")
	model := os.Getenv("ORT_TEST_MODEL")
	if library == "" || model == "" {
		t.Skip("ORT_LIBRARY and ORT_TEST_MODEL are required")
	}
	if !filepath.IsAbs(library) || !filepath.IsAbs(model) {
		t.Skip("runtimefs integration requires absolute fixture paths")
	}

	source, err := runtimefs.New(os.DirFS(string(filepath.Separator)), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	toFSPath := func(path string) string {
		return strings.TrimPrefix(filepath.ToSlash(path), "/")
	}
	engine, err := NewFromFS(context.Background(), Config{
		RuntimeLibrary:  toFSPath(library),
		DetectorModel:   toFSPath(model),
		RecognizerModel: toFSPath(model),
	}, source)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
}

type referenceResult struct {
	Box       [4]float32    `json:"box"`
	Landmarks [5][2]float32 `json:"landmarks"`
	Embedding []float32     `json:"embedding"`
}

func compareReferenceResult(t *testing.T, path string, face Face) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var reference referenceResult
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	gotBox := [4]float32{face.Detection.Box.Left, face.Detection.Box.Top, face.Detection.Box.Right, face.Detection.Box.Bottom}
	for i := range gotBox {
		if math.Abs(float64(gotBox[i]-reference.Box[i])) > 1 {
			t.Fatalf("box[%d] = %.5f, reference %.5f", i, gotBox[i], reference.Box[i])
		}
	}
	for i, landmark := range face.Detection.Landmarks {
		if math.Abs(float64(landmark.X-reference.Landmarks[i][0])) > 1 || math.Abs(float64(landmark.Y-reference.Landmarks[i][1])) > 1 {
			t.Fatalf("landmark[%d] = (%.5f, %.5f), reference (%.5f, %.5f)", i, landmark.X, landmark.Y, reference.Landmarks[i][0], reference.Landmarks[i][1])
		}
	}
	values := face.Embedding.Values()
	if len(values) != len(reference.Embedding) {
		t.Fatalf("embedding dimension = %d, reference %d", len(values), len(reference.Embedding))
	}
	var dot, leftNorm, rightNorm float64
	for i, value := range values {
		right := float64(reference.Embedding[i])
		dot += float64(value) * right
		leftNorm += float64(value) * float64(value)
		rightNorm += right * right
	}
	cosine := dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
	if cosine < 0.999 {
		t.Fatalf("embedding cosine against reference = %.8f, want >= 0.999", cosine)
	}
	t.Logf("reference comparison: embedding cosine=%.8f", cosine)
}
