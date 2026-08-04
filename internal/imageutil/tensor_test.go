package imageutil

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestToNCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	img.Set(1, 0, color.RGBA{R: 40, G: 50, B: 60, A: 255})

	got, err := ToNCHW(img, PixelTransform{Mean: [3]float32{10, 20, 30}, Scale: [3]float32{1, 0.5, 0.2}})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 30, 0, 15, 0, 6}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("value[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestFitTopLeft(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 4, 2))
	got, scale, err := FitTopLeft(src, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got.Bounds().Dx() != 2 || got.Bounds().Dy() != 2 {
		t.Fatalf("bounds = %v", got.Bounds())
	}
	if scale != 0.5 {
		t.Fatalf("scale = %v, want 0.5", scale)
	}
}

func TestFitTopLeftMatchesInsightFaceTruncation(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := range 2 {
		for x := range 3 {
			src.SetRGBA(x, y, color.RGBA{R: 255, A: 255})
		}
	}

	got, scale, err := FitTopLeft(src, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 0.5 {
		t.Fatalf("scale = %v, want 0.5", scale)
	}
	r, _, _, _ := got.At(1, 0).RGBA()
	if r != 0xffff {
		t.Fatalf("resized pixel red = %x, want ffff", r)
	}
	r, _, _, _ = got.At(0, 1).RGBA()
	if r != 0 {
		t.Fatalf("padding pixel red = %x, want 0", r)
	}
}

func TestSimilarityTransform(t *testing.T) {
	src := [5]Point{{1, 1}, {2, 1}, {1.5, 1.5}, {1, 2}, {2, 2}}
	var dst [5]Point
	for i, p := range src {
		dst[i] = Point{X: 2*p.X - 3, Y: 2*p.Y + 4}
	}

	m, err := EstimateSimilarity(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range src {
		got := m.Apply(p)
		if math.Abs(got.X-dst[i].X) > 1e-9 || math.Abs(got.Y-dst[i].Y) > 1e-9 {
			t.Errorf("point %d = %+v, want %+v", i, got, dst[i])
		}
	}
}

func TestSimilarityTransformWithRotationScaleAndTranslation(t *testing.T) {
	src := [5]Point{{10, 20}, {30, 18}, {21, 31}, {12, 43}, {34, 42}}
	wantTransform := Affine{A: 1.2, B: 0.35, TX: -7, TY: 11}
	var dst [5]Point
	for i, point := range src {
		dst[i] = wantTransform.Apply(point)
	}

	got, err := EstimateSimilarity(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.A-wantTransform.A) > 1e-12 || math.Abs(got.B-wantTransform.B) > 1e-12 ||
		math.Abs(got.TX-wantTransform.TX) > 1e-12 || math.Abs(got.TY-wantTransform.TY) > 1e-12 {
		t.Fatalf("transform = %+v, want %+v", got, wantTransform)
	}
}

func TestWarpAffineIdentity(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 2, 2))
	src.Set(0, 0, color.RGBA{R: 255, A: 255})
	got, err := WarpAffine(src, IdentityAffine(), 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	r, _, _, _ := got.At(0, 0).RGBA()
	if r != 0xffff {
		t.Fatalf("red = %x, want ffff", r)
	}
}
