package imageutil

import (
	"errors"
	"image"
	"image/color"
	"math"

	"golang.org/x/image/draw"
)

var ErrInvalidDimensions = errors.New("image dimensions must be positive")

// PixelTransform applies (channel - Mean) * Scale while converting RGB pixels
// to a planar NCHW tensor.
type PixelTransform struct {
	Mean  [3]float32
	Scale [3]float32
}

func ToNCHW(src image.Image, transform PixelTransform) ([]float32, error) {
	if src == nil || src.Bounds().Dx() <= 0 || src.Bounds().Dy() <= 0 {
		return nil, ErrInvalidDimensions
	}
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	plane := w * h
	out := make([]float32, plane*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(src.Bounds().Min.X+x, src.Bounds().Min.Y+y).RGBA()
			i := y*w + x
			values := [3]float32{float32(r >> 8), float32(g >> 8), float32(b >> 8)}
			for c := range 3 {
				out[c*plane+i] = (values[c] - transform.Mean[c]) * transform.Scale[c]
			}
		}
	}
	return out, nil
}

// FitTopLeft resizes src without changing its aspect ratio and places it at
// the top-left of a black canvas. The returned scale maps source coordinates
// to destination coordinates.
func FitTopLeft(src image.Image, width, height int) (*image.RGBA, float64, error) {
	if src == nil || src.Bounds().Dx() <= 0 || src.Bounds().Dy() <= 0 || width <= 0 || height <= 0 {
		return nil, 0, ErrInvalidDimensions
	}
	sourceWidth, sourceHeight := src.Bounds().Dx(), src.Bounds().Dy()
	imageRatio := float64(sourceHeight) / float64(sourceWidth)
	modelRatio := float64(height) / float64(width)
	var resizedWidth, resizedHeight int
	if imageRatio > modelRatio {
		resizedHeight = height
		resizedWidth = max(1, int(float64(resizedHeight)/imageRatio))
	} else {
		resizedWidth = width
		resizedHeight = max(1, int(float64(resizedWidth)*imageRatio))
	}
	scale := float64(resizedHeight) / float64(sourceHeight)
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(dst, image.Rect(0, 0, resizedWidth, resizedHeight), src, src.Bounds(), draw.Over, nil)
	return dst, scale, nil
}

type Point struct {
	X float64
	Y float64
}

// Affine maps a point as:
// x' = A*x - B*y + TX
// y' = B*x + A*y + TY
type Affine struct {
	A  float64
	B  float64
	TX float64
	TY float64
}

func IdentityAffine() Affine { return Affine{A: 1} }

func (m Affine) Apply(p Point) Point {
	return Point{X: m.A*p.X - m.B*p.Y + m.TX, Y: m.B*p.X + m.A*p.Y + m.TY}
}

// EstimateSimilarity returns the least-squares, orientation-preserving
// similarity transform from src to dst.
func EstimateSimilarity(src, dst [5]Point) (Affine, error) {
	var srcMean, dstMean Point
	for i := range src {
		srcMean.X += src[i].X
		srcMean.Y += src[i].Y
		dstMean.X += dst[i].X
		dstMean.Y += dst[i].Y
	}
	srcMean.X /= float64(len(src))
	srcMean.Y /= float64(len(src))
	dstMean.X /= float64(len(dst))
	dstMean.Y /= float64(len(dst))

	var denom, aNumerator, bNumerator float64
	for i := range src {
		sx, sy := src[i].X-srcMean.X, src[i].Y-srcMean.Y
		dx, dy := dst[i].X-dstMean.X, dst[i].Y-dstMean.Y
		denom += sx*sx + sy*sy
		aNumerator += sx*dx + sy*dy
		bNumerator += sx*dy - sy*dx
	}
	if denom <= 1e-12 {
		return Affine{}, errors.New("source landmarks are degenerate")
	}
	a, b := aNumerator/denom, bNumerator/denom
	if a*a+b*b <= 1e-12 {
		return Affine{}, errors.New("estimated transform is singular")
	}
	return Affine{
		A:  a,
		B:  b,
		TX: dstMean.X - a*srcMean.X + b*srcMean.Y,
		TY: dstMean.Y - b*srcMean.X - a*srcMean.Y,
	}, nil
}

func WarpAffine(src image.Image, transform Affine, width, height int) (*image.RGBA, error) {
	if src == nil || src.Bounds().Dx() <= 0 || src.Bounds().Dy() <= 0 || width <= 0 || height <= 0 {
		return nil, ErrInvalidDimensions
	}
	denom := transform.A*transform.A + transform.B*transform.B
	if denom <= 1e-12 {
		return nil, errors.New("affine transform is singular")
	}
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			dx, dy := float64(x)-transform.TX, float64(y)-transform.TY
			sx := (transform.A*dx + transform.B*dy) / denom
			sy := (-transform.B*dx + transform.A*dy) / denom
			dst.SetRGBA(x, y, sampleBilinear(src, sx, sy))
		}
	}
	return dst, nil
}

func sampleBilinear(src image.Image, x, y float64) color.RGBA {
	bounds := src.Bounds()
	if x < 0 || y < 0 || x > float64(bounds.Dx()-1) || y > float64(bounds.Dy()-1) {
		return color.RGBA{A: 255}
	}
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	x1, y1 := min(x0+1, bounds.Dx()-1), min(y0+1, bounds.Dy()-1)
	fx, fy := x-float64(x0), y-float64(y0)
	channels := [4]float64{}
	weights := [4]float64{(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy}
	points := [4]image.Point{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}}
	for i, p := range points {
		r, g, b, a := src.At(bounds.Min.X+p.X, bounds.Min.Y+p.Y).RGBA()
		channels[0] += float64(r>>8) * weights[i]
		channels[1] += float64(g>>8) * weights[i]
		channels[2] += float64(b>>8) * weights[i]
		channels[3] += float64(a>>8) * weights[i]
	}
	return color.RGBA{R: uint8(math.Round(channels[0])), G: uint8(math.Round(channels[1])), B: uint8(math.Round(channels[2])), A: uint8(math.Round(channels[3]))}
}
