package scrfd

import (
	"context"
	"errors"
	"fmt"
	"image"

	"github.com/lib-x/go-insightface/internal/imageutil"
	"github.com/lib-x/go-insightface/internal/ort"
)

var ErrInvalidImage = errors.New("scrfd: invalid image")

type Session interface {
	Run(context.Context, []float32, []int64) ([]ort.Tensor, error)
}

type Detector struct {
	session Session
	config  DecoderConfig
}

func NewDetector(session Session, config DecoderConfig) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("session is nil: %w", ErrInvalidConfig)
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	return &Detector{session: session, config: config}, nil
}

func (d *Detector) Detect(ctx context.Context, src image.Image) ([]Detection, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if src == nil || src.Bounds().Dx() <= 0 || src.Bounds().Dy() <= 0 {
		return nil, ErrInvalidImage
	}

	resized, scale, err := imageutil.FitTopLeft(src, d.config.InputWidth, d.config.InputHeight)
	if err != nil {
		return nil, fmt.Errorf("resize detector input: %w", err)
	}
	input, err := imageutil.ToNCHW(resized, imageutil.PixelTransform{
		Mean:  [3]float32{127.5, 127.5, 127.5},
		Scale: [3]float32{1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0},
	})
	if err != nil {
		return nil, fmt.Errorf("prepare detector input: %w", err)
	}

	outputs, err := d.session.Run(ctx, input, []int64{1, 3, int64(d.config.InputHeight), int64(d.config.InputWidth)})
	if err != nil {
		return nil, fmt.Errorf("run SCRFD detector: %w", err)
	}
	decoded := make([]Tensor, len(outputs))
	for i, output := range outputs {
		decoded[i] = Tensor{Shape: output.Shape, Data: output.Data}
	}
	detections, err := Decode(decoded, d.config)
	if err != nil {
		return nil, fmt.Errorf("decode SCRFD outputs: %w", err)
	}
	for i := range detections {
		detections[i].Box.Left /= float32(scale)
		detections[i].Box.Top /= float32(scale)
		detections[i].Box.Right /= float32(scale)
		detections[i].Box.Bottom /= float32(scale)
		for point := range detections[i].Landmarks {
			detections[i].Landmarks[point].X /= float32(scale)
			detections[i].Landmarks[point].Y /= float32(scale)
		}
	}
	return detections, nil
}
