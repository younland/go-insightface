package insightface

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"reflect"
	"sync"

	"github.com/lib-x/go-insightface/internal/arcface"
	"github.com/lib-x/go-insightface/internal/imageutil"
	"github.com/lib-x/go-insightface/internal/ort"
	"github.com/lib-x/go-insightface/internal/scrfd"
)

const (
	defaultDetectorSize       = 640
	defaultDetectionThreshold = 0.5
	defaultNMSThreshold       = 0.4
)

var (
	ErrInvalidConfig    = errors.New("insightface: invalid engine config")
	ErrInvalidContext   = errors.New("insightface: context is nil")
	ErrInvalidImage     = errors.New("insightface: invalid image")
	ErrInvalidLandmarks = errors.New("insightface: invalid landmarks")
	ErrEngineClosed     = errors.New("insightface: engine is closed")
	ErrInitialization   = errors.New("insightface: engine initialization failed")
	ErrInference        = errors.New("insightface: model inference failed")
)

// Config identifies the ONNX Runtime library and the two model files.
// DetectionThreshold and NMSThreshold default to 0.5 and 0.4 when zero.
// DetectorInputWidth and DetectorInputHeight default to 640 when zero.
type Config struct {
	RuntimeLibrary      string
	DetectorModel       string
	RecognizerModel     string
	DetectorInputWidth  int
	DetectorInputHeight int
	DetectionThreshold  float32
	NMSThreshold        float32
	IntraOpThreads      int
}

type Point struct {
	X float32
	Y float32
}

type Box struct {
	Left   float32
	Top    float32
	Right  float32
	Bottom float32
}

type Landmarks [5]Point

type Detection struct {
	Score     float32
	Box       Box
	Landmarks Landmarks
}

type Face struct {
	Detection Detection
	Embedding Embedding
}

type detector interface {
	Detect(context.Context, image.Image) ([]scrfd.Detection, error)
}

type embedder interface {
	Embed(context.Context, image.Image, [5]imageutil.Point) ([]float32, error)
}

type runtimeCloser interface {
	Close() error
}

// Engine owns the ONNX Runtime environment and detector/recognizer sessions.
// Its inference methods are safe for concurrent use.
type Engine struct {
	mu       sync.RWMutex
	runtime  runtimeCloser
	detector detector
	embedder embedder
	closed   bool
	closeErr error
}

func New(config Config) (*Engine, error) {
	resolved, decoderConfig, err := normalizeConfig(config)
	if err != nil {
		return nil, err
	}

	runtime, err := ort.Open(resolved.RuntimeLibrary)
	if err != nil {
		return nil, fmt.Errorf("%w: open ONNX Runtime: %v", ErrInitialization, err)
	}
	fail := func(cause error) (*Engine, error) {
		_ = runtime.Close()
		return nil, fmt.Errorf("%w: %v", ErrInitialization, cause)
	}

	detectorSession, err := runtime.Load(resolved.DetectorModel, resolved.IntraOpThreads)
	if err != nil {
		return fail(fmt.Errorf("load detector: %w", err))
	}
	recognizerSession, err := runtime.Load(resolved.RecognizerModel, resolved.IntraOpThreads)
	if err != nil {
		return fail(fmt.Errorf("load recognizer: %w", err))
	}
	detector, err := scrfd.NewDetector(detectorSession, decoderConfig)
	if err != nil {
		return fail(fmt.Errorf("configure detector: %w", err))
	}
	embedder, err := arcface.NewEmbedder(recognizerSession)
	if err != nil {
		return fail(fmt.Errorf("configure recognizer: %w", err))
	}

	return &Engine{runtime: runtime, detector: detector, embedder: embedder}, nil
}

func (e *Engine) Detect(ctx context.Context, src image.Image) ([]Detection, error) {
	if err := validateCall(ctx, src); err != nil {
		return nil, err
	}
	if e == nil {
		return nil, ErrEngineClosed
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.detector == nil {
		return nil, ErrEngineClosed
	}
	return e.detect(ctx, src)
}

func (e *Engine) Embed(ctx context.Context, src image.Image, landmarks Landmarks) (Embedding, error) {
	if err := validateCall(ctx, src); err != nil {
		return Embedding{}, err
	}
	if e == nil {
		return Embedding{}, ErrEngineClosed
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.embedder == nil {
		return Embedding{}, ErrEngineClosed
	}
	return e.embed(ctx, src, landmarks)
}

func (e *Engine) Analyze(ctx context.Context, src image.Image) ([]Face, error) {
	if err := validateCall(ctx, src); err != nil {
		return nil, err
	}
	if e == nil {
		return nil, ErrEngineClosed
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed || e.detector == nil || e.embedder == nil {
		return nil, ErrEngineClosed
	}

	detections, err := e.detect(ctx, src)
	if err != nil {
		return nil, err
	}
	faces := make([]Face, 0, len(detections))
	for _, detection := range detections {
		embedding, err := e.embed(ctx, src, detection.Landmarks)
		if err != nil {
			return nil, fmt.Errorf("embed detected face: %w", err)
		}
		faces = append(faces, Face{Detection: detection, Embedding: embedding})
	}
	return faces, nil
}

// Close waits for in-flight inference, then releases both sessions and the
// ONNX Runtime environment. Repeated calls return the first close result.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return e.closeErr
	}
	e.closed = true
	if e.runtime != nil {
		e.closeErr = e.runtime.Close()
	}
	e.runtime = nil
	e.detector = nil
	e.embedder = nil
	return e.closeErr
}

func (e *Engine) detect(ctx context.Context, src image.Image) ([]Detection, error) {
	internal, err := e.detector.Detect(ctx, src)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		if errors.Is(err, scrfd.ErrInvalidImage) {
			return nil, fmt.Errorf("%w: %v", ErrInvalidImage, err)
		}
		return nil, fmt.Errorf("%w: detect face: %v", ErrInference, err)
	}
	result := make([]Detection, len(internal))
	for i, detection := range internal {
		result[i] = publicDetection(detection)
	}
	return result, nil
}

func (e *Engine) embed(ctx context.Context, src image.Image, landmarks Landmarks) (Embedding, error) {
	internal := internalLandmarks(landmarks)
	values, err := e.embedder.Embed(ctx, src, internal)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return Embedding{}, err
		}
		if errors.Is(err, arcface.ErrInvalidLandmarks) {
			return Embedding{}, fmt.Errorf("%w: %v", ErrInvalidLandmarks, err)
		}
		if errors.Is(err, imageutil.ErrInvalidDimensions) {
			return Embedding{}, fmt.Errorf("%w: %v", ErrInvalidImage, err)
		}
		return Embedding{}, fmt.Errorf("%w: embed face: %v", ErrInference, err)
	}
	embedding, err := NewEmbedding(values)
	if err != nil {
		return Embedding{}, fmt.Errorf("%w: normalize embedding: %v", ErrInference, err)
	}
	return embedding, nil
}

func normalizeConfig(config Config) (Config, scrfd.DecoderConfig, error) {
	if config.RuntimeLibrary == "" || config.DetectorModel == "" || config.RecognizerModel == "" {
		return Config{}, scrfd.DecoderConfig{}, fmt.Errorf("runtime library and both model paths are required: %w", ErrInvalidConfig)
	}
	if config.DetectorInputWidth == 0 {
		config.DetectorInputWidth = defaultDetectorSize
	}
	if config.DetectorInputHeight == 0 {
		config.DetectorInputHeight = defaultDetectorSize
	}
	if config.DetectionThreshold == 0 {
		config.DetectionThreshold = defaultDetectionThreshold
	}
	if config.NMSThreshold == 0 {
		config.NMSThreshold = defaultNMSThreshold
	}
	if config.DetectorInputWidth <= 0 || config.DetectorInputHeight <= 0 || config.DetectorInputWidth%32 != 0 || config.DetectorInputHeight%32 != 0 {
		return Config{}, scrfd.DecoderConfig{}, fmt.Errorf("detector dimensions must be positive multiples of 32: %w", ErrInvalidConfig)
	}
	if config.IntraOpThreads < 0 || !finite32(config.DetectionThreshold) || config.DetectionThreshold < 0 || config.DetectionThreshold > 1 || !finite32(config.NMSThreshold) || config.NMSThreshold < 0 || config.NMSThreshold > 1 {
		return Config{}, scrfd.DecoderConfig{}, ErrInvalidConfig
	}
	decoder := scrfd.DecoderConfig{
		InputWidth: config.DetectorInputWidth, InputHeight: config.DetectorInputHeight,
		Strides: []int{8, 16, 32}, ScoreThreshold: config.DetectionThreshold, NMSThreshold: config.NMSThreshold,
	}
	return config, decoder, nil
}

func validateCall(ctx context.Context, src image.Image) error {
	if ctx == nil {
		return ErrInvalidContext
	}
	if isNilImage(src) {
		return ErrInvalidImage
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func isNilImage(src image.Image) bool {
	if src == nil {
		return true
	}
	value := reflect.ValueOf(src)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func publicDetection(detection scrfd.Detection) Detection {
	result := Detection{
		Score: detection.Score,
		Box:   Box{Left: detection.Box.Left, Top: detection.Box.Top, Right: detection.Box.Right, Bottom: detection.Box.Bottom},
	}
	for i, point := range detection.Landmarks {
		result.Landmarks[i] = Point{X: point.X, Y: point.Y}
	}
	return result
}

func internalLandmarks(landmarks Landmarks) [5]imageutil.Point {
	var result [5]imageutil.Point
	for i, point := range landmarks {
		result[i] = imageutil.Point{X: float64(point.X), Y: float64(point.Y)}
	}
	return result
}

func finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
