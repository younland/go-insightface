package insightface

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"math"
	"reflect"
	"sync"

	"github.com/younland/go-insightface/internal/arcface"
	"github.com/younland/go-insightface/internal/imageutil"
	"github.com/younland/go-insightface/internal/ort"
	"github.com/younland/go-insightface/internal/scrfd"
	"github.com/younland/go-insightface/runtimefs"
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
	ErrInvalidRuntimeFS = errors.New("insightface: runtime fs is nil")
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
	// SessionConfig lists extra ONNX Runtime session configuration entries
	// (applied through AddSessionConfigEntry). Entries merge over the adapter
	// defaults, so a key like "session.intra_op.allow_spinning" can be
	// re-enabled by setting it here. Keys and values must be non-empty.
	SessionConfig map[string]string
	// CUDAOptions lists CUDAExecutionProvider options used when the provider
	// is auto-enabled (device_id, cudnn_conv_algo_search, gpu_mem_limit,
	// arena_extend_strategy, ...); nil selects the provider defaults.
	CUDAOptions map[string]string
	// TensorRTOptions lists TensorrtExecutionProvider options used when the
	// provider is auto-enabled; nil selects the provider defaults.
	TensorRTOptions map[string]string
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
	params := ort.SessionParams{
		Threads:  resolved.IntraOpThreads,
		Entries:  resolved.SessionConfig,
		CUDA:     resolved.CUDAOptions,
		TensorRT: resolved.TensorRTOptions,
	}
	detectorSession, err := runtime.Load(resolved.DetectorModel, params)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("load detector: %w", err))
	}
	recognizerSession, err := runtime.Load(resolved.RecognizerModel, params)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("load recognizer: %w", err))
	}
	return assembleEngine(runtime, detectorSession, recognizerSession, decoderConfig)
}

// NewFromFS creates an Engine whose model files and runtime library are read
// from source. Models are loaded directly from source; the runtime library is
// materialized because operating-system dynamic loaders require a file path.
// Config path fields are interpreted as fs.ValidPath names in source.
func NewFromFS(ctx context.Context, config Config, source runtimefs.FS) (*Engine, error) {
	if ctx == nil {
		return nil, ErrInvalidContext
	}
	if isNilRuntimeFS(source) {
		return nil, ErrInvalidRuntimeFS
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	resolved, decoderConfig, err := normalizeConfig(config)
	if err != nil {
		return nil, err
	}
	if !fs.ValidPath(resolved.RuntimeLibrary) || !fs.ValidPath(resolved.DetectorModel) || !fs.ValidPath(resolved.RecognizerModel) {
		return nil, fmt.Errorf("runtime library and model names must be valid fs paths: %w", ErrInvalidConfig)
	}
	runtimePath, err := source.Materialize(ctx, resolved.RuntimeLibrary)
	if err != nil {
		if contextErr := ctx.Err(); contextErr != nil {
			return nil, contextErr
		}
		return nil, fmt.Errorf("%w: materialize ONNX Runtime: %v", ErrInitialization, err)
	}
	runtime, err := ort.Open(runtimePath)
	if err != nil {
		return nil, fmt.Errorf("%w: open ONNX Runtime: %v", ErrInitialization, err)
	}

	params := ort.SessionParams{
		Threads:  resolved.IntraOpThreads,
		Entries:  resolved.SessionConfig,
		CUDA:     resolved.CUDAOptions,
		TensorRT: resolved.TensorRTOptions,
	}
	detectorModel, err := source.Open(resolved.DetectorModel)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("open detector model: %w", err))
	}
	detectorSession, loadErr := runtime.LoadReader(resolved.DetectorModel, contextReader{ctx: ctx, reader: detectorModel}, params)
	closeErr := detectorModel.Close()
	if loadErr != nil {
		return failInitialization(runtime, fmt.Errorf("load detector: %w", loadErr))
	}
	if closeErr != nil {
		return failInitialization(runtime, fmt.Errorf("close detector model: %w", closeErr))
	}
	if err := ctx.Err(); err != nil {
		_ = runtime.Close()
		return nil, err
	}

	recognizerModel, err := source.Open(resolved.RecognizerModel)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("open recognizer model: %w", err))
	}
	recognizerSession, loadErr := runtime.LoadReader(resolved.RecognizerModel, contextReader{ctx: ctx, reader: recognizerModel}, params)
	closeErr = recognizerModel.Close()
	if loadErr != nil {
		return failInitialization(runtime, fmt.Errorf("load recognizer: %w", loadErr))
	}
	if closeErr != nil {
		return failInitialization(runtime, fmt.Errorf("close recognizer model: %w", closeErr))
	}
	if err := ctx.Err(); err != nil {
		_ = runtime.Close()
		return nil, err
	}
	return assembleEngine(runtime, detectorSession, recognizerSession, decoderConfig)
}

func assembleEngine(runtime *ort.Runtime, detectorSession, recognizerSession *ort.Session, decoderConfig scrfd.DecoderConfig) (*Engine, error) {
	detector, err := scrfd.NewDetector(detectorSession, decoderConfig)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("configure detector: %w", err))
	}
	embedder, err := arcface.NewEmbedder(recognizerSession)
	if err != nil {
		return failInitialization(runtime, fmt.Errorf("configure recognizer: %w", err))
	}

	return &Engine{runtime: runtime, detector: detector, embedder: embedder}, nil
}

func failInitialization(runtime *ort.Runtime, cause error) (*Engine, error) {
	_ = runtime.Close()
	return nil, fmt.Errorf("%w: %v", ErrInitialization, cause)
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
	for key, value := range config.SessionConfig {
		if key == "" || value == "" {
			return Config{}, scrfd.DecoderConfig{}, fmt.Errorf("session config entries require non-empty key and value: %w", ErrInvalidConfig)
		}
	}
	for _, options := range []map[string]string{config.CUDAOptions, config.TensorRTOptions} {
		for key, value := range options {
			if key == "" || value == "" {
				return Config{}, scrfd.DecoderConfig{}, fmt.Errorf("provider option entries require non-empty key and value: %w", ErrInvalidConfig)
			}
		}
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

func isNilRuntimeFS(source runtimefs.FS) bool {
	if source == nil {
		return true
	}
	value := reflect.ValueOf(source)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(buffer []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	count, err := r.reader.Read(buffer)
	if contextErr := r.ctx.Err(); contextErr != nil {
		return count, contextErr
	}
	return count, err
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
