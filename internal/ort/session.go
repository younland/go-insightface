package ort

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"sync"

	upstream "github.com/younland/onnxruntime"
)

var ErrClosed = errors.New("onnx runtime resource is closed")

type Tensor struct {
	Name  string
	Shape []int64
	Data  []float32
}

func (t Tensor) Validate() error {
	if t.Name == "" {
		return errors.New("tensor name is empty")
	}
	count, err := elementCount(t.Shape)
	if err != nil {
		return err
	}
	if count != len(t.Data) {
		return fmt.Errorf("tensor %q contains %d values, shape requires %d", t.Name, len(t.Data), count)
	}
	return nil
}

func elementCount(shape []int64) (int, error) {
	if len(shape) == 0 {
		return 0, errors.New("tensor shape is empty")
	}
	count := int64(1)
	for _, dimension := range shape {
		if dimension <= 0 {
			return 0, fmt.Errorf("invalid tensor dimension %d", dimension)
		}
		if count > math.MaxInt64/dimension {
			return 0, errors.New("tensor shape overflows int64")
		}
		count *= dimension
	}
	if count > int64(int(^uint(0)>>1)) {
		return 0, errors.New("tensor is too large for this platform")
	}
	return int(count), nil
}

type Runtime struct {
	mu       sync.Mutex
	engine   *upstream.Engine
	sessions []*Session
	closed   bool
}

func Open(libraryPath string) (*Runtime, error) {
	if libraryPath == "" {
		return nil, errors.New("library path is empty")
	}
	engine, err := upstream.NewEngine(
		upstream.WithLibraryPath(libraryPath),
		upstream.WithEnvName("go-insightface"),
	)
	if err != nil {
		return nil, fmt.Errorf("load ONNX Runtime: %w", err)
	}
	return &Runtime{engine: engine}, nil
}

func (r *Runtime) Load(modelPath string, threads int) (*Session, error) {
	if modelPath == "" {
		return nil, errors.New("model path is empty")
	}
	return r.loadLocked(modelPath, threads, func(opts *upstream.SessionOptions) (*upstream.Session, error) {
		return r.engine.NewSession(modelPath, opts)
	})
}

// LoadReader creates a session from an ONNX model stream.
// modelLabel is used only for error context.
func (r *Runtime) LoadReader(modelLabel string, modelReader io.Reader, threads int) (*Session, error) {
	if modelReader == nil {
		return nil, errors.New("model reader is nil")
	}
	if modelLabel == "" {
		modelLabel = "<reader>"
	}
	modelData, err := io.ReadAll(modelReader)
	if err != nil {
		return nil, fmt.Errorf("read ONNX model %q: %w", modelLabel, err)
	}
	return r.loadLocked(modelLabel, threads, func(opts *upstream.SessionOptions) (*upstream.Session, error) {
		return r.engine.NewSessionFromBytes(modelData, opts)
	})
}

// autoProviderOrder lists GPU execution providers by preference: TensorRT first
// (fastest on NVIDIA hardware), then CUDA as fallback. CPU is always present and
// needs no enable call, so it is never part of the auto-selection result.
var autoProviderOrder = []string{upstream.ProviderTensorRT, upstream.ProviderCUDA}

// selectAutoProviders returns the GPU providers from available, best first.
// Matching is exact and case-sensitive, mirroring the names returned by the
// ONNX Runtime OrtGetAvailableProviders.
func selectAutoProviders(available []string) []string {
	present := make(map[string]bool, len(available))
	for _, name := range available {
		present[name] = true
	}
	var selected []string
	for _, name := range autoProviderOrder {
		if present[name] {
			selected = append(selected, name)
		}
	}
	return selected
}

// AvailableProviders returns the execution provider names exposed by the loaded
// ONNX Runtime build. CPU is always present; GPU providers appear only in GPU
// builds.
func (r *Runtime) AvailableProviders() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.engine == nil {
		return nil
	}
	return append([]string(nil), r.engine.GetAvailableProviders()...)
}

func (r *Runtime) loadLocked(modelLabel string, threads int, load func(opts *upstream.SessionOptions) (*upstream.Session, error)) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.engine == nil {
		return nil, ErrClosed
	}
	providers := selectAutoProviders(r.engine.GetAvailableProviders())
	session, err := r.newSessionLocked(modelLabel, threads, providers, load)
	if err != nil && len(providers) > 0 {
		log.Printf("[go-insightface] onnxruntime: load %q with providers %v failed: %v; falling back to CPU", modelLabel, providers, err)
		session, err = r.newSessionLocked(modelLabel, threads, nil, load)
	}
	if err != nil {
		return nil, err
	}
	providers = append(providers, upstream.ProviderCPU)
	inputs := append([]string(nil), tensorNames(session.Inputs)...)
	outputs := append([]string(nil), tensorNames(session.Outputs)...)
	if len(inputs) != 1 || len(outputs) == 0 {
		session.Destroy()
		return nil, fmt.Errorf("load ONNX model %q: model must have one input and at least one output, got %d inputs and %d outputs", modelLabel, len(inputs), len(outputs))
	}
	wrapped := &Session{engine: r.engine, session: session, inputName: inputs[0], outputNames: outputs, providers: providers}
	r.sessions = append(r.sessions, wrapped)
	if len(providers) > 1 {
		log.Printf("[go-insightface] onnxruntime: session %q using execution providers %v", modelLabel, providers)
	}
	return wrapped, nil
}

// newSessionLocked creates a session with the given GPU providers enabled (nil
// for CPU only) and returns it unwrapped. On error the partially created session
// is released and the error carries the model label context.
func (r *Runtime) newSessionLocked(modelLabel string, threads int, providers []string, load func(opts *upstream.SessionOptions) (*upstream.Session, error)) (*upstream.Session, error) {
	opts, err := r.engine.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options for %q: %w", modelLabel, err)
	}
	defer opts.Destroy()
	if threads > 0 {
		if err := opts.SetIntraOpNumThreads(int32(threads)); err != nil {
			return nil, fmt.Errorf("configure session options for %q: %w", modelLabel, err)
		}
	}
	for _, provider := range providers {
		var enableErr error
		switch provider {
		case upstream.ProviderTensorRT:
			enableErr = opts.EnableTensorRT(nil)
		case upstream.ProviderCUDA:
			enableErr = opts.EnableCUDA(nil)
		default:
			enableErr = fmt.Errorf("no automatic enable path for provider %q", provider)
		}
		if enableErr != nil {
			return nil, fmt.Errorf("enable provider %q for %q: %w", provider, modelLabel, enableErr)
		}
	}
	session, err := load(opts)
	if err != nil {
		return nil, fmt.Errorf("load ONNX model %q: %w", modelLabel, err)
	}
	return session, nil
}

func tensorNames(infos []upstream.TensorInfo) []string {
	names := make([]string, len(infos))
	for i, info := range infos {
		names[i] = info.Name
	}
	return names
}

func (r *Runtime) Close() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	sessions := append([]*Session(nil), r.sessions...)
	r.sessions = nil
	engine := r.engine
	r.engine = nil
	r.mu.Unlock()

	for _, session := range sessions {
		session.Close()
	}
	if engine != nil {
		engine.Destroy()
	}
	return nil
}

type Session struct {
	mu          sync.RWMutex
	engine      *upstream.Engine
	session     *upstream.Session
	inputName   string
	outputNames []string
	providers   []string
	closed      bool
}

// Providers returns the execution providers the session was created with,
// GPU providers first and CPU always last.
func (s *Session) Providers() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]string(nil), s.providers...)
}

func (s *Session) Run(ctx context.Context, input []float32, shape []int64) ([]Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	count, err := elementCount(shape)
	if err != nil {
		return nil, fmt.Errorf("invalid input shape: %w", err)
	}
	if count != len(input) {
		return nil, fmt.Errorf("input contains %d values, shape requires %d", len(input), count)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.session == nil || s.engine == nil {
		return nil, ErrClosed
	}
	inputTensor, err := s.engine.NewTensor(shape, input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs, err := s.session.Run(map[string]*upstream.Value{s.inputName: inputTensor})
	if err != nil {
		return nil, fmt.Errorf("run ONNX inference: %w", err)
	}
	defer closeValues(outputs)

	result := make([]Tensor, 0, len(s.outputNames))
	for _, name := range s.outputNames {
		value := outputs[name]
		if value == nil {
			return nil, fmt.Errorf("model did not return output %q", name)
		}
		data, outputShape, err := tensorData(value)
		if err != nil {
			return nil, fmt.Errorf("read output %q: %w", name, err)
		}
		tensor := Tensor{Name: name, Shape: outputShape, Data: data}
		if err := tensor.Validate(); err != nil {
			return nil, err
		}
		result = append(result, tensor)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func tensorData(value *upstream.Value) ([]float32, []int64, error) {
	data, err := upstream.GetTensorData[float32](value)
	if err != nil {
		return nil, nil, err
	}
	shape, err := value.GetShape()
	if err != nil {
		return nil, nil, err
	}
	return data, shape, nil
}

func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}
}

func closeValues(values map[string]*upstream.Value) {
	for _, value := range values {
		if value != nil {
			value.Destroy()
		}
	}
}
