package ort

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"

	upstream "github.com/shota3506/onnxruntime-purego/onnxruntime"
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
	runtime  *upstream.Runtime
	env      *upstream.Env
	sessions []*Session
	closed   bool
}

func Open(libraryPath string) (*Runtime, error) {
	rt, err := upstream.NewRuntime(libraryPath, 23)
	if err != nil {
		return nil, fmt.Errorf("load ONNX Runtime API 23: %w", err)
	}
	env, err := rt.NewEnv("go-insightface", upstream.LoggingLevelWarning)
	if err != nil {
		_ = rt.Close()
		return nil, fmt.Errorf("create ONNX Runtime environment: %w", err)
	}
	return &Runtime{runtime: rt, env: env}, nil
}

func (r *Runtime) Load(modelPath string, threads int) (*Session, error) {
	if modelPath == "" {
		return nil, errors.New("model path is empty")
	}
	return r.loadLocked(modelPath, func() (*upstream.Session, error) {
		return r.runtime.NewSession(r.env, modelPath, &upstream.SessionOptions{
			IntraOpNumThreads: threads,
		})
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
	return r.loadLocked(modelLabel, func() (*upstream.Session, error) {
		return r.runtime.NewSessionFromReader(r.env, modelReader, &upstream.SessionOptions{
			IntraOpNumThreads: threads,
		})
	})
}

func (r *Runtime) loadLocked(modelLabel string, load func() (*upstream.Session, error)) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.runtime == nil || r.env == nil {
		return nil, ErrClosed
	}
	session, err := load()
	if err != nil {
		return nil, fmt.Errorf("load ONNX model %q: %w", modelLabel, err)
	}
	inputs := append([]string(nil), session.InputNames()...)
	outputs := append([]string(nil), session.OutputNames()...)
	if len(inputs) != 1 || len(outputs) == 0 {
		session.Close()
		return nil, fmt.Errorf("load ONNX model %q: model must have one input and at least one output, got %d inputs and %d outputs", modelLabel, len(inputs), len(outputs))
	}
	wrapped := &Session{runtime: r.runtime, session: session, inputName: inputs[0], outputNames: outputs}
	r.sessions = append(r.sessions, wrapped)
	return wrapped, nil
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
	env, rt := r.env, r.runtime
	r.env, r.runtime = nil, nil
	r.mu.Unlock()

	for _, session := range sessions {
		session.Close()
	}
	if env != nil {
		env.Close()
	}
	if rt != nil {
		return rt.Close()
	}
	return nil
}

type Session struct {
	mu          sync.RWMutex
	runtime     *upstream.Runtime
	session     *upstream.Session
	inputName   string
	outputNames []string
	closed      bool
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
	if s.closed || s.session == nil || s.runtime == nil {
		return nil, ErrClosed
	}
	inputTensor, err := upstream.NewTensorValue(s.runtime, input, shape)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Close()

	outputs, err := s.session.Run(ctx, map[string]*upstream.Value{s.inputName: inputTensor})
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
		data, outputShape, err := upstream.GetTensorData[float32](value)
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

func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	if s.session != nil {
		s.session.Close()
		s.session = nil
	}
}

func closeValues(values map[string]*upstream.Value) {
	for _, value := range values {
		if value != nil {
			value.Close()
		}
	}
}
