package scrfd

import (
	"errors"
	"math"
	"testing"
)

func TestClassifyOutputsByShapeWithoutNames(t *testing.T) {
	config := DecoderConfig{
		InputWidth:  64,
		InputHeight: 64,
		Strides:     []int{8, 16, 32},
	}

	outputs := []Tensor{
		makeTensor([2]int{16, 1}, false),
		makeTensor([2]int{4, 10}, true),
		makeTensor([2]int{64, 4}, true),
		makeTensor([2]int{4, 4}, false),
		makeTensor([2]int{64, 1}, true),
		makeTensor([2]int{16, 10}, false),
		makeTensor([2]int{4, 1}, false),
		makeTensor([2]int{16, 4}, true),
		makeTensor([2]int{64, 10}, false),
	}

	layout, err := classifyOutputs(outputs, config)
	if err != nil {
		t.Fatalf("classifyOutputs() error = %v", err)
	}

	if layout.anchorCount != 1 {
		t.Fatalf("classifyOutputs().anchorCount = %d, want 1", layout.anchorCount)
	}
	if layout.byStride[8].scores.rows != 64 || layout.byStride[16].scores.rows != 16 || layout.byStride[32].scores.rows != 4 {
		t.Fatalf("classifyOutputs() rows = %+v", layout.byStride)
	}
	if layout.byStride[8].boxes.channels != 4 || layout.byStride[16].landmarks.channels != 10 {
		t.Fatalf("classifyOutputs() channels = %+v", layout.byStride)
	}
}

func TestClassifyOutputsRejectsAmbiguousAssignments(t *testing.T) {
	config := DecoderConfig{
		InputWidth:  8,
		InputHeight: 8,
		Strides:     []int{8, 16, 32},
	}

	outputs := []Tensor{
		makeTensor([2]int{1, 1}, false),
		makeTensor([2]int{1, 1}, true),
		makeTensor([2]int{1, 1}, false),
		makeTensor([2]int{1, 4}, true),
		makeTensor([2]int{1, 4}, false),
		makeTensor([2]int{1, 4}, true),
		makeTensor([2]int{1, 10}, false),
		makeTensor([2]int{1, 10}, true),
		makeTensor([2]int{1, 10}, false),
	}

	_, err := classifyOutputs(outputs, config)
	if !errors.Is(err, ErrAmbiguousOutputLayout) {
		t.Fatalf("classifyOutputs() error = %v, want %v", err, ErrAmbiguousOutputLayout)
	}
}

func TestDecodeDecodesBoxLandmarksAndThresholds(t *testing.T) {
	config := DecoderConfig{
		InputWidth:     64,
		InputHeight:    64,
		Strides:        []int{8, 16, 32},
		ScoreThreshold: 0.5,
		NMSThreshold:   0.5,
	}

	score8 := makeTensor([2]int{64, 1}, false)
	box8 := makeTensor([2]int{64, 4}, true)
	landmark8 := makeTensor([2]int{64, 10}, false)
	score16 := makeTensor([2]int{16, 1}, true)
	box16 := makeTensor([2]int{16, 4}, false)
	landmark16 := makeTensor([2]int{16, 10}, true)
	score32 := makeTensor([2]int{4, 1}, false)
	box32 := makeTensor([2]int{4, 4}, true)
	landmark32 := makeTensor([2]int{4, 10}, false)

	row := 10 // stride 8, x=2, y=1
	score8.Data[row] = 0.95
	copy(box8.Data[row*4:(row+1)*4], []float32{1, 1, 2, 2})
	copy(landmark8.Data[row*10:(row+1)*10], []float32{
		0, 0,
		1, 0,
		0, 1,
		-1, 0,
		0, -1,
	})
	score16.Data[1] = 0.49
	score32.Data[0] = 0.1

	outputs := []Tensor{score16, landmark32, box8, box32, score8, landmark16, score32, box16, landmark8}

	detections, err := Decode(outputs, config)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if len(detections) != 1 {
		t.Fatalf("len(Decode()) = %d, want 1", len(detections))
	}

	got := detections[0]
	if got.Stride != 8 {
		t.Fatalf("Decode()[0].Stride = %d, want 8", got.Stride)
	}
	if diff := math.Abs(float64(got.Score - 0.95)); diff > 1e-6 {
		t.Fatalf("Decode()[0].Score = %f, want 0.95", got.Score)
	}
	if got.Box != (Box{Left: 8, Top: 0, Right: 32, Bottom: 24}) {
		t.Fatalf("Decode()[0].Box = %+v", got.Box)
	}
	wantLandmarks := [5]Point{
		{X: 16, Y: 8},
		{X: 24, Y: 8},
		{X: 16, Y: 16},
		{X: 8, Y: 8},
		{X: 16, Y: 0},
	}
	if got.Landmarks != wantLandmarks {
		t.Fatalf("Decode()[0].Landmarks = %+v, want %+v", got.Landmarks, wantLandmarks)
	}
}

func TestApplyNMSSuppressesLowerScoreOverlap(t *testing.T) {
	detections := []Detection{
		{Score: 0.9, Box: Box{Left: 0, Top: 0, Right: 10, Bottom: 10}},
		{Score: 0.8, Box: Box{Left: 1, Top: 1, Right: 9, Bottom: 9}},
		{Score: 0.7, Box: Box{Left: 20, Top: 20, Right: 30, Bottom: 30}},
	}

	filtered := applyNMS(detections, 0.5)
	if len(filtered) != 2 {
		t.Fatalf("len(applyNMS()) = %d, want 2", len(filtered))
	}
	if filtered[0].Score != 0.9 || filtered[1].Score != 0.7 {
		t.Fatalf("applyNMS() = %+v", filtered)
	}
}

func TestApplyNMSMatchesInsightFaceInclusiveCoordinates(t *testing.T) {
	detections := []Detection{
		{Score: 0.9, Box: Box{Left: 0, Top: 0, Right: 10, Bottom: 10}},
		{Score: 0.8, Box: Box{Left: 1, Top: 1, Right: 9, Bottom: 9}},
	}

	filtered := applyNMS(detections, 0.65)
	if len(filtered) != 1 {
		t.Fatalf("len(applyNMS()) = %d, want 1", len(filtered))
	}
}

func TestDecodeRejectsInvalidTensorLayouts(t *testing.T) {
	config := DecoderConfig{
		InputWidth:  64,
		InputHeight: 64,
		Strides:     []int{8, 16, 32},
	}

	tests := []struct {
		name    string
		outputs []Tensor
		want    error
	}{
		{
			name: "unsupported rank",
			outputs: []Tensor{
				{Shape: []int64{1, 1, 1, 1}, Data: make([]float32, 1)},
			},
			want: ErrInvalidTensorShape,
		},
		{
			name: "unsupported channels",
			outputs: []Tensor{
				{Shape: []int64{64, 3}, Data: make([]float32, 64*3)},
			},
			want: ErrUnknownOutputChannels,
		},
		{
			name: "non-finite value",
			outputs: []Tensor{
				{Shape: []int64{1, 1}, Data: []float32{float32(math.NaN())}},
			},
			want: ErrInvalidTensorValue,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := classifyOutputs(tt.outputs, config)
			if !errors.Is(err, tt.want) {
				t.Fatalf("classifyOutputs() error = %v, want %v", err, tt.want)
			}
		})
	}
}

func makeTensor(shape [2]int, batched bool) Tensor {
	var dims []int64
	if batched {
		dims = []int64{1, int64(shape[0]), int64(shape[1])}
	} else {
		dims = []int64{int64(shape[0]), int64(shape[1])}
	}
	return Tensor{
		Shape: dims,
		Data:  make([]float32, shape[0]*shape[1]),
	}
}
