package scrfd

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
)

var (
	ErrInvalidConfig         = errors.New("scrfd: invalid decoder config")
	ErrInvalidTensorShape    = errors.New("scrfd: invalid tensor shape")
	ErrUnknownOutputChannels = errors.New("scrfd: unknown output channels")
	ErrIncompleteOutputSet   = errors.New("scrfd: incomplete output set")
	ErrAmbiguousOutputLayout = errors.New("scrfd: ambiguous output layout")
	ErrInvalidTensorValue    = errors.New("scrfd: non-finite tensor value")
)

type Tensor struct {
	Shape []int64
	Data  []float32
}

type DecoderConfig struct {
	InputWidth     int
	InputHeight    int
	Strides        []int
	ScoreThreshold float32
	NMSThreshold   float32
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

type Detection struct {
	Score     float32
	Box       Box
	Landmarks [5]Point
	Stride    int
}

type classifiedOutputs struct {
	anchorCount int
	byStride    map[int]outputGroup
}

type outputGroup struct {
	scores    tensorView
	boxes     tensorView
	landmarks tensorView
}

type tensorView struct {
	rows     int
	channels int
	data     []float32
}

type typedTensor struct {
	kind string
	view tensorView
}

func Decode(outputs []Tensor, config DecoderConfig) ([]Detection, error) {
	layout, err := classifyOutputs(outputs, config)
	if err != nil {
		return nil, err
	}

	detections := make([]Detection, 0)
	strides := append([]int(nil), config.Strides...)
	sort.Ints(strides)
	for _, stride := range strides {
		group := layout.byStride[stride]
		gridWidth := ceilDiv(config.InputWidth, stride)
		for row := 0; row < group.scores.rows; row++ {
			score := group.scores.data[row]
			if score < config.ScoreThreshold {
				continue
			}

			cellIndex := row / layout.anchorCount
			x := cellIndex % gridWidth
			y := cellIndex / gridWidth
			centerX := float32(x * stride)
			centerY := float32(y * stride)

			boxOffset := row * 4
			left := group.boxes.data[boxOffset] * float32(stride)
			top := group.boxes.data[boxOffset+1] * float32(stride)
			right := group.boxes.data[boxOffset+2] * float32(stride)
			bottom := group.boxes.data[boxOffset+3] * float32(stride)

			detection := Detection{
				Score: score,
				Box: Box{
					Left:   centerX - left,
					Top:    centerY - top,
					Right:  centerX + right,
					Bottom: centerY + bottom,
				},
				Stride: stride,
			}

			landmarkOffset := row * 10
			for i := range detection.Landmarks {
				detection.Landmarks[i] = Point{
					X: centerX + group.landmarks.data[landmarkOffset+i*2]*float32(stride),
					Y: centerY + group.landmarks.data[landmarkOffset+i*2+1]*float32(stride),
				}
			}

			detections = append(detections, detection)
		}
	}

	return applyNMS(detections, config.NMSThreshold), nil
}

func classifyOutputs(outputs []Tensor, config DecoderConfig) (classifiedOutputs, error) {
	if err := config.validate(); err != nil {
		return classifiedOutputs{}, err
	}
	if len(outputs) == 0 {
		return classifiedOutputs{}, ErrIncompleteOutputSet
	}

	typed := make(map[string][]typedTensor, 3)
	for _, output := range outputs {
		view, err := normalizeTensor(output)
		if err != nil {
			return classifiedOutputs{}, err
		}

		kind := ""
		switch view.channels {
		case 1:
			kind = "scores"
		case 4:
			kind = "boxes"
		case 10:
			kind = "landmarks"
		default:
			return classifiedOutputs{}, fmt.Errorf("channels=%d: %w", view.channels, ErrUnknownOutputChannels)
		}
		typed[kind] = append(typed[kind], typedTensor{kind: kind, view: view})
	}

	if len(typed["scores"]) != len(config.Strides) || len(typed["boxes"]) != len(config.Strides) || len(typed["landmarks"]) != len(config.Strides) {
		return classifiedOutputs{}, ErrIncompleteOutputSet
	}

	scoreAssignments, anchors, err := assignByStride(typed["scores"], config)
	if err != nil {
		return classifiedOutputs{}, err
	}
	boxAssignments, boxAnchors, err := assignByStride(typed["boxes"], config)
	if err != nil {
		return classifiedOutputs{}, err
	}
	landmarkAssignments, landmarkAnchors, err := assignByStride(typed["landmarks"], config)
	if err != nil {
		return classifiedOutputs{}, err
	}
	if anchors != boxAnchors || anchors != landmarkAnchors {
		return classifiedOutputs{}, fmt.Errorf("score anchors=%d box anchors=%d landmark anchors=%d: %w", anchors, boxAnchors, landmarkAnchors, ErrAmbiguousOutputLayout)
	}

	byStride := make(map[int]outputGroup, len(config.Strides))
	for _, stride := range config.Strides {
		byStride[stride] = outputGroup{
			scores:    scoreAssignments[stride].view,
			boxes:     boxAssignments[stride].view,
			landmarks: landmarkAssignments[stride].view,
		}
	}

	return classifiedOutputs{
		anchorCount: anchors,
		byStride:    byStride,
	}, nil
}

func assignByStride(tensors []typedTensor, config DecoderConfig) (map[int]typedTensor, int, error) {
	strides := append([]int(nil), config.Strides...)
	sort.Ints(strides)

	indexes := make([]int, len(tensors))
	for i := range indexes {
		indexes[i] = i
	}

	var (
		matchCount  int
		result      map[int]typedTensor
		anchorCount int
	)
	for _, permutation := range permutations(indexes) {
		current := make(map[int]typedTensor, len(strides))
		candidateAnchors := 0
		valid := true
		for strideIndex, stride := range strides {
			tensor := tensors[permutation[strideIndex]]
			cells := ceilDiv(config.InputWidth, stride) * ceilDiv(config.InputHeight, stride)
			if cells <= 0 || tensor.view.rows%cells != 0 {
				valid = false
				break
			}
			anchors := tensor.view.rows / cells
			if anchors <= 0 {
				valid = false
				break
			}
			if strideIndex == 0 {
				candidateAnchors = anchors
			} else if anchors != candidateAnchors {
				valid = false
				break
			}
			current[stride] = tensor
		}
		if !valid {
			continue
		}
		matchCount++
		if matchCount > 1 {
			return nil, 0, ErrAmbiguousOutputLayout
		}
		result = current
		anchorCount = candidateAnchors
	}
	if matchCount == 0 {
		return nil, 0, ErrIncompleteOutputSet
	}
	return result, anchorCount, nil
}

func normalizeTensor(tensor Tensor) (tensorView, error) {
	rows, channels, err := rowsAndChannels(tensor.Shape)
	if err != nil {
		return tensorView{}, err
	}
	if rows <= 0 || channels <= 0 {
		return tensorView{}, ErrInvalidTensorShape
	}
	if rows*channels != len(tensor.Data) {
		return tensorView{}, fmt.Errorf("shape requires %d values, got %d: %w", rows*channels, len(tensor.Data), ErrInvalidTensorShape)
	}
	for _, value := range tensor.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return tensorView{}, ErrInvalidTensorValue
		}
	}
	return tensorView{
		rows:     rows,
		channels: channels,
		data:     tensor.Data,
	}, nil
}

func rowsAndChannels(shape []int64) (int, int, error) {
	switch len(shape) {
	case 2:
		if shape[0] <= 0 || shape[1] <= 0 {
			return 0, 0, ErrInvalidTensorShape
		}
		return int(shape[0]), int(shape[1]), nil
	case 3:
		if shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 {
			return 0, 0, ErrInvalidTensorShape
		}
		return int(shape[1]), int(shape[2]), nil
	default:
		return 0, 0, ErrInvalidTensorShape
	}
}

func applyNMS(detections []Detection, threshold float32) []Detection {
	if len(detections) == 0 {
		return nil
	}

	sorted := slices.Clone(detections)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	kept := make([]Detection, 0, len(sorted))
	for _, detection := range sorted {
		suppressed := false
		for _, existing := range kept {
			if iou(detection.Box, existing.Box) > threshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, detection)
		}
	}
	return kept
}

func iou(a, b Box) float32 {
	intersectionLeft := max32(a.Left, b.Left)
	intersectionTop := max32(a.Top, b.Top)
	intersectionRight := min32(a.Right, b.Right)
	intersectionBottom := min32(a.Bottom, b.Bottom)
	intersectionWidth := max32(0, intersectionRight-intersectionLeft+1)
	intersectionHeight := max32(0, intersectionBottom-intersectionTop+1)
	intersectionArea := intersectionWidth * intersectionHeight
	if intersectionArea <= 0 {
		return 0
	}

	areaA := max32(0, a.Right-a.Left+1) * max32(0, a.Bottom-a.Top+1)
	areaB := max32(0, b.Right-b.Left+1) * max32(0, b.Bottom-b.Top+1)
	unionArea := areaA + areaB - intersectionArea
	if unionArea <= 0 {
		return 0
	}
	return intersectionArea / unionArea
}

func (c DecoderConfig) validate() error {
	if c.InputWidth <= 0 || c.InputHeight <= 0 {
		return ErrInvalidConfig
	}
	if len(c.Strides) == 0 {
		return ErrInvalidConfig
	}
	seen := make(map[int]struct{}, len(c.Strides))
	for _, stride := range c.Strides {
		if stride <= 0 {
			return ErrInvalidConfig
		}
		if _, ok := seen[stride]; ok {
			return ErrInvalidConfig
		}
		seen[stride] = struct{}{}
	}
	if math.IsNaN(float64(c.ScoreThreshold)) || math.IsInf(float64(c.ScoreThreshold), 0) || c.ScoreThreshold < 0 || c.ScoreThreshold > 1 {
		return ErrInvalidConfig
	}
	if math.IsNaN(float64(c.NMSThreshold)) || math.IsInf(float64(c.NMSThreshold), 0) || c.NMSThreshold < 0 || c.NMSThreshold > 1 {
		return ErrInvalidConfig
	}
	return nil
}

func ceilDiv(value, divisor int) int {
	return int(math.Ceil(float64(value) / float64(divisor)))
}

func permutations(items []int) [][]int {
	if len(items) == 1 {
		return [][]int{{items[0]}}
	}
	result := make([][]int, 0)
	for i, item := range items {
		rest := make([]int, 0, len(items)-1)
		rest = append(rest, items[:i]...)
		rest = append(rest, items[i+1:]...)
		for _, tail := range permutations(rest) {
			permutation := append([]int{item}, tail...)
			result = append(result, permutation)
		}
	}
	return result
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
