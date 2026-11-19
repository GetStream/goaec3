package aec3

// MovingAverage mirrors aec3::MovingAverage (moving_average.h): the average
// of the current input and the memLen-1 previous inputs.
type MovingAverage struct {
	numElem  int
	memLen   int
	scaling  float32
	memory   []float32
	memIndex int

	// The memory of an average of up to movingAverageInlineSize samples.
	inline [movingAverageInlineSize]float32
}

// movingAverageInlineSize is the memory of the suppressor's default nearend
// smoothing: FftLengthBy2Plus1 bins over 4 blocks.
const movingAverageInlineSize = FftLengthBy2Plus1 * 3

// NewMovingAverage accepts inputs of length numElem and averages over memLen
// inputs. A memory of up to movingAverageInlineSize samples lives in the
// MovingAverage, which must not be copied (see channelSlice).
func NewMovingAverage(numElem, memLen int) *MovingAverage {
	m := &MovingAverage{
		numElem: numElem,
		memLen:  memLen - 1,
		scaling: 1 / float32(memLen),
	}
	m.memory = channelSlice(m.inline[:], numElem*(memLen-1))
	return m
}

// Average stores the average of input and the memLen-1 previous inputs in
// output.
func (m *MovingAverage) Average(input, output []float32) {
	n := m.numElem
	input = input[:n]
	output = output[:n]
	copy(output, input)
	for i := 0; i < len(m.memory); i += n {
		kernAdd(m.memory[i:i+n], output, output)
	}
	kernScale(output, m.scaling, output)
	if m.memLen > 0 {
		copy(m.memory[m.memIndex*n:], input)
		m.memIndex = (m.memIndex + 1) % m.memLen
	}
}
