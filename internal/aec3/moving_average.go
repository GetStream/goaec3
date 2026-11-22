package aec3

// MovingAverage averages its input with the memLen-1 inputs before it
// (moving_average.h).
type MovingAverage struct {
	movingAverageState
	memory []float32
}

// movingAverageState is a MovingAverage without its memory, which the caller
// keeps and passes to average. It holds no pointer.
type movingAverageState struct {
	numElem  int
	memLen   int
	scaling  float32
	memIndex int
}

// NewMovingAverage returns a MovingAverage of memLen inputs of numElem
// elements.
func NewMovingAverage(numElem, memLen int) *MovingAverage {
	m := &MovingAverage{}
	m.memory = make([]float32, m.init(numElem, memLen))
	return m
}

// Average sets output to the average of input and the memLen-1 inputs before
// it.
func (m *MovingAverage) Average(input, output []float32) {
	m.average(m.memory, input, output)
}

// init sets up m and returns the number of samples of its memory.
func (m *movingAverageState) init(numElem, memLen int) int {
	*m = movingAverageState{
		numElem: numElem,
		memLen:  memLen - 1,
		scaling: 1 / float32(memLen),
	}
	return numElem * (memLen - 1)
}

// average is MovingAverage.Average on the given memory.
func (m *movingAverageState) average(memory, input, output []float32) {
	n := m.numElem
	input = input[:n]
	output = output[:n]
	copy(output, input)
	for i := 0; i < len(memory); i += n {
		kernAdd(memory[i:i+n], output, output)
	}
	kernScale(output, m.scaling, output)
	if m.memLen > 0 {
		copy(memory[m.memIndex*n:], input)
		m.memIndex = (m.memIndex + 1) % m.memLen
	}
}
