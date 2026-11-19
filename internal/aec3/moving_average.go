package aec3

// MovingAverage mirrors aec3::MovingAverage (moving_average.h): the average
// of the current input and the memLen-1 previous inputs.
type MovingAverage struct {
	movingAverageState
	memory []float32
}

// movingAverageState is a MovingAverage without its memory, which the caller
// keeps and passes to average; init returns its size. It holds no pointer.
type movingAverageState struct {
	numElem  int
	memLen   int
	scaling  float32
	memIndex int
}

// NewMovingAverage accepts inputs of length numElem and averages over memLen
// inputs.
func NewMovingAverage(numElem, memLen int) *MovingAverage {
	m := &MovingAverage{}
	m.memory = make([]float32, m.init(numElem, memLen))
	return m
}

// Average stores the average of input and the memLen-1 previous inputs in
// output.
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

// average stores the average of input and the memLen-1 previous inputs,
// which memory holds, in output.
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
