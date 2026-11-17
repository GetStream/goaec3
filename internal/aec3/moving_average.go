package aec3

// MovingAverage mirrors aec3::MovingAverage (moving_average.h): the average
// of the current input and the memLen-1 previous inputs.
type MovingAverage struct {
	numElem  int
	memLen   int
	scaling  float32
	memory   []float32
	memIndex int
}

// NewMovingAverage accepts inputs of length numElem and averages over memLen
// inputs.
func NewMovingAverage(numElem, memLen int) *MovingAverage {
	return &MovingAverage{
		numElem: numElem,
		memLen:  memLen - 1,
		scaling: 1 / float32(memLen),
		memory:  make([]float32, numElem*(memLen-1)),
	}
}

// Average stores the average of input and the memLen-1 previous inputs in
// output.
func (m *MovingAverage) Average(input, output []float32) {
	n := m.numElem
	input = input[:n]
	output = output[:n]
	copy(output, input)
	for i := 0; i < len(m.memory); i += n {
		mem := m.memory[i : i+n]
		for k := range output {
			output[k] = mem[k] + output[k]
		}
	}
	for k := range output {
		output[k] *= m.scaling
	}
	if m.memLen > 0 {
		copy(m.memory[m.memIndex*n:], input)
		m.memIndex = (m.memIndex + 1) % m.memLen
	}
}
