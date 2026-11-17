package aec3

// SubbandNearendDetectionConfig mirrors
// EchoCanceller3Config::Suppressor::SubbandNearendDetection.
type SubbandNearendDetectionConfig struct {
	NearendAverageBlocks int
	Subband1             SubbandRegion
	Subband2             SubbandRegion
	NearendThreshold     float32
	SnrThreshold         float32
}

// supMovingAverage mirrors aec3::MovingAverage (moving_average.cc).
type supMovingAverage struct {
	numElem  int
	memLen   int
	scaling  float32
	memory   []float32
	memIndex int
}

func newSupMovingAverage(numElem, memLen int) supMovingAverage {
	return supMovingAverage{
		numElem: numElem,
		memLen:  memLen - 1,
		scaling: 1 / float32(memLen),
		memory:  make([]float32, numElem*(memLen-1)),
	}
}

func (m *supMovingAverage) Average(input, output []float32) {
	input = input[:m.numElem]
	output = output[:m.numElem]
	// Sum all contributions.
	copy(output, input)
	for i := 0; i < len(m.memory); i += m.numElem {
		mem := m.memory[i : i+m.numElem]
		for k, v := range mem {
			output[k] += v
		}
	}
	// Divide by mem_len_.
	for k := range output {
		output[k] *= m.scaling
	}
	// Update memory.
	if m.memLen > 0 {
		copy(m.memory[m.memIndex*m.numElem:], input)
		m.memIndex = (m.memIndex + 1) % m.memLen
	}
}

// SubbandNearendDetector detects nearend activity from the power in two
// subbands.
type SubbandNearendDetector struct {
	config                SubbandNearendDetectionConfig
	numCaptureChannels    int
	nearendSmoothers      []supMovingAverage
	oneOverSubbandLength1 float32
	oneOverSubbandLength2 float32
	nearendState          bool
}

var _ NearendDetector = (*SubbandNearendDetector)(nil)

func NewSubbandNearendDetector(config *SubbandNearendDetectionConfig, numCaptureChannels int) *SubbandNearendDetector {
	d := &SubbandNearendDetector{
		config:                *config,
		numCaptureChannels:    numCaptureChannels,
		nearendSmoothers:      make([]supMovingAverage, numCaptureChannels),
		oneOverSubbandLength1: 1 / float32(config.Subband1.High-config.Subband1.Low+1),
		oneOverSubbandLength2: 1 / float32(config.Subband2.High-config.Subband2.Low+1),
	}
	for ch := range d.nearendSmoothers {
		d.nearendSmoothers[ch] = newSupMovingAverage(FftLengthBy2Plus1, config.NearendAverageBlocks)
	}
	return d
}

func (d *SubbandNearendDetector) IsNearendState() bool { return d.nearendState }

func supSumRange(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v
	}
	return sum
}

func (d *SubbandNearendDetector) Update(nearendSpectrum, _,
	comfortNoiseSpectrum [][FftLengthBy2Plus1]float32, _ bool) {
	d.nearendState = false
	s1, s2 := d.config.Subband1, d.config.Subband2
	for ch := 0; ch < d.numCaptureChannels; ch++ {
		noise := &comfortNoiseSpectrum[ch]
		var nearend [FftLengthBy2Plus1]float32
		d.nearendSmoothers[ch].Average(nearendSpectrum[ch][:], nearend[:])

		// Noise power of the first region.
		noisePower := supSumRange(noise[s1.Low:s1.High+1]) * d.oneOverSubbandLength1

		// Nearend power of the first region.
		nearendPowerSubband1 := supSumRange(nearend[s1.Low:s1.High+1]) * d.oneOverSubbandLength1

		// Nearend power of the second region.
		nearendPowerSubband2 := supSumRange(nearend[s2.Low:s2.High+1]) * d.oneOverSubbandLength2

		// One channel is sufficient to trigger nearend state.
		d.nearendState = d.nearendState ||
			(nearendPowerSubband1 < d.config.NearendThreshold*nearendPowerSubband2 &&
				(nearendPowerSubband1 > d.config.SnrThreshold*noisePower))
	}
}
