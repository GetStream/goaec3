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

// SubbandNearendDetector detects nearend activity from the power in two
// subbands.
type SubbandNearendDetector struct {
	config                SubbandNearendDetectionConfig
	numCaptureChannels    int
	nearendSmoothers      []*MovingAverage
	oneOverSubbandLength1 float32
	oneOverSubbandLength2 float32
	nearendState          bool
}

var _ NearendDetector = (*SubbandNearendDetector)(nil)

func NewSubbandNearendDetector(config *SubbandNearendDetectionConfig, numCaptureChannels int) *SubbandNearendDetector {
	d := &SubbandNearendDetector{
		config:                *config,
		numCaptureChannels:    numCaptureChannels,
		nearendSmoothers:      make([]*MovingAverage, numCaptureChannels),
		oneOverSubbandLength1: 1 / float32(config.Subband1.High-config.Subband1.Low+1),
		oneOverSubbandLength2: 1 / float32(config.Subband2.High-config.Subband2.Low+1),
	}
	for ch := range d.nearendSmoothers {
		d.nearendSmoothers[ch] = NewMovingAverage(FftLengthBy2Plus1, config.NearendAverageBlocks)
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
