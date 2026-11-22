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
// subbands (subband_nearend_detector.h).
type SubbandNearendDetector struct {
	config                SubbandNearendDetectionConfig
	numCaptureChannels    int
	nearendSmoothers      []MovingAverage
	oneOverSubbandLength1 float32
	oneOverSubbandLength2 float32
	nearendState          bool
}

func NewSubbandNearendDetector(config *SubbandNearendDetectionConfig, numCaptureChannels int) *SubbandNearendDetector {
	d := &SubbandNearendDetector{}
	d.init(config, numCaptureChannels)
	return d
}

func (d *SubbandNearendDetector) init(config *SubbandNearendDetectionConfig, numCaptureChannels int) {
	*d = SubbandNearendDetector{
		config:                *config,
		numCaptureChannels:    numCaptureChannels,
		oneOverSubbandLength1: 1 / float32(config.Subband1.High-config.Subband1.Low+1),
		oneOverSubbandLength2: 1 / float32(config.Subband2.High-config.Subband2.Low+1),
	}
	d.nearendSmoothers = make([]MovingAverage, numCaptureChannels)
	for ch := range d.nearendSmoothers {
		d.nearendSmoothers[ch] = *NewMovingAverage(FftLengthBy2Plus1, config.NearendAverageBlocks)
	}
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
	smoothers := d.nearendSmoothers
	for ch := range smoothers {
		noise := &comfortNoiseSpectrum[ch]
		var nearend [FftLengthBy2Plus1]float32
		smoothers[ch].Average(nearendSpectrum[ch][:], nearend[:])

		noisePower := supSumRange(noise[s1.Low:s1.High+1]) * d.oneOverSubbandLength1

		nearendPowerSubband1 := supSumRange(nearend[s1.Low:s1.High+1]) * d.oneOverSubbandLength1

		nearendPowerSubband2 := supSumRange(nearend[s2.Low:s2.High+1]) * d.oneOverSubbandLength2

		// One channel in the nearend state is enough.
		d.nearendState = d.nearendState ||
			(nearendPowerSubband1 < d.config.NearendThreshold*nearendPowerSubband2 &&
				(nearendPowerSubband1 > d.config.SnrThreshold*noisePower))
	}
}
