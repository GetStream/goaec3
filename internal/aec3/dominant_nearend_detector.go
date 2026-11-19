package aec3

// DominantNearendDetectionConfig mirrors
// EchoCanceller3Config::Suppressor::DominantNearendDetection.
type DominantNearendDetectionConfig struct {
	EnrThreshold             float32
	EnrExitThreshold         float32
	SnrThreshold             float32
	HoldDuration             int
	TriggerThreshold         int
	UseDuringInitialPhase    bool
	UseUnboundedEchoSpectrum bool
}

// DominantNearendDetector detects when the nearend signal dominates the echo.
type DominantNearendDetector struct {
	moreCounters          []dominantNearendCounters // with more than inlineChannels channels
	counters              [inlineChannels]dominantNearendCounters
	enrThreshold          float32
	enrExitThreshold      float32
	snrThreshold          float32
	holdDuration          int
	triggerThreshold      int
	useDuringInitialPhase bool
	numCaptureChannels    int
	nearendState          bool
}

// dominantNearendCounters are the counters of one capture channel.
type dominantNearendCounters struct {
	trigger int
	hold    int
}

var _ NearendDetector = (*DominantNearendDetector)(nil)

func NewDominantNearendDetector(config *DominantNearendDetectionConfig, numCaptureChannels int) *DominantNearendDetector {
	d := &DominantNearendDetector{}
	d.init(config, numCaptureChannels)
	return d
}

// init sets up d in place.
func (d *DominantNearendDetector) init(config *DominantNearendDetectionConfig, numCaptureChannels int) {
	*d = DominantNearendDetector{
		enrThreshold:          config.EnrThreshold,
		enrExitThreshold:      config.EnrExitThreshold,
		snrThreshold:          config.SnrThreshold,
		holdDuration:          config.HoldDuration,
		triggerThreshold:      config.TriggerThreshold,
		useDuringInitialPhase: config.UseDuringInitialPhase,
		numCaptureChannels:    numCaptureChannels,
	}
	d.moreCounters = moreChannels[dominantNearendCounters](numCaptureChannels)
}

func (d *DominantNearendDetector) IsNearendState() bool { return d.nearendState }

func supLowFrequencyEnergy(spectrum *[FftLengthBy2Plus1]float32) float32 {
	var sum float32
	for _, v := range spectrum[1:16] {
		sum += v
	}
	return sum
}

func (d *DominantNearendDetector) Update(nearendSpectrum, residualEchoSpectrum,
	comfortNoiseSpectrum [][FftLengthBy2Plus1]float32, initialState bool) {
	d.nearendState = false

	counters := channelState(&d.counters, d.numCaptureChannels, d.moreCounters)
	for ch := range counters {
		c := &counters[ch]
		neSum := supLowFrequencyEnergy(&nearendSpectrum[ch])
		echoSum := supLowFrequencyEnergy(&residualEchoSpectrum[ch])
		noiseSum := supLowFrequencyEnergy(&comfortNoiseSpectrum[ch])

		// Detect strong active nearend if the nearend is sufficiently stronger
		// than the echo and the nearend noise.
		if (!initialState || d.useDuringInitialPhase) &&
			echoSum < d.enrThreshold*neSum &&
			neSum > d.snrThreshold*noiseSum {
			c.trigger++
			if c.trigger >= d.triggerThreshold {
				// After a period of strong active nearend activity, flag
				// nearend mode.
				c.hold = d.holdDuration
				c.trigger = d.triggerThreshold
			}
		} else {
			// Forget previously detected strong active nearend activity.
			c.trigger = max(0, c.trigger-1)
		}

		// Exit nearend-state early at strong echo.
		if echoSum > d.enrExitThreshold*neSum &&
			echoSum > d.snrThreshold*noiseSum {
			c.hold = 0
		}

		// Remain in any nearend mode for a certain duration.
		c.hold = max(0, c.hold-1)
		d.nearendState = d.nearendState || c.hold > 0
	}
}
