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
	enrThreshold          float32
	enrExitThreshold      float32
	snrThreshold          float32
	holdDuration          int
	triggerThreshold      int
	useDuringInitialPhase bool
	numCaptureChannels    int
	nearendState          bool
	triggerCounters       []int
	holdCounters          []int

	// The counters above for up to inlineChannels channels.
	inline struct {
		triggerCounters [inlineChannels]int
		holdCounters    [inlineChannels]int
	}
}

var _ NearendDetector = (*DominantNearendDetector)(nil)

func NewDominantNearendDetector(config *DominantNearendDetectionConfig, numCaptureChannels int) *DominantNearendDetector {
	// The counters live in the detector, which must not be copied (see
	// channelSlice).
	d := &DominantNearendDetector{
		enrThreshold:          config.EnrThreshold,
		enrExitThreshold:      config.EnrExitThreshold,
		snrThreshold:          config.SnrThreshold,
		holdDuration:          config.HoldDuration,
		triggerThreshold:      config.TriggerThreshold,
		useDuringInitialPhase: config.UseDuringInitialPhase,
		numCaptureChannels:    numCaptureChannels,
	}
	d.triggerCounters = channelSlice(d.inline.triggerCounters[:], numCaptureChannels)
	d.holdCounters = channelSlice(d.inline.holdCounters[:], numCaptureChannels)
	return d
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

	for ch := 0; ch < d.numCaptureChannels; ch++ {
		neSum := supLowFrequencyEnergy(&nearendSpectrum[ch])
		echoSum := supLowFrequencyEnergy(&residualEchoSpectrum[ch])
		noiseSum := supLowFrequencyEnergy(&comfortNoiseSpectrum[ch])

		// Detect strong active nearend if the nearend is sufficiently stronger
		// than the echo and the nearend noise.
		if (!initialState || d.useDuringInitialPhase) &&
			echoSum < d.enrThreshold*neSum &&
			neSum > d.snrThreshold*noiseSum {
			d.triggerCounters[ch]++
			if d.triggerCounters[ch] >= d.triggerThreshold {
				// After a period of strong active nearend activity, flag
				// nearend mode.
				d.holdCounters[ch] = d.holdDuration
				d.triggerCounters[ch] = d.triggerThreshold
			}
		} else {
			// Forget previously detected strong active nearend activity.
			d.triggerCounters[ch] = max(0, d.triggerCounters[ch]-1)
		}

		// Exit nearend-state early at strong echo.
		if echoSum > d.enrExitThreshold*neSum &&
			echoSum > d.snrThreshold*noiseSum {
			d.holdCounters[ch] = 0
		}

		// Remain in any nearend mode for a certain duration.
		d.holdCounters[ch] = max(0, d.holdCounters[ch]-1)
		d.nearendState = d.nearendState || d.holdCounters[ch] > 0
	}
}
