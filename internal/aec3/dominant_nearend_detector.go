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
	*dominantNearendState
	more *dominantNearendMore // with more than inlineChannels channels
}

// dominantNearendState is the state of a DominantNearendDetector but its
// pointer.
type dominantNearendState struct {
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

// dominantNearendMore holds the counters of a DominantNearendDetector with
// more than inlineChannels capture channels.
type dominantNearendMore struct {
	counters []dominantNearendCounters
}

// bind makes d the view of st.
func (d *DominantNearendDetector) bind(st *dominantNearendState, b *binder) {
	d.dominantNearendState = st
	if b != nil {
		bindPtr(b, &d.more, &b.x.dominantNearendMore)
	}
}

func NewDominantNearendDetector(config *DominantNearendDetectionConfig, numCaptureChannels int) *DominantNearendDetector {
	d := &DominantNearendDetector{}
	d.bind(&dominantNearendState{}, bindAlone)
	d.init(config, numCaptureChannels)
	return d
}

// init sets up d in place.
func (d *DominantNearendDetector) init(config *DominantNearendDetectionConfig, numCaptureChannels int) {
	st := d.dominantNearendState
	*d = DominantNearendDetector{}
	d.bind(st, bindAlone)
	*st = dominantNearendState{
		enrThreshold:          config.EnrThreshold,
		enrExitThreshold:      config.EnrExitThreshold,
		snrThreshold:          config.SnrThreshold,
		holdDuration:          config.HoldDuration,
		triggerThreshold:      config.TriggerThreshold,
		useDuringInitialPhase: config.UseDuringInitialPhase,
		numCaptureChannels:    numCaptureChannels,
	}
	if numCaptureChannels > inlineChannelLimit {
		d.more = &dominantNearendMore{counters: make([]dominantNearendCounters, numCaptureChannels)}
	}
}

func (d DominantNearendDetector) IsNearendState() bool { return d.nearendState }

func supLowFrequencyEnergy(spectrum *[FftLengthBy2Plus1]float32) float32 {
	var sum float32
	for _, v := range spectrum[1:16] {
		sum += v
	}
	return sum
}

func (d DominantNearendDetector) Update(nearendSpectrum, residualEchoSpectrum,
	comfortNoiseSpectrum [][FftLengthBy2Plus1]float32, initialState bool) {
	d.nearendState = false

	var more []dominantNearendCounters
	if d.more != nil {
		more = d.more.counters
	}
	counters := channelState(&d.counters, d.numCaptureChannels, more)
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
