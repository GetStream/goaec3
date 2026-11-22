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

// DominantNearendDetector detects a nearend signal that dominates the echo
// (dominant_nearend_detector.h).
type DominantNearendDetector struct {
	*dominantNearendState
	more *dominantNearendMore // with more than inlineChannels channels
}

// dominantNearendState is the state of a DominantNearendDetector except its
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

// dominantNearendCounters holds the counters of one capture channel.
type dominantNearendCounters struct {
	trigger int
	hold    int
}

// dominantNearendMore holds the counters of a DominantNearendDetector with
// more than inlineChannels capture channels.
type dominantNearendMore struct {
	counters []dominantNearendCounters
}

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

		// Strong nearend: sufficiently above the echo and the nearend noise.
		if (!initialState || d.useDuringInitialPhase) &&
			echoSum < d.enrThreshold*neSum &&
			neSum > d.snrThreshold*noiseSum {
			c.trigger++
			if c.trigger >= d.triggerThreshold {
				// A period of strong nearend sets the nearend state.
				c.hold = d.holdDuration
				c.trigger = d.triggerThreshold
			}
		} else {
			c.trigger = max(0, c.trigger-1)
		}

		// Strong echo ends the nearend state early.
		if echoSum > d.enrExitThreshold*neSum &&
			echoSum > d.snrThreshold*noiseSum {
			c.hold = 0
		}

		c.hold = max(0, c.hold-1)
		d.nearendState = d.nearendState || c.hold > 0
	}
}
