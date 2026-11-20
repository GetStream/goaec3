package aec3

const stBlocksForOnsetDetection = stBlocksToHoldErle + 150

// SubbandErleEstimator mirrors subband_erle_estimator.h: estimates the ERLE
// per frequency band.
type SubbandErleEstimator struct {
	more                   *sbErleChannels // with more than inlineChannels channels
	numCaptureChannels     int
	useOnsetDetection      bool
	minErle                float32
	maxErle                [FftLengthBy2Plus1]float32
	useMinErleDuringOnsets bool

	// The per-channel state for up to inlineChannels channels.
	inline struct {
		accumY2              [inlineChannels][FftLengthBy2Plus1]float32
		accumE2              [inlineChannels][FftLengthBy2Plus1]float32
		accumLowRenderEnergy [inlineChannels][FftLengthBy2Plus1]bool
		accumNumPoints       [inlineChannels]int
		erle                 [inlineChannels][FftLengthBy2Plus1]float32
		erleOnsetCompensated [inlineChannels][FftLengthBy2Plus1]float32
		erleUnbounded        [inlineChannels][FftLengthBy2Plus1]float32
		erleDuringOnsets     [inlineChannels][FftLengthBy2Plus1]float32
		comingOnset          [inlineChannels][FftLengthBy2Plus1]bool
		holdCounters         [inlineChannels][FftLengthBy2Plus1]int
	}
}

// sbErleChannels is the per-channel state of a SubbandErleEstimator.
type sbErleChannels struct {
	accumY2              [][FftLengthBy2Plus1]float32
	accumE2              [][FftLengthBy2Plus1]float32
	accumLowRenderEnergy [][FftLengthBy2Plus1]bool
	accumNumPoints       []int

	erle                 [][FftLengthBy2Plus1]float32
	erleOnsetCompensated [][FftLengthBy2Plus1]float32
	erleUnbounded        [][FftLengthBy2Plus1]float32
	erleDuringOnsets     [][FftLengthBy2Plus1]float32
	comingOnset          [][FftLengthBy2Plus1]bool
	holdCounters         [][FftLengthBy2Plus1]int
}

// channels returns the per-channel state: views of s for up to
// inlineChannels channels.
func (s *SubbandErleEstimator) channels() sbErleChannels {
	if s.more != nil {
		return *s.more
	}
	n := s.numCaptureChannels
	return sbErleChannels{
		accumY2:              s.inline.accumY2[:n:n],
		accumE2:              s.inline.accumE2[:n:n],
		accumLowRenderEnergy: s.inline.accumLowRenderEnergy[:n:n],
		accumNumPoints:       s.inline.accumNumPoints[:n:n],
		erle:                 s.inline.erle[:n:n],
		erleOnsetCompensated: s.inline.erleOnsetCompensated[:n:n],
		erleUnbounded:        s.inline.erleUnbounded[:n:n],
		erleDuringOnsets:     s.inline.erleDuringOnsets[:n:n],
		comingOnset:          s.inline.comingOnset[:n:n],
		holdCounters:         s.inline.holdCounters[:n:n],
	}
}

// NewSubbandErleEstimator builds the estimator with field trials at their
// defaults (WebRTC-Aec3MinErleDuringOnsetsKillSwitch disabled).
func NewSubbandErleEstimator(config *EchoCanceller3Config, numCaptureChannels int) *SubbandErleEstimator {
	s := &SubbandErleEstimator{}
	s.init(config, numCaptureChannels)
	return s
}

// init sets up s in place.
func (s *SubbandErleEstimator) init(config *EchoCanceller3Config, numCaptureChannels int) {
	*s = SubbandErleEstimator{
		numCaptureChannels:     numCaptureChannels,
		useOnsetDetection:      config.Erle.OnsetDetection,
		minErle:                config.Erle.Min,
		useMinErleDuringOnsets: true,
	}
	if n := numCaptureChannels; n > inlineChannelLimit {
		s.more = &sbErleChannels{
			accumY2:              make([][FftLengthBy2Plus1]float32, n),
			accumE2:              make([][FftLengthBy2Plus1]float32, n),
			accumLowRenderEnergy: make([][FftLengthBy2Plus1]bool, n),
			accumNumPoints:       make([]int, n),
			erle:                 make([][FftLengthBy2Plus1]float32, n),
			erleOnsetCompensated: make([][FftLengthBy2Plus1]float32, n),
			erleUnbounded:        make([][FftLengthBy2Plus1]float32, n),
			erleDuringOnsets:     make([][FftLengthBy2Plus1]float32, n),
			comingOnset:          make([][FftLengthBy2Plus1]bool, n),
			holdCounters:         make([][FftLengthBy2Plus1]int, n),
		}
	}
	for k := range s.maxErle {
		if k < FftLengthBy2/2 {
			s.maxErle[k] = config.Erle.MaxL
		} else {
			s.maxErle[k] = config.Erle.MaxH
		}
	}
	s.Reset()
}

func (s *SubbandErleEstimator) Reset() {
	c := s.channels()
	for ch := range c.erle {
		for k := 0; k < FftLengthBy2Plus1; k++ {
			c.erle[ch][k] = s.minErle
			c.erleOnsetCompensated[ch][k] = s.minErle
			c.erleUnbounded[ch][k] = s.minErle
			c.erleDuringOnsets[ch][k] = s.minErle
			c.comingOnset[ch][k] = true
			c.holdCounters[ch][k] = 0
		}
	}
	s.resetAccumulatedSpectra(c)
}

// Erle returns the ERLE, onset compensated when requested and enabled.
func (s *SubbandErleEstimator) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	c := s.channels()
	if onsetCompensated && s.useOnsetDetection {
		return c.erleOnsetCompensated
	}
	return c.erle
}

func (s *SubbandErleEstimator) ErleUnbounded() [][FftLengthBy2Plus1]float32 {
	return s.channels().erleUnbounded
}
func (s *SubbandErleEstimator) ErleDuringOnsets() [][FftLengthBy2Plus1]float32 {
	c := s.channels()
	return c.erleDuringOnsets
}

// Update updates the ERLE estimate.
func (s *SubbandErleEstimator) Update(X2 *[FftLengthBy2Plus1]float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	c := s.channels()
	s.updateAccumulatedSpectra(c, X2, Y2, E2, convergedFilters)
	s.updateBands(c, convergedFilters)
	if s.useOnsetDetection {
		s.decreaseErlePerBandForLowRenderSignals(c)
	}
	for ch := range c.erle {
		e := &c.erle[ch]
		e[0] = e[1]
		e[FftLengthBy2] = e[FftLengthBy2-1]
		oc := &c.erleOnsetCompensated[ch]
		oc[0] = oc[1]
		oc[FftLengthBy2] = oc[FftLengthBy2-1]
		u := &c.erleUnbounded[ch]
		u[0] = u[1]
		u[FftLengthBy2] = u[FftLengthBy2-1]
	}
}

func stUpdateErleBand(erle *float32, newErle float32, lowRenderEnergy bool, minErle, maxErle float32) {
	alpha := float32(0.05)
	if newErle < *erle {
		if lowRenderEnergy {
			alpha = 0
		} else {
			alpha = 0.1
		}
	}
	*erle = stSafeClamp(*erle+float32(alpha*(newErle-*erle)), minErle, maxErle)
}

// updateBands and the loops below take each per-channel array through a
// local pointer: indexing the slices of s inside a bin loop reloads the
// slice and checks the channel on every access, because the stores of the
// loop may alias s.
func (s *SubbandErleEstimator) updateBands(c sbErleChannels, convergedFilters []bool) {
	for ch := range c.accumY2 {
		if !convergedFilters[ch] {
			continue
		}
		if c.accumNumPoints[ch] != stErlePointsToAccumulate {
			continue
		}
		accumY2, accumE2 := &c.accumY2[ch], &c.accumE2[ch]
		lowRender := &c.accumLowRenderEnergy[ch]
		var newErle [FftLengthBy2]float32
		var isErleUpdated [FftLengthBy2]bool
		for k := 1; k < FftLengthBy2; k++ {
			if accumE2[k] > 0 {
				newErle[k] = accumY2[k] / accumE2[k]
				isErleUpdated[k] = true
			}
		}
		minErle := s.minErle
		maxErle := &s.maxErle
		if s.useOnsetDetection {
			comingOnset, holdCounters := &c.comingOnset[ch], &c.holdCounters[ch]
			erleDuringOnsets := &c.erleDuringOnsets[ch]
			useMinErleDuringOnsets := s.useMinErleDuringOnsets
			for k := 1; k < FftLengthBy2; k++ {
				if isErleUpdated[k] && !lowRender[k] {
					if comingOnset[k] {
						comingOnset[k] = false
						if !useMinErleDuringOnsets {
							alpha := float32(0.15)
							if newErle[k] < erleDuringOnsets[k] {
								alpha = 0.3
							}
							erleDuringOnsets[k] = stSafeClamp(
								erleDuringOnsets[k]+float32(alpha*(newErle[k]-erleDuringOnsets[k])),
								minErle, maxErle[k])
						}
					}
					holdCounters[k] = stBlocksForOnsetDetection
				}
			}
		}
		erle, erleUnbounded := &c.erle[ch], &c.erleUnbounded[ch]
		erleOnsetCompensated := &c.erleOnsetCompensated[ch]
		useOnsetDetection := s.useOnsetDetection
		for k := 1; k < FftLengthBy2; k++ {
			if isErleUpdated[k] {
				lowRenderEnergy := lowRender[k]
				stUpdateErleBand(&erle[k], newErle[k], lowRenderEnergy, minErle, maxErle[k])
				if useOnsetDetection {
					stUpdateErleBand(&erleOnsetCompensated[k], newErle[k], lowRenderEnergy, minErle, maxErle[k])
				}
				const unboundedErleMax float32 = 100000.0
				stUpdateErleBand(&erleUnbounded[k], newErle[k], lowRenderEnergy, minErle, unboundedErleMax)
			}
		}
	}
}

func (s *SubbandErleEstimator) decreaseErlePerBandForLowRenderSignals(c sbErleChannels) {
	for ch := range c.accumY2 {
		holdCounters, comingOnset := &c.holdCounters[ch], &c.comingOnset[ch]
		erleOnsetCompensated, erleDuringOnsets := &c.erleOnsetCompensated[ch], &c.erleDuringOnsets[ch]
		for k := 1; k < FftLengthBy2; k++ {
			c := holdCounters[k] - 1
			holdCounters[k] = c
			if c <= stBlocksForOnsetDetection-stBlocksToHoldErle {
				if erleOnsetCompensated[k] > erleDuringOnsets[k] {
					erleOnsetCompensated[k] = stdMax(erleDuringOnsets[k], 0.97*erleOnsetCompensated[k])
				}
				if c <= 0 {
					comingOnset[k] = true
					holdCounters[k] = 0
				}
			}
		}
	}
}

func (s *SubbandErleEstimator) resetAccumulatedSpectra(c sbErleChannels) {
	for ch := range c.erleDuringOnsets {
		c.accumY2[ch] = [FftLengthBy2Plus1]float32{}
		c.accumE2[ch] = [FftLengthBy2Plus1]float32{}
		c.accumNumPoints[ch] = 0
		c.accumLowRenderEnergy[ch] = [FftLengthBy2Plus1]bool{}
	}
}

func (s *SubbandErleEstimator) updateAccumulatedSpectra(c sbErleChannels, X2 *[FftLengthBy2Plus1]float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	for ch := range Y2 {
		if !convergedFilters[ch] {
			continue
		}
		if c.accumNumPoints[ch] == stErlePointsToAccumulate {
			c.accumNumPoints[ch] = 0
			c.accumY2[ch] = [FftLengthBy2Plus1]float32{}
			c.accumE2[ch] = [FftLengthBy2Plus1]float32{}
			c.accumLowRenderEnergy[ch] = [FftLengthBy2Plus1]bool{}
		}
		kernAdd(Y2[ch][:], c.accumY2[ch][:], c.accumY2[ch][:])
		kernAdd(E2[ch][:], c.accumE2[ch][:], c.accumE2[ch][:])
		lowRender := &c.accumLowRenderEnergy[ch]
		for k := range X2 {
			lowRender[k] = lowRender[k] || X2[k] < stX2BandEnergyThreshold
		}
		c.accumNumPoints[ch]++
	}
}
