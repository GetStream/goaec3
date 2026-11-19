package aec3

const stBlocksForOnsetDetection = stBlocksToHoldErle + 150

// SubbandErleEstimator mirrors subband_erle_estimator.h: estimates the ERLE
// per frequency band.
type SubbandErleEstimator struct {
	useOnsetDetection      bool
	minErle                float32
	maxErle                [FftLengthBy2Plus1]float32
	useMinErleDuringOnsets bool

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

	// The per-channel state above for up to inlineChannels channels.
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

// NewSubbandErleEstimator builds the estimator with field trials at their
// defaults (WebRTC-Aec3MinErleDuringOnsetsKillSwitch disabled).
func NewSubbandErleEstimator(config *EchoCanceller3Config, numCaptureChannels int) *SubbandErleEstimator {
	s := &SubbandErleEstimator{}
	s.init(config, numCaptureChannels)
	return s
}

// init sets up s in place; s must not be copied afterwards (see
// channelSlice).
func (s *SubbandErleEstimator) init(config *EchoCanceller3Config, numCaptureChannels int) {
	*s = SubbandErleEstimator{
		useOnsetDetection:      config.Erle.OnsetDetection,
		minErle:                config.Erle.Min,
		useMinErleDuringOnsets: true,
	}
	n := numCaptureChannels
	s.accumY2 = channelSlice(s.inline.accumY2[:], n)
	s.accumE2 = channelSlice(s.inline.accumE2[:], n)
	s.accumLowRenderEnergy = channelSlice(s.inline.accumLowRenderEnergy[:], n)
	s.accumNumPoints = channelSlice(s.inline.accumNumPoints[:], n)
	s.erle = channelSlice(s.inline.erle[:], n)
	s.erleOnsetCompensated = channelSlice(s.inline.erleOnsetCompensated[:], n)
	s.erleUnbounded = channelSlice(s.inline.erleUnbounded[:], n)
	s.erleDuringOnsets = channelSlice(s.inline.erleDuringOnsets[:], n)
	s.comingOnset = channelSlice(s.inline.comingOnset[:], n)
	s.holdCounters = channelSlice(s.inline.holdCounters[:], n)
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
	for ch := range s.erle {
		for k := 0; k < FftLengthBy2Plus1; k++ {
			s.erle[ch][k] = s.minErle
			s.erleOnsetCompensated[ch][k] = s.minErle
			s.erleUnbounded[ch][k] = s.minErle
			s.erleDuringOnsets[ch][k] = s.minErle
			s.comingOnset[ch][k] = true
			s.holdCounters[ch][k] = 0
		}
	}
	s.resetAccumulatedSpectra()
}

// Erle returns the ERLE, onset compensated when requested and enabled.
func (s *SubbandErleEstimator) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	if onsetCompensated && s.useOnsetDetection {
		return s.erleOnsetCompensated
	}
	return s.erle
}

func (s *SubbandErleEstimator) ErleUnbounded() [][FftLengthBy2Plus1]float32 { return s.erleUnbounded }
func (s *SubbandErleEstimator) ErleDuringOnsets() [][FftLengthBy2Plus1]float32 {
	return s.erleDuringOnsets
}

// Update updates the ERLE estimate.
func (s *SubbandErleEstimator) Update(X2 *[FftLengthBy2Plus1]float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	s.updateAccumulatedSpectra(X2, Y2, E2, convergedFilters)
	s.updateBands(convergedFilters)
	if s.useOnsetDetection {
		s.decreaseErlePerBandForLowRenderSignals()
	}
	for ch := range s.erle {
		e := &s.erle[ch]
		e[0] = e[1]
		e[FftLengthBy2] = e[FftLengthBy2-1]
		oc := &s.erleOnsetCompensated[ch]
		oc[0] = oc[1]
		oc[FftLengthBy2] = oc[FftLengthBy2-1]
		u := &s.erleUnbounded[ch]
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
func (s *SubbandErleEstimator) updateBands(convergedFilters []bool) {
	for ch := range s.accumY2 {
		if !convergedFilters[ch] {
			continue
		}
		if s.accumNumPoints[ch] != stErlePointsToAccumulate {
			continue
		}
		accumY2, accumE2 := &s.accumY2[ch], &s.accumE2[ch]
		lowRender := &s.accumLowRenderEnergy[ch]
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
			comingOnset, holdCounters := &s.comingOnset[ch], &s.holdCounters[ch]
			erleDuringOnsets := &s.erleDuringOnsets[ch]
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
		erle, erleUnbounded := &s.erle[ch], &s.erleUnbounded[ch]
		erleOnsetCompensated := &s.erleOnsetCompensated[ch]
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

func (s *SubbandErleEstimator) decreaseErlePerBandForLowRenderSignals() {
	for ch := range s.accumY2 {
		holdCounters, comingOnset := &s.holdCounters[ch], &s.comingOnset[ch]
		erleOnsetCompensated, erleDuringOnsets := &s.erleOnsetCompensated[ch], &s.erleDuringOnsets[ch]
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

func (s *SubbandErleEstimator) resetAccumulatedSpectra() {
	for ch := range s.erleDuringOnsets {
		s.accumY2[ch] = [FftLengthBy2Plus1]float32{}
		s.accumE2[ch] = [FftLengthBy2Plus1]float32{}
		s.accumNumPoints[ch] = 0
		s.accumLowRenderEnergy[ch] = [FftLengthBy2Plus1]bool{}
	}
}

func (s *SubbandErleEstimator) updateAccumulatedSpectra(X2 *[FftLengthBy2Plus1]float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	for ch := range Y2 {
		if !convergedFilters[ch] {
			continue
		}
		if s.accumNumPoints[ch] == stErlePointsToAccumulate {
			s.accumNumPoints[ch] = 0
			s.accumY2[ch] = [FftLengthBy2Plus1]float32{}
			s.accumE2[ch] = [FftLengthBy2Plus1]float32{}
			s.accumLowRenderEnergy[ch] = [FftLengthBy2Plus1]bool{}
		}
		kernAdd(Y2[ch][:], s.accumY2[ch][:], s.accumY2[ch][:])
		kernAdd(E2[ch][:], s.accumE2[ch][:], s.accumE2[ch][:])
		lowRender := &s.accumLowRenderEnergy[ch]
		for k := range X2 {
			lowRender[k] = lowRender[k] || X2[k] < stX2BandEnergyThreshold
		}
		s.accumNumPoints[ch]++
	}
}
