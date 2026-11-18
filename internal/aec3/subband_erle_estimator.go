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
}

// NewSubbandErleEstimator builds the estimator with field trials at their
// defaults (WebRTC-Aec3MinErleDuringOnsetsKillSwitch disabled).
func NewSubbandErleEstimator(config *EchoCanceller3Config, numCaptureChannels int) *SubbandErleEstimator {
	s := &SubbandErleEstimator{
		useOnsetDetection:      config.Erle.OnsetDetection,
		minErle:                config.Erle.Min,
		useMinErleDuringOnsets: true,
		accumY2:                make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		accumE2:                make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		accumLowRenderEnergy:   make([][FftLengthBy2Plus1]bool, numCaptureChannels),
		accumNumPoints:         make([]int, numCaptureChannels),
		erle:                   make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		erleOnsetCompensated:   make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		erleUnbounded:          make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		erleDuringOnsets:       make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		comingOnset:            make([][FftLengthBy2Plus1]bool, numCaptureChannels),
		holdCounters:           make([][FftLengthBy2Plus1]int, numCaptureChannels),
	}
	for k := range s.maxErle {
		if k < FftLengthBy2/2 {
			s.maxErle[k] = config.Erle.MaxL
		} else {
			s.maxErle[k] = config.Erle.MaxH
		}
	}
	s.Reset()
	return s
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

func (s *SubbandErleEstimator) updateBands(convergedFilters []bool) {
	for ch := range s.accumY2 {
		if !convergedFilters[ch] {
			continue
		}
		if s.accumNumPoints[ch] != stErlePointsToAccumulate {
			continue
		}
		var newErle [FftLengthBy2]float32
		var isErleUpdated [FftLengthBy2]bool
		for k := 1; k < FftLengthBy2; k++ {
			if s.accumE2[ch][k] > 0 {
				newErle[k] = s.accumY2[ch][k] / s.accumE2[ch][k]
				isErleUpdated[k] = true
			}
		}
		if s.useOnsetDetection {
			for k := 1; k < FftLengthBy2; k++ {
				if isErleUpdated[k] && !s.accumLowRenderEnergy[ch][k] {
					if s.comingOnset[ch][k] {
						s.comingOnset[ch][k] = false
						if !s.useMinErleDuringOnsets {
							alpha := float32(0.15)
							if newErle[k] < s.erleDuringOnsets[ch][k] {
								alpha = 0.3
							}
							s.erleDuringOnsets[ch][k] = stSafeClamp(
								s.erleDuringOnsets[ch][k]+float32(alpha*(newErle[k]-s.erleDuringOnsets[ch][k])),
								s.minErle, s.maxErle[k])
						}
					}
					s.holdCounters[ch][k] = stBlocksForOnsetDetection
				}
			}
		}
		for k := 1; k < FftLengthBy2; k++ {
			if isErleUpdated[k] {
				lowRenderEnergy := s.accumLowRenderEnergy[ch][k]
				stUpdateErleBand(&s.erle[ch][k], newErle[k], lowRenderEnergy, s.minErle, s.maxErle[k])
				if s.useOnsetDetection {
					stUpdateErleBand(&s.erleOnsetCompensated[ch][k], newErle[k], lowRenderEnergy, s.minErle, s.maxErle[k])
				}
				const unboundedErleMax float32 = 100000.0
				stUpdateErleBand(&s.erleUnbounded[ch][k], newErle[k], lowRenderEnergy, s.minErle, unboundedErleMax)
			}
		}
	}
}

func (s *SubbandErleEstimator) decreaseErlePerBandForLowRenderSignals() {
	for ch := range s.accumY2 {
		for k := 1; k < FftLengthBy2; k++ {
			s.holdCounters[ch][k]--
			if s.holdCounters[ch][k] <= stBlocksForOnsetDetection-stBlocksToHoldErle {
				if s.erleOnsetCompensated[ch][k] > s.erleDuringOnsets[ch][k] {
					s.erleOnsetCompensated[ch][k] = stdMax(s.erleDuringOnsets[ch][k], 0.97*s.erleOnsetCompensated[ch][k])
				}
				if s.holdCounters[ch][k] <= 0 {
					s.comingOnset[ch][k] = true
					s.holdCounters[ch][k] = 0
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
		for k := range X2 {
			s.accumLowRenderEnergy[ch][k] = s.accumLowRenderEnergy[ch][k] || X2[k] < stX2BandEnergyThreshold
		}
		s.accumNumPoints[ch]++
	}
}
