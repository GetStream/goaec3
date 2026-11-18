package aec3

// Port of subtractor.{h,cc}.

import "math"

// subPredictionError computes the prediction error e = y - s and the scaled
// filter output s from the frequency-domain filter output S.
func subPredictionError(fft *Aec3Fft, S *FftData, y []float32, tmp *[FftLength]float32, e, s *[BlockSize]float32) {
	fft.Ifft(S, tmp)
	const scale float32 = 1.0 / FftLengthBy2
	// e = y - float32(tmp*scale) and s = scale*tmp: the scaled output is
	// rounded once and shared.
	var scaled [BlockSize]float32
	if s == nil {
		s = &scaled
	}
	kernScale(tmp[FftLengthBy2:], scale, s[:])
	kernSub(y[:BlockSize], s[:], e[:])
}

// subScaleFilterOutput scales the filter output s by factor and recomputes
// the error e = y - s.
func subScaleFilterOutput(y []float32, factor float32, e, s []float32) {
	for k := range y {
		s[k] = float32(s[k] * factor) // rounded before the subtraction (no FMA)
		e[k] = y[k] - s[k]
	}
}

// subFilterMisadjustmentEstimator is Subtractor::FilterMisadjustmentEstimator.
type subFilterMisadjustmentEstimator struct {
	nBlocksAcum      int
	e2Acum           float32
	y2Acum           float32
	invMisadjustment float32
	overhang         int
}

const subMisadjustmentNBlocks = 4

// Update updates the misadjustment estimator.
func (m *subFilterMisadjustmentEstimator) Update(output *SubtractorOutput) {
	m.e2Acum += output.E2RefinedSum
	m.y2Acum += output.Y2
	m.nBlocksAcum++
	if m.nBlocksAcum == subMisadjustmentNBlocks {
		const y2Threshold float32 = subMisadjustmentNBlocks * 200 * 200 * BlockSize
		const e2Threshold float32 = subMisadjustmentNBlocks * 7500 * 7500 * BlockSize
		if m.y2Acum > y2Threshold {
			update := m.e2Acum / m.y2Acum
			if m.e2Acum > e2Threshold {
				m.overhang = 4
			} else {
				m.overhang = max(m.overhang-1, 0)
			}
			if update < m.invMisadjustment || m.overhang > 0 {
				m.invMisadjustment += float32(0.1 * (update - m.invMisadjustment))
			}
		}
		m.e2Acum = 0
		m.y2Acum = 0
		m.nBlocksAcum = 0
	}
}

// GetMisadjustment returns the recommended filter scale (half of the
// estimated mismatch).
func (m *subFilterMisadjustmentEstimator) GetMisadjustment() float32 {
	return 2 / float32(math.Sqrt(float64(m.invMisadjustment)))
}

// IsAdjustmentNeeded reports whether the prediction error energy is
// significantly larger than the microphone signal energy.
func (m *subFilterMisadjustmentEstimator) IsAdjustmentNeeded() bool {
	return m.invMisadjustment > 10
}

func (m *subFilterMisadjustmentEstimator) Reset() {
	m.e2Acum = 0
	m.y2Acum = 0
	m.nBlocksAcum = 0
	m.invMisadjustment = 0
	m.overhang = 0
}

// Subtractor performs the linear echo cancellation with a refined and a
// coarse adaptive filter per capture channel.
type Subtractor struct {
	fft                           Aec3Fft
	optimization                  Aec3Optimization
	config                        EchoCanceller3Config
	numCaptureChannels            int
	useCoarseFilterResetHangover  bool
	refinedFilters                []*AdaptiveFirFilter
	coarseFilter                  []*AdaptiveFirFilter
	refinedGains                  []*RefinedFilterUpdateGain
	coarseGains                   []*CoarseFilterUpdateGain
	filterMisadjustmentEstimators []subFilterMisadjustmentEstimator
	poorCoarseFilterCounters      []int
	coarseFilterResetHangover     []int
	refinedFrequencyResponses     [][][FftLengthBy2Plus1]float32
	refinedImpulseResponses       [][]float32

	// Per-block scratch.
	x2Refined     [FftLengthBy2Plus1]float32
	x2CoarseData  [FftLengthBy2Plus1]float32
	eCoarseFft    FftData
	s             FftData
	erl           [FftLengthBy2Plus1]float32
	predictionTmp [FftLength]float32
}

// NewSubtractor creates a Subtractor. Field trials are fixed at their
// defaults, so the coarse filter reset hangover is in use.
func NewSubtractor(config *EchoCanceller3Config, numRenderChannels, numCaptureChannels int, optimization Aec3Optimization) *Subtractor {
	s := &Subtractor{
		optimization:                  optimization,
		config:                        *config,
		numCaptureChannels:            numCaptureChannels,
		useCoarseFilterResetHangover:  true,
		refinedFilters:                make([]*AdaptiveFirFilter, numCaptureChannels),
		coarseFilter:                  make([]*AdaptiveFirFilter, numCaptureChannels),
		refinedGains:                  make([]*RefinedFilterUpdateGain, numCaptureChannels),
		coarseGains:                   make([]*CoarseFilterUpdateGain, numCaptureChannels),
		filterMisadjustmentEstimators: make([]subFilterMisadjustmentEstimator, numCaptureChannels),
		poorCoarseFilterCounters:      make([]int, numCaptureChannels),
		coarseFilterResetHangover:     make([]int, numCaptureChannels),
		refinedFrequencyResponses:     make([][][FftLengthBy2Plus1]float32, numCaptureChannels),
		refinedImpulseResponses:       make([][]float32, numCaptureChannels),
	}
	c := &s.config
	maxRefinedBlocks := max(c.Filter.RefinedInitial.LengthBlocks, c.Filter.Refined.LengthBlocks)
	for ch := 0; ch < numCaptureChannels; ch++ {
		s.refinedFrequencyResponses[ch] = make([][FftLengthBy2Plus1]float32, maxRefinedBlocks)
		s.refinedImpulseResponses[ch] = make([]float32, GetTimeDomainLength(maxRefinedBlocks))

		s.refinedFilters[ch] = NewAdaptiveFirFilter(c.Filter.Refined.LengthBlocks, c.Filter.RefinedInitial.LengthBlocks,
			c.Filter.ConfigChangeDurationBlocks, numRenderChannels, optimization)
		s.coarseFilter[ch] = NewAdaptiveFirFilter(c.Filter.Coarse.LengthBlocks, c.Filter.CoarseInitial.LengthBlocks,
			c.Filter.ConfigChangeDurationBlocks, numRenderChannels, optimization)
		s.refinedGains[ch] = NewRefinedFilterUpdateGain(c.Filter.RefinedInitial, c.Filter.ConfigChangeDurationBlocks)
		s.coarseGains[ch] = NewCoarseFilterUpdateGain(c.Filter.CoarseInitial, c.Filter.ConfigChangeDurationBlocks)
	}
	return s
}

// HandleEchoPathChange handles echo path changes.
func (s *Subtractor) HandleEchoPathChange(echoPathVariability EchoPathVariability) {
	c := &s.config
	if echoPathVariability.DelayChange != DelayAdjustmentNone {
		for ch := 0; ch < s.numCaptureChannels; ch++ {
			s.refinedFilters[ch].HandleEchoPathChange()
			s.coarseFilter[ch].HandleEchoPathChange()
			s.refinedGains[ch].HandleEchoPathChange(echoPathVariability)
			s.coarseGains[ch].HandleEchoPathChange()
			s.refinedGains[ch].SetConfig(c.Filter.RefinedInitial, true)
			s.coarseGains[ch].SetConfig(c.Filter.CoarseInitial, true)
			s.refinedFilters[ch].SetSizePartitions(c.Filter.RefinedInitial.LengthBlocks, true)
			s.coarseFilter[ch].SetSizePartitions(c.Filter.CoarseInitial.LengthBlocks, true)
		}
	}

	if echoPathVariability.GainChange {
		for ch := 0; ch < s.numCaptureChannels; ch++ {
			s.refinedGains[ch].HandleEchoPathChange(echoPathVariability)
		}
	}
}

// ExitInitialState switches the filters and gains to the normal configs.
func (s *Subtractor) ExitInitialState() {
	c := &s.config
	for ch := 0; ch < s.numCaptureChannels; ch++ {
		s.refinedGains[ch].SetConfig(c.Filter.Refined, false)
		s.coarseGains[ch].SetConfig(c.Filter.Coarse, false)
		s.refinedFilters[ch].SetSizePartitions(c.Filter.Refined.LengthBlocks, false)
		s.coarseFilter[ch].SetSizePartitions(c.Filter.Coarse.LengthBlocks, false)
	}
}

// FilterFrequencyResponses returns the block-wise frequency responses of the
// refined adaptive filters ([channel][partition]).
func (s *Subtractor) FilterFrequencyResponses() [][][FftLengthBy2Plus1]float32 {
	return s.refinedFrequencyResponses
}

// FilterImpulseResponses returns the impulse response estimates of the
// refined adaptive filters ([channel][sample]).
func (s *Subtractor) FilterImpulseResponses() [][]float32 {
	return s.refinedImpulseResponses
}

// Process performs the echo subtraction. outputs has one entry per capture
// channel.
func (s *Subtractor) Process(renderBuffer *RenderBuffer, capture *Block, renderSignalAnalyzer *RenderSignalAnalyzer,
	aecState *AecState, outputs []SubtractorOutput) {
	// Compute the render powers.
	refinedSize := s.refinedFilters[0].SizePartitions()
	coarseSize := s.coarseFilter[0].SizePartitions()
	sameFilterSizes := refinedSize == coarseSize
	X2Refined := &s.x2Refined
	X2Coarse := &s.x2CoarseData
	if sameFilterSizes {
		X2Coarse = X2Refined
		renderBuffer.SpectralSum(refinedSize, X2Refined)
	} else if refinedSize > coarseSize {
		renderBuffer.SpectralSums(coarseSize, refinedSize, X2Coarse, X2Refined)
	} else {
		renderBuffer.SpectralSums(refinedSize, coarseSize, X2Refined, X2Coarse)
	}

	saturatedCapture := aecState.SaturatedCapture()

	// Process all capture channels.
	for ch := 0; ch < s.numCaptureChannels; ch++ {
		output := &outputs[ch]
		y := capture.View(0, ch)
		ERefined := &output.ERefinedFft
		ECoarse := &s.eCoarseFft
		eRefined := &output.ERefined
		eCoarse := &output.ECoarse

		S := &s.s
		G := S

		// Form the outputs of the refined and coarse filters.
		s.refinedFilters[ch].Filter(renderBuffer, S)
		subPredictionError(&s.fft, S, y, &s.predictionTmp, eRefined, &output.SRefined)

		s.coarseFilter[ch].Filter(renderBuffer, S)
		subPredictionError(&s.fft, S, y, &s.predictionTmp, eCoarse, &output.SCoarse)

		// Compute the signal powers in the subtractor output.
		output.ComputeMetrics(y)

		// Adjust the filter if needed.
		refinedFiltersAdjusted := false
		est := &s.filterMisadjustmentEstimators[ch]
		est.Update(output)
		if est.IsAdjustmentNeeded() {
			scale := est.GetMisadjustment()
			s.refinedFilters[ch].ScaleFilter(scale)
			h := s.refinedImpulseResponses[ch]
			for k := range h {
				h[k] *= scale
			}
			subScaleFilterOutput(y, scale, eRefined[:], output.SRefined[:])
			est.Reset()
			refinedFiltersAdjusted = true
		}

		// Compute the FFTs of the refined and coarse filter outputs.
		s.fft.ZeroPaddedFft(eRefined[:], WindowHanning, ERefined)
		s.fft.ZeroPaddedFft(eCoarse[:], WindowHanning, ECoarse)

		// Compute spectra for future use.
		ECoarse.Spectrum(s.optimization, output.E2Coarse[:])
		ERefined.Spectrum(s.optimization, output.E2Refined[:])

		// Update the refined filter.
		if !refinedFiltersAdjusted {
			// Do not allow the performance of the coarse filter to affect the
			// adaptation speed of the refined filter just after the coarse
			// filter has been reset.
			disallowLeakageDiverged := s.coarseFilterResetHangover[ch] > 0 && s.useCoarseFilterResetHangover

			ComputeErl(s.optimization, s.refinedFrequencyResponses[ch], s.erl[:])
			s.refinedGains[ch].Compute(X2Refined, renderSignalAnalyzer, output, s.erl[:],
				s.refinedFilters[ch].SizePartitions(), saturatedCapture, disallowLeakageDiverged, G)
		} else {
			G.Clear()
		}
		s.refinedFilters[ch].AdaptWithImpulseResponse(renderBuffer, G, &s.refinedImpulseResponses[ch])
		s.refinedFilters[ch].ComputeFrequencyResponse(&s.refinedFrequencyResponses[ch])

		// Update the coarse filter.
		if output.E2RefinedSum < output.E2CoarseSum {
			s.poorCoarseFilterCounters[ch]++
		} else {
			s.poorCoarseFilterCounters[ch] = 0
		}
		if s.poorCoarseFilterCounters[ch] < 5 {
			s.coarseGains[ch].Compute(X2Coarse, renderSignalAnalyzer, ECoarse,
				s.coarseFilter[ch].SizePartitions(), saturatedCapture, G)
			s.coarseFilterResetHangover[ch] = max(s.coarseFilterResetHangover[ch]-1, 0)
		} else {
			s.poorCoarseFilterCounters[ch] = 0
			s.coarseFilter[ch].SetFilter(s.refinedFilters[ch].SizePartitions(), s.refinedFilters[ch].GetFilter())
			s.coarseGains[ch].Compute(X2Coarse, renderSignalAnalyzer, ERefined,
				s.coarseFilter[ch].SizePartitions(), saturatedCapture, G)
			s.coarseFilterResetHangover[ch] = s.config.Filter.CoarseResetHangoverBlocks
		}

		s.coarseFilter[ch].Adapt(renderBuffer, G)

		kernClamp(eRefined[:], -32768, 32767)
	}
}
