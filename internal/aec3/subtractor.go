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

// subPredictionErrorPair is subPredictionError for the refined (S0) and the
// coarse (S1) filter output of one capture signal y.
func subPredictionErrorPair(fft *aec3FftPair, S0, S1 *FftData, y []float32, tmp0, tmp1 *[FftLength]float32,
	e0, s0, e1, s1 *[BlockSize]float32) {
	fft.Ifft(S0, S1, tmp0, tmp1)
	const scale float32 = 1.0 / FftLengthBy2
	kernScale(tmp0[FftLengthBy2:], scale, s0[:])
	kernSub(y[:BlockSize], s0[:], e0[:])
	kernScale(tmp1[FftLengthBy2:], scale, s1[:])
	kernSub(y[:BlockSize], s1[:], e1[:])
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
//
// The filters, gains and counters of up to inlineChannels capture channels
// are part of the Subtractor. The filter coefficients of all channels share
// one array, as do the frequency responses and the impulse responses.
type Subtractor struct {
	more                      *subMore                                     // with more than inlineChannels channels
	refinedFrequencyResponses [inlineChannels][][FftLengthBy2Plus1]float32 // [channel][partition]
	refinedImpulseResponses   [inlineChannels][]float32                    // [channel][sample]
	channels                  [inlineChannels]subChannel

	fft                          aec3FftPair
	optimization                 Aec3Optimization
	config                       EchoCanceller3Config
	numCaptureChannels           int
	useCoarseFilterResetHangover bool

	// Per-block scratch.
	x2Refined      [FftLengthBy2Plus1]float32
	x2CoarseData   [FftLengthBy2Plus1]float32
	eCoarseFft     FftData
	s              FftData
	sCoarse        FftData
	erl            [FftLengthBy2Plus1]float32
	predictionTmp  [FftLength]float32
	predictionTmp2 [FftLength]float32
}

// subChannel is the state of one capture channel of a Subtractor.
type subChannel struct {
	refinedFilter                AdaptiveFirFilter
	coarseFilter                 AdaptiveFirFilter
	refinedGain                  RefinedFilterUpdateGain
	coarseGain                   CoarseFilterUpdateGain
	filterMisadjustmentEstimator subFilterMisadjustmentEstimator
	poorCoarseFilterCounter      int
	coarseFilterResetHangover    int
}

// subMore is the per-channel state of a Subtractor with more than
// inlineChannels capture channels.
type subMore struct {
	refinedFrequencyResponses [][][FftLengthBy2Plus1]float32
	refinedImpulseResponses   [][]float32
	channels                  []subChannel
}

// NewSubtractor creates a Subtractor. Field trials are fixed at their
// defaults, so the coarse filter reset hangover is in use.
func NewSubtractor(config *EchoCanceller3Config, numRenderChannels, numCaptureChannels int, optimization Aec3Optimization) *Subtractor {
	s := &Subtractor{}
	s.init(config, numRenderChannels, numCaptureChannels, optimization)
	return s
}

// init sets up s in place.
func (s *Subtractor) init(config *EchoCanceller3Config, numRenderChannels, numCaptureChannels int, optimization Aec3Optimization) {
	*s = Subtractor{}
	s.optimization = optimization
	s.config = *config
	s.numCaptureChannels = numCaptureChannels
	s.useCoarseFilterResetHangover = true
	if numCaptureChannels > inlineChannelLimit {
		s.more = &subMore{
			refinedFrequencyResponses: make([][][FftLengthBy2Plus1]float32, numCaptureChannels),
			refinedImpulseResponses:   make([][]float32, numCaptureChannels),
			channels:                  make([]subChannel, numCaptureChannels),
		}
	}
	c := &s.config
	maxRefinedBlocks := max(c.Filter.RefinedInitial.LengthBlocks, c.Filter.Refined.LengthBlocks)
	refinedCoefficients := c.Filter.Refined.LengthBlocks * numRenderChannels
	coarseCoefficients := c.Filter.Coarse.LengthBlocks * numRenderChannels
	impulseResponseLength := GetTimeDomainLength(maxRefinedBlocks)
	coefficients := make([]FftData, numCaptureChannels*(refinedCoefficients+coarseCoefficients))
	frequencyResponses := make([][FftLengthBy2Plus1]float32, numCaptureChannels*maxRefinedBlocks)
	impulseResponses := make([]float32, numCaptureChannels*impulseResponseLength)

	channels := s.channelStates()
	H2 := s.FilterFrequencyResponses()
	h := s.FilterImpulseResponses()
	for ch := range channels {
		H2[ch] = frequencyResponses[:maxRefinedBlocks:maxRefinedBlocks]
		frequencyResponses = frequencyResponses[maxRefinedBlocks:]
		h[ch] = impulseResponses[:impulseResponseLength:impulseResponseLength]
		impulseResponses = impulseResponses[impulseResponseLength:]

		sc := &channels[ch]
		sc.refinedFilter.init(c.Filter.Refined.LengthBlocks, c.Filter.RefinedInitial.LengthBlocks,
			c.Filter.ConfigChangeDurationBlocks, numRenderChannels, optimization, coefficients[:refinedCoefficients])
		coefficients = coefficients[refinedCoefficients:]
		sc.coarseFilter.init(c.Filter.Coarse.LengthBlocks, c.Filter.CoarseInitial.LengthBlocks,
			c.Filter.ConfigChangeDurationBlocks, numRenderChannels, optimization, coefficients[:coarseCoefficients])
		coefficients = coefficients[coarseCoefficients:]
		sc.refinedGain.init(c.Filter.RefinedInitial, c.Filter.ConfigChangeDurationBlocks)
		sc.coarseGain.init(c.Filter.CoarseInitial, c.Filter.ConfigChangeDurationBlocks)
	}
}

// channelStates returns the state of every capture channel.
func (s *Subtractor) channelStates() []subChannel {
	if s.more != nil {
		return s.more.channels
	}
	return s.channels[:s.numCaptureChannels:s.numCaptureChannels]
}

// HandleEchoPathChange handles echo path changes.
func (s *Subtractor) HandleEchoPathChange(echoPathVariability EchoPathVariability) {
	c := &s.config
	channels := s.channelStates()
	if echoPathVariability.DelayChange != DelayAdjustmentNone {
		for ch := range channels {
			sc := &channels[ch]
			sc.refinedFilter.HandleEchoPathChange()
			sc.coarseFilter.HandleEchoPathChange()
			sc.refinedGain.HandleEchoPathChange(echoPathVariability)
			sc.coarseGain.HandleEchoPathChange()
			sc.refinedGain.SetConfig(c.Filter.RefinedInitial, true)
			sc.coarseGain.SetConfig(c.Filter.CoarseInitial, true)
			sc.refinedFilter.SetSizePartitions(c.Filter.RefinedInitial.LengthBlocks, true)
			sc.coarseFilter.SetSizePartitions(c.Filter.CoarseInitial.LengthBlocks, true)
		}
	}

	if echoPathVariability.GainChange {
		for ch := range channels {
			channels[ch].refinedGain.HandleEchoPathChange(echoPathVariability)
		}
	}
}

// ExitInitialState switches the filters and gains to the normal configs.
func (s *Subtractor) ExitInitialState() {
	c := &s.config
	channels := s.channelStates()
	for ch := range channels {
		sc := &channels[ch]
		sc.refinedGain.SetConfig(c.Filter.Refined, false)
		sc.coarseGain.SetConfig(c.Filter.Coarse, false)
		sc.refinedFilter.SetSizePartitions(c.Filter.Refined.LengthBlocks, false)
		sc.coarseFilter.SetSizePartitions(c.Filter.Coarse.LengthBlocks, false)
	}
}

// FilterFrequencyResponses returns the block-wise frequency responses of the
// refined adaptive filters ([channel][partition]).
func (s *Subtractor) FilterFrequencyResponses() [][][FftLengthBy2Plus1]float32 {
	if s.more != nil {
		return s.more.refinedFrequencyResponses
	}
	return s.refinedFrequencyResponses[:s.numCaptureChannels:s.numCaptureChannels]
}

// FilterImpulseResponses returns the impulse response estimates of the
// refined adaptive filters ([channel][sample]).
func (s *Subtractor) FilterImpulseResponses() [][]float32 {
	if s.more != nil {
		return s.more.refinedImpulseResponses
	}
	return s.refinedImpulseResponses[:s.numCaptureChannels:s.numCaptureChannels]
}

// Process performs the echo subtraction. outputs has one entry per capture
// channel.
func (s *Subtractor) Process(renderBuffer *RenderBuffer, capture *Block, renderSignalAnalyzer *RenderSignalAnalyzer,
	aecState *AecState, outputs []SubtractorOutput) {
	// Compute the render powers.
	channels := s.channelStates()
	frequencyResponses := s.FilterFrequencyResponses()
	impulseResponses := s.FilterImpulseResponses()
	refinedSize := channels[0].refinedFilter.SizePartitions()
	coarseSize := channels[0].coarseFilter.SizePartitions()
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
	for ch := range channels {
		sc := &channels[ch]
		output := &outputs[ch]
		y := capture.View(0, ch)
		ERefined := &output.ERefinedFft
		ECoarse := &s.eCoarseFft
		eRefined := &output.ERefined
		eCoarse := &output.ECoarse

		S := &s.s
		G := S

		// Form the outputs of the refined and coarse filters; the two
		// inverse transforms run as a pair.
		sc.refinedFilter.Filter(renderBuffer, S)
		sc.coarseFilter.Filter(renderBuffer, &s.sCoarse)
		subPredictionErrorPair(&s.fft, S, &s.sCoarse, y, &s.predictionTmp, &s.predictionTmp2,
			eRefined, &output.SRefined, eCoarse, &output.SCoarse)

		// Compute the signal powers in the subtractor output.
		output.ComputeMetrics(y)

		// Adjust the filter if needed.
		refinedFiltersAdjusted := false
		est := &sc.filterMisadjustmentEstimator
		est.Update(output)
		if est.IsAdjustmentNeeded() {
			scale := est.GetMisadjustment()
			sc.refinedFilter.ScaleFilter(scale)
			h := impulseResponses[ch]
			for k := range h {
				h[k] *= scale
			}
			subScaleFilterOutput(y, scale, eRefined[:], output.SRefined[:])
			est.Reset()
			refinedFiltersAdjusted = true
		}

		// Compute the FFTs of the refined and coarse filter outputs.
		s.fft.ZeroPaddedFft(eRefined[:], eCoarse[:], ERefined, ECoarse)

		// Compute spectra for future use.
		ECoarse.Spectrum(s.optimization, output.E2Coarse[:])
		ERefined.Spectrum(s.optimization, output.E2Refined[:])

		// Update the refined filter.
		if !refinedFiltersAdjusted {
			// Do not allow the performance of the coarse filter to affect the
			// adaptation speed of the refined filter just after the coarse
			// filter has been reset.
			disallowLeakageDiverged := sc.coarseFilterResetHangover > 0 && s.useCoarseFilterResetHangover

			ComputeErl(s.optimization, frequencyResponses[ch], s.erl[:])
			sc.refinedGain.Compute(X2Refined, renderSignalAnalyzer, output, s.erl[:],
				sc.refinedFilter.SizePartitions(), saturatedCapture, disallowLeakageDiverged, G)
		} else {
			G.Clear()
		}
		// The C++ constrains the refined filter right here
		// (Adapt(render_buffer, G, &impulse_response)) and then computes its
		// frequency response. Both only concern the refined filter, and the
		// coarse filter update below reads the refined filter only where it
		// copies it, so the constraint waits for the coarse filter's and
		// the transforms of the two run as pairs; the frequency response
		// follows them.
		sc.refinedFilter.adaptAndUpdateSize(renderBuffer, G)
		refinedConstrained := false

		// Update the coarse filter.
		if output.E2RefinedSum < output.E2CoarseSum {
			sc.poorCoarseFilterCounter++
		} else {
			sc.poorCoarseFilterCounter = 0
		}
		if sc.poorCoarseFilterCounter < 5 {
			sc.coarseGain.Compute(X2Coarse, renderSignalAnalyzer, ECoarse,
				sc.coarseFilter.SizePartitions(), saturatedCapture, G)
			sc.coarseFilterResetHangover = max(sc.coarseFilterResetHangover-1, 0)
		} else {
			sc.poorCoarseFilterCounter = 0
			// The coarse filter takes the constrained refined filter.
			sc.refinedFilter.constrainAndUpdateImpulseResponse(&impulseResponses[ch])
			refinedConstrained = true
			sc.coarseFilter.SetFilter(sc.refinedFilter.SizePartitions(), sc.refinedFilter.GetFilter())
			sc.coarseGain.Compute(X2Coarse, renderSignalAnalyzer, ERefined,
				sc.coarseFilter.SizePartitions(), saturatedCapture, G)
			sc.coarseFilterResetHangover = s.config.Filter.CoarseResetHangoverBlocks
		}

		sc.coarseFilter.adaptAndUpdateSize(renderBuffer, G)
		if refinedConstrained {
			sc.coarseFilter.constrain()
		} else {
			afirConstrainPair(&s.fft, &sc.refinedFilter, &impulseResponses[ch], &sc.coarseFilter)
		}
		sc.refinedFilter.ComputeFrequencyResponse(&frequencyResponses[ch])

		// SafeClamp(a, -32768.f, 32767.f); see kernClamp.
		kernClamp(eRefined[:], -32768, 32767)
	}
}
