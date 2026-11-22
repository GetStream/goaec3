package aec3

// Port of subtractor.{h,cc}.

import "math"

// subPredictionErrorPair computes, for the refined (S0) and the coarse (S1)
// filter output of one capture signal y, the scaled time-domain output s and
// the prediction error e = y - s.
func subPredictionErrorPair(fft *aec3FftPair, S0, S1 *FftData, y []float32, tmp0, tmp1 *[FftLength]float32,
	e0, s0, e1, s1 *[BlockSize]float32) {
	fft.Ifft(S0, S1, tmp0, tmp1)
	const scale float32 = 1.0 / FftLengthBy2
	kernScale(tmp0[FftLengthBy2:], scale, s0[:])
	kernSub(y[:BlockSize], s0[:], e0[:])
	kernScale(tmp1[FftLengthBy2:], scale, s1[:])
	kernSub(y[:BlockSize], s1[:], e1[:])
}

// subScaleFilterOutput scales the filter output s by factor and sets e = y -
// s.
func subScaleFilterOutput(y []float32, factor float32, e, s []float32) {
	for k := range y {
		s[k] = float32(s[k] * factor) // rounded before the subtraction (no FMA)
		e[k] = y[k] - s[k]
	}
}

// subFilterMisadjustmentEstimator mirrors
// Subtractor::FilterMisadjustmentEstimator.
type subFilterMisadjustmentEstimator struct {
	nBlocksAcum      int
	e2Acum           float32
	y2Acum           float32
	invMisadjustment float32
	overhang         int
}

const subMisadjustmentNBlocks = 4

// Update takes the output of one block.
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

// GetMisadjustment returns the scale to apply to the filter: half the
// estimated mismatch.
func (m *subFilterMisadjustmentEstimator) GetMisadjustment() float32 {
	return 2 / float32(math.Sqrt(float64(m.invMisadjustment)))
}

// IsAdjustmentNeeded reports whether the energy of the prediction error far
// exceeds that of the microphone signal.
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

// Subtractor cancels the echo linearly with a refined and a coarse adaptive
// filter per capture channel (subtractor.h).
//
// The coefficients of all filters share one array, as do the frequency
// responses and the impulse responses.
type Subtractor struct {
	*subtractorState
	more                      *subMore                                     // with more than inlineChannels channels
	refinedFrequencyResponses [inlineChannels][][FftLengthBy2Plus1]float32 // [channel][partition]
	refinedImpulseResponses   [inlineChannels][]float32                    // [channel][sample]
	channels                  [inlineChannels]subChannel
}

// subtractorState is the state of a Subtractor except its slices.
type subtractorState struct {
	channels [inlineChannels]subChannelState
	h2Spans  [inlineChannels]span
	hSpans   [inlineChannels]span
	// The lengths of the responses, which follow the filter size. A view
	// bound for a call takes them from here, and Process stores them.
	h2Lens [inlineChannels]int32
	hLens  [inlineChannels]int32

	fft                          aec3FftPair
	optimization                 Aec3Optimization
	config                       EchoCanceller3Config
	numCaptureChannels           int
	useCoarseFilterResetHangover bool

	// Scratch of one block.
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
	*subChannelState
	refinedFilter AdaptiveFirFilter
	coarseFilter  AdaptiveFirFilter
}

// subChannelState is the state of a subChannel except the filter
// coefficients.
type subChannelState struct {
	refinedFilter                afirState
	coarseFilter                 afirState
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
	states                    []subChannelState
}

func (s *Subtractor) bind(st *subtractorState, b *binder) {
	s.subtractorState = st
	if b != nil {
		bindPtr(b, &s.more, &b.x.subMore)
	}
	for ch := range s.channels {
		H2, h := &s.refinedFrequencyResponses[ch], &s.refinedImpulseResponses[ch]
		if b.mode == bindMeasure {
			// The arenas hold the responses at their full size.
			st.h2Lens[ch], st.hLens[ch] = int32(len(*H2)), int32(len(*h))
			*H2, *h = (*H2)[:cap(*H2)], (*h)[:cap(*h)]
		}
		b.spec(H2, &st.h2Spans[ch])
		b.f32(h, &st.hSpans[ch])
		if b.mode == bindMove || b.mode == bindView {
			*H2, *h = (*H2)[:st.h2Lens[ch]], (*h)[:st.hLens[ch]]
		}
		s.channels[ch].bind(&st.channels[ch], b)
	}
}

func (sc *subChannel) bind(st *subChannelState, b *binder) {
	sc.subChannelState = st
	sc.refinedFilter.bind(&st.refinedFilter, b)
	sc.coarseFilter.bind(&st.coarseFilter, b)
}

// NewSubtractor mirrors the C++ constructor with field trials at their
// defaults: the reset hangover of the coarse filter is in use.
func NewSubtractor(config *EchoCanceller3Config, numRenderChannels, numCaptureChannels int, optimization Aec3Optimization) *Subtractor {
	s := &Subtractor{}
	s.bind(&subtractorState{}, bindAlone)
	s.init(config, numRenderChannels, numCaptureChannels, optimization)
	return s
}

func (s *Subtractor) init(config *EchoCanceller3Config, numRenderChannels, numCaptureChannels int, optimization Aec3Optimization) {
	st := s.subtractorState
	*st = subtractorState{}
	*s = Subtractor{}
	s.bind(st, bindAlone)
	s.optimization = optimization
	s.config = *config
	s.numCaptureChannels = numCaptureChannels
	s.useCoarseFilterResetHangover = true
	if numCaptureChannels > inlineChannelLimit {
		s.more = &subMore{
			refinedFrequencyResponses: make([][][FftLengthBy2Plus1]float32, numCaptureChannels),
			refinedImpulseResponses:   make([][]float32, numCaptureChannels),
			channels:                  make([]subChannel, numCaptureChannels),
			states:                    make([]subChannelState, numCaptureChannels),
		}
		for ch := range s.more.channels {
			s.more.channels[ch].bind(&s.more.states[ch], bindAlone)
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

// ZeroGainSkipsForTesting returns the number of adaptations the zero-gain
// skip left out, over all filters.
func (s *Subtractor) ZeroGainSkipsForTesting() uint64 {
	var n uint64
	channels := s.channelStates()
	for ch := range channels {
		n += channels[ch].refinedFilter.zeroGainSkips + channels[ch].coarseFilter.zeroGainSkips
	}
	return n
}

// channelStates returns the state of every capture channel.
func (s *Subtractor) channelStates() []subChannel {
	if s.more != nil {
		return s.more.channels
	}
	return s.channels[:s.numCaptureChannels:s.numCaptureChannels]
}

// HandleEchoPathChange resets what the echo path change invalidates.
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

// ExitInitialState switches the filters and gains to their normal configs.
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

// FilterFrequencyResponses returns the frequency responses of the refined
// filters, [channel][partition].
func (s *Subtractor) FilterFrequencyResponses() [][][FftLengthBy2Plus1]float32 {
	if s.more != nil {
		return s.more.refinedFrequencyResponses
	}
	return s.refinedFrequencyResponses[:s.numCaptureChannels:s.numCaptureChannels]
}

// FilterImpulseResponses returns the impulse responses of the refined
// filters, one per channel.
func (s *Subtractor) FilterImpulseResponses() [][]float32 {
	if s.more != nil {
		return s.more.refinedImpulseResponses
	}
	return s.refinedImpulseResponses[:s.numCaptureChannels:s.numCaptureChannels]
}

// Process subtracts the echo of one block. outputs has one element per
// capture channel.
func (s *Subtractor) Process(renderBuffer *RenderBuffer, capture *Block, renderSignalAnalyzer *RenderSignalAnalyzer,
	aecState *AecState, outputs []SubtractorOutput) {
	// The render powers.
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

		// The outputs of the refined and the coarse filter. The two inverse
		// transforms run as a pair.
		sc.refinedFilter.Filter(renderBuffer, S)
		sc.coarseFilter.Filter(renderBuffer, &s.sCoarse)
		subPredictionErrorPair(&s.fft, S, &s.sCoarse, y, &s.predictionTmp, &s.predictionTmp2,
			eRefined, &output.SRefined, eCoarse, &output.SCoarse)

		output.ComputeMetrics(y)

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

		s.fft.ZeroPaddedFft(eRefined[:], eCoarse[:], ERefined, ECoarse)

		ECoarse.Spectrum(s.optimization, output.E2Coarse[:])
		ERefined.Spectrum(s.optimization, output.E2Refined[:])

		if !refinedFiltersAdjusted {
			// A coarse filter that was just reset must not slow the
			// adaptation of the refined filter.
			disallowLeakageDiverged := sc.coarseFilterResetHangover > 0 && s.useCoarseFilterResetHangover

			ComputeErl(s.optimization, frequencyResponses[ch], s.erl[:])
			sc.refinedGain.Compute(X2Refined, renderSignalAnalyzer, output, s.erl[:],
				sc.refinedFilter.SizePartitions(), saturatedCapture, disallowLeakageDiverged, G)
		} else {
			G.Clear()
		}
		// The C++ constrains the refined filter here and then computes its
		// frequency response. The update of the coarse filter below reads
		// the refined filter only where it copies it, so the constraint
		// waits and its transforms run as pairs with those of the coarse
		// filter. The frequency response follows them.
		sc.refinedFilter.adaptAndUpdateSizeKnownRender(renderBuffer, G, X2Refined, refinedSize)
		refinedConstrained := false

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

		sc.coarseFilter.adaptAndUpdateSizeKnownRender(renderBuffer, G, X2Coarse, coarseSize)
		if refinedConstrained {
			sc.coarseFilter.constrain()
		} else {
			afirConstrainPair(&s.fft, &sc.refinedFilter, &impulseResponses[ch], &sc.coarseFilter)
		}
		sc.refinedFilter.updateFrequencyResponse(&frequencyResponses[ch])

		// SafeClamp(a, -32768.f, 32767.f). See kernClampGeneric.
		kernClamp(eRefined[:], -32768, 32767)
	}
	if s.more == nil {
		for ch := range channels {
			s.h2Lens[ch], s.hLens[ch] = int32(len(frequencyResponses[ch])), int32(len(impulseResponses[ch]))
		}
	}
}
