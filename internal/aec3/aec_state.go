package aec3

// LinearFilterQuality is a std::optional<float>: Value is meaningful only
// when Ok is set.
type LinearFilterQuality struct {
	Value float32
	Ok    bool
}

// stAccumulate is std::accumulate(..., 0.f): a sequential float sum.
func stAccumulate(x []float32) float32 {
	var acc float32
	for _, v := range x {
		acc += v
	}
	return acc
}

func stBoolToFloat(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

// stMul, stSub and stDiv evaluate float constants with float rounding (as
// C++ constexpr float arithmetic does) rather than Go's exact constant
// arithmetic.
func stMul(a, b float32) float32 { return a * b }
func stSub(a, b float32) float32 { return a - b }
func stDiv(a, b float32) float32 { return a / b }

// stInitialState mirrors AecState::InitialState.
type stInitialState struct {
	conservativeInitialPhase       bool
	initialStateSeconds            float32
	transitionTriggered            bool
	initialState                   bool
	strongNotSaturatedRenderBlocks int
}

func (s *stInitialState) Reset() {
	s.initialState = true
	s.strongNotSaturatedRenderBlocks = 0
}

func (s *stInitialState) Update(activeRender, saturatedCapture bool) {
	if activeRender && !saturatedCapture {
		s.strongNotSaturatedRenderBlocks++
	}
	prevInitialState := s.initialState
	if s.conservativeInitialPhase {
		s.initialState = s.strongNotSaturatedRenderBlocks < 5*NumBlocksPerSecond
	} else {
		s.initialState = float32(s.strongNotSaturatedRenderBlocks) < s.initialStateSeconds*NumBlocksPerSecond
	}
	s.transitionTriggered = !s.initialState && prevInitialState
}

// stFilterDelay mirrors AecState::FilterDelay. The per-channel delays
// (filterDelaysBlocks) live with the AecState.
type stFilterDelay struct {
	delayHeadroomBlocks   int
	externalDelayReported bool
	minFilterDelay        int
	externalDelay         DelayEstimate
	externalDelayValid    bool
}

func (f *stFilterDelay) Update(filterDelaysBlocks, analyzerFilterDelayEstimatesBlocks []int, externalDelay *DelayEstimate,
	blocksWithProperFilterAdaptation int) {
	if externalDelay != nil && (!f.externalDelayValid || f.externalDelay.Delay != externalDelay.Delay) {
		f.externalDelay = *externalDelay
		f.externalDelayValid = true
		f.externalDelayReported = true
	}
	delayEstimatorMayNotHaveConverged := blocksWithProperFilterAdaptation < 2*NumBlocksPerSecond
	if delayEstimatorMayNotHaveConverged && f.externalDelayValid {
		for i := range filterDelaysBlocks {
			filterDelaysBlocks[i] = f.delayHeadroomBlocks
		}
	} else {
		copy(filterDelaysBlocks, analyzerFilterDelayEstimatesBlocks)
	}
	m := filterDelaysBlocks[0]
	for _, d := range filterDelaysBlocks[1:] {
		m = min(m, d)
	}
	f.minFilterDelay = m
}

// stFilteringQualityAnalyzer mirrors AecState::FilteringQualityAnalyzer.
// The per-channel flags (usableLinearFilterEstimates) live with the
// AecState.
type stFilteringQualityAnalyzer struct {
	useLinearFilter              bool
	overallUsableLinearEstimates bool
	filterUpdateBlocksSinceReset int
	filterUpdateBlocksSinceStart int
	convergenceSeen              bool
}

func (f *stFilteringQualityAnalyzer) Reset(usableLinearFilterEstimates []bool) {
	clear(usableLinearFilterEstimates)
	f.overallUsableLinearEstimates = false
	f.filterUpdateBlocksSinceReset = 0
}

var (
	stConvergeAtStartupBlocks = stMul(NumBlocksPerSecond, 0.4)
	stConvergeAtResetBlocks   = stMul(NumBlocksPerSecond, 0.2)
)

func (f *stFilteringQualityAnalyzer) Update(usableLinearFilterEstimates []bool, activeRender, transparentMode, saturatedCapture bool,
	externalDelay *DelayEstimate, anyFilterConverged bool) {
	filterUpdate := activeRender && !saturatedCapture
	if filterUpdate {
		f.filterUpdateBlocksSinceReset++
		f.filterUpdateBlocksSinceStart++
	}
	f.convergenceSeen = f.convergenceSeen || anyFilterConverged
	sufficientDataToConvergeAtStartup := float32(f.filterUpdateBlocksSinceStart) > stConvergeAtStartupBlocks
	sufficientDataToConvergeAtReset := sufficientDataToConvergeAtStartup &&
		float32(f.filterUpdateBlocksSinceReset) > stConvergeAtResetBlocks
	f.overallUsableLinearEstimates = sufficientDataToConvergeAtStartup && sufficientDataToConvergeAtReset
	f.overallUsableLinearEstimates = f.overallUsableLinearEstimates && (externalDelay != nil || f.convergenceSeen)
	f.overallUsableLinearEstimates = f.overallUsableLinearEstimates && !transparentMode
	if f.useLinearFilter {
		for i := range usableLinearFilterEstimates {
			usableLinearFilterEstimates[i] = f.overallUsableLinearEstimates
		}
	}
}

// stSaturationDetector mirrors AecState::SaturationDetector.
type stSaturationDetector struct {
	saturatedEcho bool
}

func (s *stSaturationDetector) Update(x *Block, saturatedCapture, usableLinearEstimate bool,
	subtractorOutput []SubtractorOutput, echoPathGain float32) {
	s.saturatedEcho = false
	if !saturatedCapture {
		return
	}
	if usableLinearEstimate {
		const saturationThreshold float32 = 20000
		for ch := range subtractorOutput {
			s.saturatedEcho = s.saturatedEcho ||
				(subtractorOutput[ch].SRefinedMaxAbs > saturationThreshold ||
					subtractorOutput[ch].SCoarseMaxAbs > saturationThreshold)
		}
	} else {
		var maxSample float32
		for ch := 0; ch < x.NumChannels(); ch++ {
			for _, sample := range x.View(0, ch) {
				maxSample = stdMax(maxSample, fabsf(sample))
			}
		}
		const margin float32 = 10
		peakEchoAmplitude := maxSample * echoPathGain * margin
		s.saturatedEcho = s.saturatedEcho || peakEchoAmplitude > 32000
	}
}

// stTransparentKind selects the transparent mode classifier of an AecState.
type stTransparentKind uint8

const (
	stTransparentNone stTransparentKind = iota
	stTransparentLegacy
	stTransparentHmm
)

// stTransparentState is the transparent mode classifier of an AecState: one
// of the two implementations of TransparentMode, or none.
type stTransparentState struct {
	kind   stTransparentKind
	legacy LegacyTransparentMode
	hmm    TransparentModeHmm
}

func (t *stTransparentState) Active() bool {
	switch t.kind {
	case stTransparentLegacy:
		return t.legacy.Active()
	case stTransparentHmm:
		return t.hmm.Active()
	}
	return false
}

func (t *stTransparentState) Reset() {
	switch t.kind {
	case stTransparentLegacy:
		t.legacy.Reset()
	case stTransparentHmm:
		t.hmm.Reset()
	}
}

func (t *stTransparentState) Update(filterDelayBlocks int, anyFilterConsistent, anyFilterConverged,
	anyCoarseFilterConverged, allFiltersDiverged, activeRender, saturatedCapture bool) {
	switch t.kind {
	case stTransparentLegacy:
		t.legacy.Update(filterDelayBlocks, anyFilterConsistent, anyFilterConverged, anyCoarseFilterConverged,
			allFiltersDiverged, activeRender, saturatedCapture)
	case stTransparentHmm:
		t.hmm.Update(filterDelayBlocks, anyFilterConsistent, anyFilterConverged, anyCoarseFilterConverged,
			allFiltersDiverged, activeRender, saturatedCapture)
	}
}

// AecState mirrors aec_state.h: the state and the conditions for the echo
// removal functionality. Field trials are at their defaults.
//
// An AecState is the view of an aecStateState, which holds the estimators
// and analyzers and the per-channel state of up to inlineChannels capture
// channels; the view holds the views of those that have pointers.
type AecState struct {
	*aecStateState
	more                     *aecMore // with more than inlineChannels channels
	subtractorOutputAnalyzer SubtractorOutputAnalyzer
	filterAnalyzer           FilterAnalyzer
	reverbModelEstimator     ReverbModelEstimator
	erleEstimator            ErleEstimator
}

// aecStateState is the state of an AecState but its pointers.
type aecStateState struct {
	subtractorOutputAnalyzer subtractorOutputAnalyzerState
	filterAnalyzer           filterAnalyzerState
	reverbModelEstimator     reverbModelEstimatorState
	erleEstimator            erleEstimatorState

	config                                      EchoCanceller3Config
	numCaptureChannels                          int
	deactivateInitialStateResetAtEchoPathChange bool
	fullResetAtEchoPathChange                   bool
	subtractorAnalyzerResetAtEchoPathChange     bool
	initialState                                stInitialState
	delayState                                  stFilterDelay
	transparentState                            stTransparentState
	filterQualityState                          stFilteringQualityAnalyzer
	saturationDetector                          stSaturationDetector
	erlEstimator                                ErlEstimator
	strongNotSaturatedRenderBlocks              int
	blocksWithActiveRender                      int
	captureSignalSaturation                     bool
	echoAudibility                              EchoAudibility
	avgRenderReverb                             ReverbModel

	// The per-channel state of delayState and filterQualityState for up to
	// inlineChannels channels.
	inline struct {
		filterDelaysBlocks          [inlineChannels]int
		usableLinearFilterEstimates [inlineChannels]bool
	}
}

// aecMore is the per-channel state of the delayState and the
// filterQualityState of an AecState with more than inlineChannels capture
// channels.
type aecMore struct {
	filterDelaysBlocks          []int
	usableLinearFilterEstimates []bool
	adjustedFilters             [][]float32 // scratch of Update
}

// filterDelaysBlocks returns the per-channel delays of delayState.
func (a *AecState) filterDelaysBlocks() []int {
	if a.more != nil {
		return a.more.filterDelaysBlocks
	}
	return a.inline.filterDelaysBlocks[:a.numCaptureChannels:a.numCaptureChannels]
}

// usableLinearFilterEstimates returns the per-channel flags of
// filterQualityState.
func (a *AecState) usableLinearFilterEstimates() []bool {
	if a.more != nil {
		return a.more.usableLinearFilterEstimates
	}
	return a.inline.usableLinearFilterEstimates[:a.numCaptureChannels:a.numCaptureChannels]
}

// bind makes a the view of st.
func (a *AecState) bind(st *aecStateState, b *binder) {
	a.aecStateState = st
	if b != nil {
		bindPtr(b, &a.more, &b.x.aecMore)
	}
	a.subtractorOutputAnalyzer.bind(&st.subtractorOutputAnalyzer, b)
	a.filterAnalyzer.bind(&st.filterAnalyzer, b)
	a.reverbModelEstimator.bind(&st.reverbModelEstimator, b)
	a.erleEstimator.bind(&st.erleEstimator, b)
}

// aecStorageSize returns the number of samples an AecState keeps for config:
// those of its FilterAnalyzer and of its ReverbModelEstimator.
func aecStorageSize(config *EchoCanceller3Config, numCaptureChannels int) int {
	return faStorageSize(config, numCaptureChannels) + rmStorageSize(config, numCaptureChannels)
}

// NewAecState builds an AecState with the transparent mode classifier of
// CreateTransparentMode. The estimators and analyzers are part of its
// state, which is one heap object; the buffers whose size depends on the
// filter length are another.
func NewAecState(config *EchoCanceller3Config, numCaptureChannels int) *AecState {
	a := &AecState{}
	a.bind(&aecStateState{}, bindAlone)
	a.initDefault(config, numCaptureChannels, make([]float32, aecStorageSize(config, numCaptureChannels)))
	return a
}

// initDefault sets up a in place as NewAecState does, over storage, which
// holds aecStorageSize zeros.
func (a *AecState) initDefault(config *EchoCanceller3Config, numCaptureChannels int, storage []float32) {
	a.init(config, numCaptureChannels, storage)
	if !config.EpStrength.BoundedErl {
		a.transparentState.kind = stTransparentLegacy
		a.transparentState.legacy = *NewLegacyTransparentMode(config)
	}
}

// NewAecStateWithTransparentMode builds an AecState with an explicit
// transparent mode classifier (nil disables it), which is what the C++
// selects through field trials. The AecState continues from a copy of tm,
// which is a *LegacyTransparentMode or a *TransparentModeHmm.
func NewAecStateWithTransparentMode(config *EchoCanceller3Config, numCaptureChannels int, tm TransparentMode) *AecState {
	a := &AecState{}
	a.bind(&aecStateState{}, bindAlone)
	a.init(config, numCaptureChannels, make([]float32, aecStorageSize(config, numCaptureChannels)))
	switch t := tm.(type) {
	case nil:
	case *LegacyTransparentMode:
		if t != nil {
			a.transparentState.kind = stTransparentLegacy
			a.transparentState.legacy = *t
		}
	case *TransparentModeHmm:
		if t != nil {
			a.transparentState.kind = stTransparentHmm
			a.transparentState.hmm = *t
		}
	default:
		panic("aec3: unknown TransparentMode implementation")
	}
	return a
}

// init sets up a in place, without a transparent mode classifier, over
// storage, which holds aecStorageSize zeros.
func (a *AecState) init(config *EchoCanceller3Config, numCaptureChannels int, storage []float32) {
	st := a.aecStateState
	*st = aecStateState{
		config:                                  *config,
		numCaptureChannels:                      numCaptureChannels,
		fullResetAtEchoPathChange:               true,
		subtractorAnalyzerResetAtEchoPathChange: true,
	}
	*a = AecState{}
	a.bind(st, bindAlone)
	c := &a.config
	if n := numCaptureChannels; n > inlineChannelLimit {
		a.more = &aecMore{
			filterDelaysBlocks:          make([]int, n),
			usableLinearFilterEstimates: make([]bool, n),
			adjustedFilters:             make([][]float32, n),
		}
	}
	a.initialState = stInitialState{
		conservativeInitialPhase: c.Filter.ConservativeInitialPhase,
		initialStateSeconds:      c.Filter.InitialStateSeconds,
	}
	a.initialState.Reset()
	headroom := c.Delay.DelayHeadroomSamples / BlockSize
	a.delayState = stFilterDelay{
		delayHeadroomBlocks: headroom,
		minFilterDelay:      headroom,
	}
	filterDelaysBlocks := a.filterDelaysBlocks()
	for i := range filterDelaysBlocks {
		filterDelaysBlocks[i] = headroom
	}
	a.filterQualityState = stFilteringQualityAnalyzer{
		useLinearFilter: c.Filter.UseLinearFilter,
	}
	a.erlEstimator.init(2 * NumBlocksPerSecond)
	a.erleEstimator.init(2*NumBlocksPerSecond, c, numCaptureChannels)
	fa := faStorageSize(c, numCaptureChannels)
	a.filterAnalyzer.init(c, numCaptureChannels, storage[:fa:fa])
	a.echoAudibility.init(c.EchoAudibility.UseStationarityPropertiesAtInit)
	a.reverbModelEstimator.init(c, numCaptureChannels, storage[fa:])
	a.subtractorOutputAnalyzer.init(numCaptureChannels)
}

// UsableLinearEstimate reports whether the echo subtractor can be used to
// determine the residual echo.
func (a *AecState) UsableLinearEstimate() bool {
	return a.filterQualityState.overallUsableLinearEstimates && a.config.Filter.UseLinearFilter
}

// UseLinearFilterOutput reports whether the echo subtractor output should be
// used as output.
func (a *AecState) UseLinearFilterOutput() bool {
	return a.filterQualityState.overallUsableLinearEstimates && a.config.Filter.UseLinearFilter
}

// ActiveRender reports whether the render signal is currently active.
func (a *AecState) ActiveRender() bool { return a.blocksWithActiveRender > 200 }

// GetResidualEchoScaling returns the residual echo scaling matching the
// audibility.
func (a *AecState) GetResidualEchoScaling(residualScaling []float32) {
	var filterHasHadTimeToConverge bool
	if a.config.Filter.ConservativeInitialPhase {
		filterHasHadTimeToConverge = float32(a.strongNotSaturatedRenderBlocks) >= 1.5*NumBlocksPerSecond
	} else {
		filterHasHadTimeToConverge = float32(a.strongNotSaturatedRenderBlocks) >= stMul(0.8, NumBlocksPerSecond)
	}
	a.echoAudibility.GetResidualEchoScaling(filterHasHadTimeToConverge, residualScaling)
}

// UseStationarityProperties reports whether stationarity properties are used.
func (a *AecState) UseStationarityProperties() bool {
	return a.config.EchoAudibility.UseStationarityProperties
}

// Erle returns the ERLE.
func (a *AecState) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	return a.erleEstimator.Erle(onsetCompensated)
}

// ErleUnbounded returns the non-capped ERLE.
func (a *AecState) ErleUnbounded() [][FftLengthBy2Plus1]float32 {
	return a.erleEstimator.ErleUnbounded()
}

// FullBandErleLog2 returns the fullband ERLE estimate in log2 units.
func (a *AecState) FullBandErleLog2() float32 { return a.erleEstimator.FullbandErleLog2() }

// Erl returns the ERL.
func (a *AecState) Erl() *[FftLengthBy2Plus1]float32 { return a.erlEstimator.Erl() }

// ErlTimeDomain returns the time-domain ERL.
func (a *AecState) ErlTimeDomain() float32 { return a.erlEstimator.ErlTimeDomain() }

// MinDirectPathFilterDelay returns the delay estimate based on the linear
// filter.
func (a *AecState) MinDirectPathFilterDelay() int { return a.delayState.minFilterDelay }

// SaturatedCapture reports whether the capture signal is saturated.
func (a *AecState) SaturatedCapture() bool { return a.captureSignalSaturation }

// SaturatedEcho reports whether the echo signal is saturated.
func (a *AecState) SaturatedEcho() bool { return a.saturationDetector.saturatedEcho }

// UpdateCaptureSaturation updates the capture signal saturation.
func (a *AecState) UpdateCaptureSaturation(captureSignalSaturation bool) {
	a.captureSignalSaturation = captureSignalSaturation
}

// TransparentModeActive reports whether the transparent mode is active.
func (a *AecState) TransparentModeActive() bool {
	return a.transparentState.Active()
}

// ReverbDecay returns the decay factor for the echo reverberation.
func (a *AecState) ReverbDecay(mild bool) float32 { return a.reverbModelEstimator.ReverbDecay(mild) }

// GetReverbFrequencyResponse returns the frequency response of the
// reverberant echo.
func (a *AecState) GetReverbFrequencyResponse() *[FftLengthBy2Plus1]float32 {
	return a.reverbModelEstimator.GetReverbFrequencyResponse()
}

// TransitionTriggered reports whether the transition out of the initial
// state has been triggered.
func (a *AecState) TransitionTriggered() bool { return a.initialState.transitionTriggered }

// FilterLengthBlocks returns the filter length in blocks.
func (a *AecState) FilterLengthBlocks() int { return a.filterAnalyzer.FilterLengthBlocks() }

// HandleEchoPathChange takes action at an echo path change.
func (a *AecState) HandleEchoPathChange(echoPathVariability EchoPathVariability) {
	if a.fullResetAtEchoPathChange && echoPathVariability.DelayChange != DelayAdjustmentNone {
		a.filterAnalyzer.Reset()
		a.captureSignalSaturation = false
		a.strongNotSaturatedRenderBlocks = 0
		a.blocksWithActiveRender = 0
		if !a.deactivateInitialStateResetAtEchoPathChange {
			a.initialState.Reset()
		}
		a.transparentState.Reset()
		a.erleEstimator.Reset(true)
		a.erlEstimator.Reset()
		a.filterQualityState.Reset(a.usableLinearFilterEstimates())
	} else if echoPathVariability.GainChange {
		a.erleEstimator.Reset(false)
	}
	if a.subtractorAnalyzerResetAtEchoPathChange {
		a.subtractorOutputAnalyzer.HandleEchoPathChange()
	}
}

func stComputeAvgRenderReverb(spectrumBuffer *SpectrumBuffer, delayBlocks int, reverbDecay float32,
	reverbModel *ReverbModel, reverbPowerSpectrum *[FftLengthBy2Plus1]float32) {
	numRenderChannels := len(spectrumBuffer.At(0))
	idxAtDelay := spectrumBuffer.OffsetIndex(spectrumBuffer.Read, delayBlocks)
	idxPast := spectrumBuffer.IncIndex(idxAtDelay)
	var X2Data [FftLengthBy2Plus1]float32
	var X2 *[FftLengthBy2Plus1]float32
	if numRenderChannels > 1 {
		averageChannels := func(spectrumBand0 [][FftLengthBy2Plus1]float32, renderPower *[FftLengthBy2Plus1]float32) {
			*renderPower = [FftLengthBy2Plus1]float32{}
			for ch := 0; ch < numRenderChannels; ch++ {
				vectorAccumulate(spectrumBand0[ch][:], renderPower[:])
			}
			normalizer := 1 / float32(numRenderChannels)
			kernScale(renderPower[:], normalizer, renderPower[:])
		}
		averageChannels(spectrumBuffer.At(idxPast), &X2Data)
		reverbModel.UpdateReverbNoFreqShaping(X2Data[:], 1, reverbDecay)
		averageChannels(spectrumBuffer.At(idxAtDelay), &X2Data)
		X2 = &X2Data
	} else {
		reverbModel.UpdateReverbNoFreqShaping(spectrumBuffer.At(idxPast)[0][:], 1, reverbDecay)
		X2 = &spectrumBuffer.At(idxAtDelay)[0]
	}
	reverbPower := reverbModel.Reverb()
	kernAdd(X2[:], reverbPower[:], reverbPowerSpectrum[:])
}

// Update updates the AEC state. externalDelay is nil when no external delay
// estimate is available.
func (a *AecState) Update(externalDelay *DelayEstimate, adaptiveFilterFrequencyResponses [][][FftLengthBy2Plus1]float32,
	adaptiveFilterImpulseResponses [][]float32, renderBuffer *RenderBuffer, E2Refined, Y2 [][FftLengthBy2Plus1]float32,
	subtractorOutput []SubtractorOutput) {
	anyFilterConverged, anyCoarseFilterConverged, allFiltersDiverged := a.subtractorOutputAnalyzer.Update(subtractorOutput)

	anyFilterConsistent, maxEchoPathGain := a.filterAnalyzer.Update(adaptiveFilterImpulseResponses, renderBuffer)

	if a.config.Filter.UseLinearFilter {
		a.delayState.Update(a.filterDelaysBlocks(), a.filterAnalyzer.FilterDelaysBlocks(), externalDelay, a.strongNotSaturatedRenderBlocks)
	}

	alignedRenderBlock := renderBuffer.GetBlock(-a.delayState.minFilterDelay)

	activeRender := false
	activeRenderThreshold := a.config.RenderLevels.ActiveRenderLimit * a.config.RenderLevels.ActiveRenderLimit * FftLengthBy2
	for ch := 0; ch < alignedRenderBlock.NumChannels(); ch++ {
		var renderEnergy float32
		for _, v := range alignedRenderBlock.View(0, ch) {
			renderEnergy += float32(v * v)
		}
		if renderEnergy > activeRenderThreshold {
			activeRender = true
			break
		}
	}
	if activeRender {
		a.blocksWithActiveRender++
	}
	if activeRender && !a.SaturatedCapture() {
		a.strongNotSaturatedRenderBlocks++
	}

	var avgRenderSpectrumWithReverb [FftLengthBy2Plus1]float32
	stComputeAvgRenderReverb(renderBuffer.GetSpectrumBuffer(), a.delayState.minFilterDelay, a.ReverbDecay(false),
		&a.avgRenderReverb, &avgRenderSpectrumWithReverb)

	if a.config.EchoAudibility.UseStationarityProperties {
		a.echoAudibility.Update(renderBuffer, a.avgRenderReverb.Reverb()[:], a.delayState.minFilterDelay,
			a.delayState.externalDelayReported)
	}

	if a.initialState.transitionTriggered {
		a.erleEstimator.Reset(false)
	}

	a.erleEstimator.Update(renderBuffer, adaptiveFilterFrequencyResponses, &avgRenderSpectrumWithReverb, Y2, E2Refined,
		a.subtractorOutputAnalyzer.ConvergedFilters())

	a.erlEstimator.Update(a.subtractorOutputAnalyzer.ConvergedFilters(),
		renderBuffer.Spectrum(a.delayState.minFilterDelay), Y2)

	if a.config.EpStrength.EchoCanSaturate {
		a.saturationDetector.Update(&alignedRenderBlock, a.SaturatedCapture(), a.UsableLinearEstimate(),
			subtractorOutput, maxEchoPathGain)
	}

	a.initialState.Update(activeRender, a.SaturatedCapture())

	a.transparentState.Update(a.delayState.minFilterDelay, anyFilterConsistent, anyFilterConverged,
		anyCoarseFilterConverged, allFiltersDiverged, activeRender, a.SaturatedCapture())

	a.filterQualityState.Update(a.usableLinearFilterEstimates(), activeRender, a.TransparentModeActive(), a.SaturatedCapture(), externalDelay,
		anyFilterConverged)

	stationaryBlock := a.config.EchoAudibility.UseStationarityProperties && a.echoAudibility.IsBlockStationary()

	var adjusted [inlineChannels][]float32
	adjustedFilters := adjusted[:]
	if a.more != nil {
		adjustedFilters = a.more.adjustedFilters
	}
	a.reverbModelEstimator.Update(a.filterAnalyzer.adjustedFilters(adjustedFilters), adaptiveFilterFrequencyResponses,
		a.erleEstimator.GetInstLinearQualityEstimates(), a.filterDelaysBlocks(),
		a.usableLinearFilterEstimates(), stationaryBlock)
}
