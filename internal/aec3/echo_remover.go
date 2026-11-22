package aec3

// Port of echo_remover.{h,cc}: the EchoRemover interface and
// EchoRemoverImpl.

// EchoControlMetrics mirrors EchoControl::Metrics (api/audio/echo_control.h).
type EchoControlMetrics struct {
	EchoReturnLoss            float64
	EchoReturnLossEnhancement float64
	DelayMs                   int
}

// EchoRemover removes the echo from the capture signal.
//
// The components that hold pointers come first.
type EchoRemover struct {
	*echoRemoverState
	more       *erMore // with more than inlineChannels channels
	aecState   AecState
	subtractor Subtractor

	suppressionGain   SuppressionGain
	cng               ComfortNoiseGenerator
	suppressionFilter SuppressionFilter
}

// echoRemoverState is the state of an EchoRemover except its pointers.
type echoRemoverState struct {
	_                     noCopy
	aecState              aecStateState
	subtractor            subtractorState
	suppressionGain       suppressionGainState
	cng                   cngState
	suppressionFilter     suppressionFilterState
	residualEchoEstimator ResidualEchoEstimator
	renderSignalAnalyzer  RenderSignalAnalyzer
	metrics               EchoRemoverMetrics

	config                          EchoCanceller3Config
	fft                             aec3FftPair
	optimization                    Aec3Optimization
	sampleRateHz                    int
	numRenderChannels               int
	numCaptureChannels              int
	useCoarseFilterOutput           bool
	echoLeakageDetected             bool
	captureOutputUsed               bool
	blockCounter                    int
	gainChangeHangover              int
	refinedFilterOutputLastSelected bool
	erChannels
}

// erChannels is the per-channel state of an EchoRemover. All but channels is
// scratch of one block.
type erChannels struct {
	channels             [inlineChannels]erChannel
	Y2                   [inlineChannels][FftLengthBy2Plus1]float32
	E2                   [inlineChannels][FftLengthBy2Plus1]float32
	R2                   [inlineChannels][FftLengthBy2Plus1]float32
	R2Unbounded          [inlineChannels][FftLengthBy2Plus1]float32
	S2Linear             [inlineChannels][FftLengthBy2Plus1]float32
	Y                    [inlineChannels]FftData
	E                    [inlineChannels]FftData
	comfortNoise         [inlineChannels]FftData
	highBandComfortNoise [inlineChannels]FftData
	subtractorOutput     [inlineChannels]SubtractorOutput
	G                    [FftLengthBy2Plus1]float32
}

// erMore is the per-channel state of an EchoRemover with more than
// inlineChannels capture channels.
type erMore struct {
	channels             []erChannel
	Y2                   [][FftLengthBy2Plus1]float32
	E2                   [][FftLengthBy2Plus1]float32
	R2                   [][FftLengthBy2Plus1]float32
	R2Unbounded          [][FftLengthBy2Plus1]float32
	S2Linear             [][FftLengthBy2Plus1]float32
	Y                    []FftData
	E                    []FftData
	comfortNoise         []FftData
	highBandComfortNoise []FftData
	subtractorOutput     []SubtractorOutput
}

// erChannel holds what the EchoRemover keeps of one capture channel: the
// previous block of the capture signal and of the linear filter output, and
// the current linear filter output.
type erChannel struct {
	eOld [FftLengthBy2]float32
	yOld [FftLengthBy2]float32
	e    [FftLengthBy2]float32
}

func (e *EchoRemover) bind(st *echoRemoverState, b *binder) {
	e.echoRemoverState = st
	if b != nil {
		bindPtr(b, &e.more, &b.x.erMore)
	}
	e.aecState.bind(&st.aecState, b)
	e.subtractor.bind(&st.subtractor, b)
	e.suppressionGain.bind(&st.suppressionGain, b)
	e.cng.bind(&st.cng, b)
	e.suppressionFilter.bind(&st.suppressionFilter, b)
}

// NewEchoRemover mirrors EchoRemover::Create in the lane of
// DetectOptimization.
func NewEchoRemover(config *EchoCanceller3Config, sampleRateHz, numRenderChannels, numCaptureChannels int) *EchoRemover {
	return NewEchoRemoverWithOptimization(config, sampleRateHz, numRenderChannels, numCaptureChannels,
		DetectOptimization())
}

// NewEchoRemoverWithOptimization is NewEchoRemover in the given lane.
func NewEchoRemoverWithOptimization(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, optimization Aec3Optimization) *EchoRemover {
	e := &EchoRemover{}
	e.bind(&echoRemoverState{}, bindAlone)
	e.init(config, sampleRateHz, numRenderChannels, numCaptureChannels, optimization)
	return e
}

func (e *EchoRemover) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, optimization Aec3Optimization) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate")
	}
	st := e.echoRemoverState
	*st = echoRemoverState{}
	*e = EchoRemover{}
	e.bind(st, bindAlone)
	e.config = *config
	e.optimization = optimization
	e.sampleRateHz = sampleRateHz
	e.numRenderChannels = numRenderChannels
	e.numCaptureChannels = numCaptureChannels
	e.captureOutputUsed = true
	e.refinedFilterOutputLastSelected = true
	e.metrics.resetMetrics()
	if n := numCaptureChannels; n > inlineChannelLimit {
		e.more = &erMore{
			channels:             make([]erChannel, n),
			Y2:                   make([][FftLengthBy2Plus1]float32, n),
			E2:                   make([][FftLengthBy2Plus1]float32, n),
			R2:                   make([][FftLengthBy2Plus1]float32, n),
			R2Unbounded:          make([][FftLengthBy2Plus1]float32, n),
			S2Linear:             make([][FftLengthBy2Plus1]float32, n),
			Y:                    make([]FftData, n),
			E:                    make([]FftData, n),
			comfortNoise:         make([]FftData, n),
			highBandComfortNoise: make([]FftData, n),
			subtractorOutput:     make([]SubtractorOutput, n),
		}
	}
	e.useCoarseFilterOutput = config.Filter.EnableCoarseFilterOutputUsage
	e.subtractor.init(config, numRenderChannels, numCaptureChannels, optimization)
	e.suppressionGain.init(config, optimization, sampleRateHz, numCaptureChannels)
	e.cng.init(config, optimization, numCaptureChannels)
	e.suppressionFilter.init(optimization, sampleRateHz, numCaptureChannels)
	e.renderSignalAnalyzer = RenderSignalAnalyzer{strongPeakFreezeDuration: config.Filter.Refined.LengthBlocks}
	e.residualEchoEstimator.init(config, numRenderChannels)
	e.aecState.initDefault(&e.config, numCaptureChannels, make([]float32, aecStorageSize(&e.config, numCaptureChannels)))
}

// GetMetrics returns the current metrics. It leaves DelayMs at zero.
func (e *EchoRemover) GetMetrics() EchoControlMetrics {
	// The ERL is inverted: from gain to attenuation.
	return EchoControlMetrics{
		EchoReturnLoss:            -10.0 * float64(Log10f(e.aecState.ErlTimeDomain())),
		EchoReturnLossEnhancement: float64(Log2TodB(e.aecState.FullBandErleLog2())),
	}
}

// ProcessCapture removes the echo from one capture block, which the render
// signal is aligned with. externalDelay is nil without an external delay
// estimate; linearOutput is nil when the output of the linear filter is not
// wanted.
func (e *EchoRemover) ProcessCapture(echoPathVariability EchoPathVariability, captureSignalSaturation bool,
	externalDelay *DelayEstimate, renderBuffer *RenderBuffer, linearOutput, capture *Block) {
	e.blockCounter++
	x := renderBuffer.GetBlock(0)
	y := capture
	nch := e.numCaptureChannels
	var more erMore
	if e.more != nil {
		more = *e.more
	}
	channels := channelState(&e.channels, nch, more.channels)
	Y2 := channelState(&e.Y2, nch, more.Y2)
	E2 := channelState(&e.E2, nch, more.E2)
	R2 := channelState(&e.R2, nch, more.R2)
	R2Unbounded := channelState(&e.R2Unbounded, nch, more.R2Unbounded)
	S2Linear := channelState(&e.S2Linear, nch, more.S2Linear)
	Y := channelState(&e.Y, nch, more.Y)
	E := channelState(&e.E, nch, more.E)
	comfortNoise := channelState(&e.comfortNoise, nch, more.comfortNoise)
	highBandComfortNoise := channelState(&e.highBandComfortNoise, nch, more.highBandComfortNoise)
	subtractorOutput := channelState(&e.subtractorOutput, nch, more.subtractorOutput)

	e.aecState.UpdateCaptureSaturation(captureSignalSaturation)

	if echoPathVariability.AudioPathChanged() {
		// A gain change is acted on once per frame.
		if echoPathVariability.GainChange {
			if e.gainChangeHangover == 0 {
				const maxBlocksPerFrame = 3
				e.gainChangeHangover = maxBlocksPerFrame
			} else {
				echoPathVariability.GainChange = false
			}
		}

		e.subtractor.HandleEchoPathChange(echoPathVariability)
		e.aecState.HandleEchoPathChange(echoPathVariability)

		if echoPathVariability.DelayChange != DelayAdjustmentNone {
			e.suppressionGain.SetInitialState(true)
		}
	}
	if e.gainChangeHangover > 0 {
		e.gainChangeHangover--
	}

	e.renderSignalAnalyzer.Update(renderBuffer, e.aecState.MinDirectPathFilterDelay(), true)

	if e.aecState.TransitionTriggered() {
		e.subtractor.ExitInitialState()
		e.suppressionGain.SetInitialState(false)
	}

	e.subtractor.Process(renderBuffer, y, &e.renderSignalAnalyzer, &e.aecState, subtractorOutput)

	for ch := range channels {
		c := &channels[ch]
		e.formLinearFilterOutput(&subtractorOutput[ch], &c.e)
		// WindowedPaddedFft of y and of e, as a pair.
		y0 := y.View(0, ch)
		e.fft.PaddedFft(y0, c.yOld[:], c.e[:], c.eOld[:], &Y[ch], &E[ch])
		copy(c.yOld[:], y0)
		c.eOld = c.e
		erLinearEchoPower(&E[ch], &Y[ch], &S2Linear[ch])
		Y[ch].Spectrum(e.optimization, Y2[ch][:])
		E[ch].Spectrum(e.optimization, E2[ch][:])
	}

	if linearOutput != nil {
		for ch := 0; ch < nch; ch++ {
			copy(linearOutput.View(0, ch), channels[ch].e[:])
		}
	}

	e.aecState.Update(externalDelay, e.subtractor.FilterFrequencyResponses(), e.subtractor.FilterImpulseResponses(),
		renderBuffer, E2, Y2, subtractorOutput)

	YFft := Y
	if e.aecState.UseLinearFilterOutput() {
		YFft = E
	}

	e.cng.Compute(e.aecState.SaturatedCapture(), Y2, comfortNoise, highBandComfortNoise)

	// The rest runs only when the output is used.
	G := &e.G
	if e.captureOutputUsed {
		e.residualEchoEstimator.Estimate(&e.aecState, renderBuffer, S2Linear, Y2,
			e.suppressionGain.IsDominantNearend(), R2, R2Unbounded)

		// The nearend estimate of the suppressor.
		if e.aecState.UsableLinearEstimate() {
			// E2 is bound by Y2.
			for ch := 0; ch < nch; ch++ {
				kernStdMin(E2[ch][:], Y2[ch][:], E2[ch][:])
			}
		}
		nearendSpectrum := Y2
		if e.aecState.UsableLinearEstimate() {
			nearendSpectrum = E2
		}

		// The echo estimate of the suppressor.
		echoSpectrum := R2
		if e.aecState.UsableLinearEstimate() {
			echoSpectrum = S2Linear
		}

		clockDrift := e.config.EchoRemovalControl.HasClockDrift || echoPathVariability.ClockDrift

		var highBandsGain float32
		e.suppressionGain.GetGain(nearendSpectrum, echoSpectrum, R2, R2Unbounded, e.cng.NoiseSpectrum(),
			&e.renderSignalAnalyzer, &e.aecState, &x, clockDrift, &highBandsGain, G)

		e.suppressionFilter.ApplyGain(comfortNoise, highBandComfortNoise, G, highBandsGain, YFft, y)
	} else {
		*G = [FftLengthBy2Plus1]float32{}
	}

	e.metrics.Update(&e.aecState, &e.cng.NoiseSpectrum()[0], G)
}

// UpdateEchoLeakageStatus tells the remover whether echo leaks into its
// output.
func (e *EchoRemover) UpdateEchoLeakageStatus(leakageDetected bool) {
	e.echoLeakageDetected = leakageDetected
}

// SetCaptureOutputUsage tells the remover whether its output is used. An
// unused output skips the residual echo estimation and the suppression.
func (e *EchoRemover) SetCaptureOutputUsage(captureOutputUsed bool) {
	e.captureOutputUsed = captureOutputUsed
}

// formLinearFilterOutput selects the coarse or the refined filter output for
// the suppressor and fades to it from the one selected before.
func (e *EchoRemover) formLinearFilterOutput(so *SubtractorOutput, output *[FftLengthBy2]float32) {
	useRefinedOutput := true
	if e.useCoarseFilterOutput {
		// The refined output is the better one as a rule: the coarse output
		// needs a margin to be chosen.
		if so.E2CoarseSum < float32(0.9*so.E2RefinedSum) && so.Y2 > 30*30*BlockSize &&
			(so.S2Refined > 60*60*BlockSize || so.S2Coarse > 60*60*BlockSize) {
			useRefinedOutput = false
		} else if so.E2CoarseSum < so.E2RefinedSum && so.Y2 < so.E2RefinedSum {
			// A diverged refined filter: the output of lower power.
			useRefinedOutput = false
		}
	}

	from := &so.ECoarse
	if e.refinedFilterOutputLastSelected {
		from = &so.ERefined
	}
	to := &so.ECoarse
	if useRefinedOutput {
		to = &so.ERefined
	}
	erSignalTransition(from, to, output)
	e.refinedFilterOutputLastSelected = useRefinedOutput
}

// erLinearEchoPower sets S2 to the power of Y - E.
func erLinearEchoPower(E, Y *FftData, S2 *[FftLengthBy2Plus1]float32) {
	kernDiffPower(Y.Re[:], E.Re[:], Y.Im[:], E.Im[:], S2[:])
}

// erSignalTransition fades from one signal to another over a fixed
// transition. It compares from and to by identity, as ArrayView equality
// does.
func erSignalTransition(from, to *[BlockSize]float32, out *[FftLengthBy2]float32) {
	if from == to {
		*out = *to
		return
	}
	const transitionSize = 30
	const oneByTransitionSizePlusOne = float32(1) / (transitionSize + 1)
	for k := 0; k < transitionSize; k++ {
		a := float32(float32(k+1) * oneByTransitionSizePlusOne)
		out[k] = float32(a*to[k]) + float32((1-a)*from[k])
	}
	copy(out[transitionSize:], to[transitionSize:])
}
