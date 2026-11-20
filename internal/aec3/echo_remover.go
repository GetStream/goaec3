package aec3

// Port of echo_remover.{h,cc} (the EchoRemover interface and EchoRemoverImpl).

// EchoControlMetrics mirrors EchoControl::Metrics (api/audio/echo_control.h).
type EchoControlMetrics struct {
	EchoReturnLoss            float64
	EchoReturnLossEnhancement float64
	DelayMs                   int
}

// EchoRemover removes the echo from the capture signal.
//
// Its components are part of the EchoRemover, as is the per-channel state of
// up to inlineChannels capture channels. The components that hold pointers
// come first.
type EchoRemover struct {
	*echoRemoverState
	more       *erMore // with more than inlineChannels channels
	subtractor Subtractor
}

// echoRemoverState is the state of an EchoRemover but its pointers.
type echoRemoverState struct {
	_                     noCopy
	aecState              AecState
	subtractor            subtractorState
	suppressionGain       SuppressionGain
	cng                   ComfortNoiseGenerator
	suppressionFilter     SuppressionFilter
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

// erChannels is the per-channel state of an EchoRemover with up to
// inlineChannels capture channels; all but channels is per-block scratch
// (the C++ stack/heap channel data).
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

// erChannel is the state of one capture channel that only the EchoRemover
// reads: the previous block of the capture signal and of the linear filter
// output, and the current linear filter output.
type erChannel struct {
	eOld [FftLengthBy2]float32
	yOld [FftLengthBy2]float32
	e    [FftLengthBy2]float32
}

// bind makes e the view of st.
func (e *EchoRemover) bind(st *echoRemoverState, b *binder) {
	e.echoRemoverState = st
	if b != nil {
		bindPtr(b, &e.more, &b.x.erMore)
	}
	e.subtractor.bind(&st.subtractor, b)
}

// NewEchoRemover mirrors EchoRemover::Create; the optimization is
// DetectOptimization().
func NewEchoRemover(config *EchoCanceller3Config, sampleRateHz, numRenderChannels, numCaptureChannels int) *EchoRemover {
	return NewEchoRemoverWithOptimization(config, sampleRateHz, numRenderChannels, numCaptureChannels,
		DetectOptimization())
}

// NewEchoRemoverWithOptimization creates an EchoRemover that runs the given
// optimization lane instead of DetectOptimization().
func NewEchoRemoverWithOptimization(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, optimization Aec3Optimization) *EchoRemover {
	e := &EchoRemover{}
	e.bind(&echoRemoverState{}, nil)
	e.init(config, sampleRateHz, numRenderChannels, numCaptureChannels, optimization)
	return e
}

// init sets up e in place.
func (e *EchoRemover) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, optimization Aec3Optimization) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate")
	}
	st := e.echoRemoverState
	*st = echoRemoverState{}
	*e = EchoRemover{}
	e.bind(st, nil)
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

// GetMetrics returns the current metrics. DelayMs is left at zero; the
// EchoRemover does not set it.
func (e *EchoRemover) GetMetrics() EchoControlMetrics {
	// Echo return loss (ERL) is inverted to go from gain to attenuation.
	return EchoControlMetrics{
		EchoReturnLoss:            -10.0 * float64(Log10f(e.aecState.ErlTimeDomain())),
		EchoReturnLossEnhancement: float64(Log2TodB(e.aecState.FullBandErleLog2())),
	}
}

// ProcessCapture removes the echo from one capture block. externalDelay is nil
// when absent; linearOutput is nil when the linear AEC output is not exported.
// The render signal is assumed to be pre-aligned with the capture signal.
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
		// Ensure that the gain change is only acted on once per frame.
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

	// Analyze the render signal.
	e.renderSignalAnalyzer.Update(renderBuffer, e.aecState.MinDirectPathFilterDelay(), true)

	// State transition.
	if e.aecState.TransitionTriggered() {
		e.subtractor.ExitInitialState()
		e.suppressionGain.SetInitialState(false)
	}

	// Perform linear echo cancellation.
	e.subtractor.Process(renderBuffer, y, &e.renderSignalAnalyzer, &e.aecState, subtractorOutput)

	// Compute spectra.
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

	// Optionally return the linear filter output.
	if linearOutput != nil {
		for ch := 0; ch < nch; ch++ {
			copy(linearOutput.View(0, ch), channels[ch].e[:])
		}
	}

	// Update the AEC state information.
	e.aecState.Update(externalDelay, e.subtractor.FilterFrequencyResponses(), e.subtractor.FilterImpulseResponses(),
		renderBuffer, E2, Y2, subtractorOutput)

	// Choose the linear output.
	YFft := Y
	if e.aecState.UseLinearFilterOutput() {
		YFft = E
	}

	// Estimate the comfort noise.
	e.cng.Compute(e.aecState.SaturatedCapture(), Y2, comfortNoise, highBandComfortNoise)

	// Only do the below processing if the output of the audio processing
	// module is used.
	G := &e.G
	if e.captureOutputUsed {
		// Estimate the residual echo power.
		e.residualEchoEstimator.Estimate(&e.aecState, renderBuffer, S2Linear, Y2,
			e.suppressionGain.IsDominantNearend(), R2, R2Unbounded)

		// Suppressor nearend estimate.
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

		// Suppressor echo estimate.
		echoSpectrum := R2
		if e.aecState.UsableLinearEstimate() {
			echoSpectrum = S2Linear
		}

		// Determine if the suppressor should assume clock drift.
		clockDrift := e.config.EchoRemovalControl.HasClockDrift || echoPathVariability.ClockDrift

		// Compute preferred gains.
		var highBandsGain float32
		e.suppressionGain.GetGain(nearendSpectrum, echoSpectrum, R2, R2Unbounded, e.cng.NoiseSpectrum(),
			&e.renderSignalAnalyzer, &e.aecState, &x, clockDrift, &highBandsGain, G)

		e.suppressionFilter.ApplyGain(comfortNoise, highBandComfortNoise, G, highBandsGain, YFft, y)
	} else {
		*G = [FftLengthBy2Plus1]float32{}
	}

	// Update the metrics.
	e.metrics.Update(&e.aecState, &e.cng.NoiseSpectrum()[0], G)
}

// UpdateEchoLeakageStatus reports whether echo leakage is detected in the
// output of the echo remover.
func (e *EchoRemover) UpdateEchoLeakageStatus(leakageDetected bool) {
	e.echoLeakageDetected = leakageDetected
}

// SetCaptureOutputUsage tells the remover whether its output is used. When it
// is not, the residual echo estimation and the suppression are skipped.
func (e *EchoRemover) SetCaptureOutputUsage(captureOutputUsed bool) {
	e.captureOutputUsed = captureOutputUsed
}

// formLinearFilterOutput selects which of the coarse and refined linear filter
// outputs is most appropriate to pass to the suppressor and forms the linear
// filter output by smoothly transitioning between those.
func (e *EchoRemover) formLinearFilterOutput(so *SubtractorOutput, output *[FftLengthBy2]float32) {
	useRefinedOutput := true
	if e.useCoarseFilterOutput {
		// As the output of the refined adaptive filter generally should be
		// better than the coarse filter output, add a margin and threshold for
		// when choosing the coarse filter output.
		if so.E2CoarseSum < float32(0.9*so.E2RefinedSum) && so.Y2 > 30*30*BlockSize &&
			(so.S2Refined > 60*60*BlockSize || so.S2Coarse > 60*60*BlockSize) {
			useRefinedOutput = false
		} else if so.E2CoarseSum < so.E2RefinedSum && so.Y2 < so.E2RefinedSum {
			// If the refined filter is diverged, choose the filter output that
			// has the lowest power.
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

// erLinearEchoPower computes the power of the difference between Y and E.
func erLinearEchoPower(E, Y *FftData, S2 *[FftLengthBy2Plus1]float32) {
	kernDiffPower(Y.Re[:], E.Re[:], Y.Im[:], E.Im[:], S2[:])
}

// erSignalTransition fades between two input signals using a fixed-size
// transition. As in the C++ (ArrayView equality), from and to are compared by
// identity, not by content.
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
