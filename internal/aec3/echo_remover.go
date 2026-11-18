package aec3

// Port of echo_remover.{h,cc} (the EchoRemover interface and EchoRemoverImpl).

// EchoControlMetrics mirrors EchoControl::Metrics (api/audio/echo_control.h).
type EchoControlMetrics struct {
	EchoReturnLoss            float64
	EchoReturnLossEnhancement float64
	DelayMs                   int
}

// EchoRemover removes the echo from the capture signal.
type EchoRemover struct {
	config                          *EchoCanceller3Config // shared, never modified
	fft                             Aec3Fft
	optimization                    Aec3Optimization
	sampleRateHz                    int
	numRenderChannels               int
	numCaptureChannels              int
	useCoarseFilterOutput           bool
	subtractor                      *Subtractor
	suppressionGain                 *SuppressionGain
	cng                             *ComfortNoiseGenerator
	suppressionFilter               *SuppressionFilter
	renderSignalAnalyzer            *RenderSignalAnalyzer
	residualEchoEstimator           *ResidualEchoEstimator
	echoLeakageDetected             bool
	captureOutputUsed               bool
	aecState                        *AecState
	metrics                         *EchoRemoverMetrics
	eOld                            [][FftLengthBy2]float32
	yOld                            [][FftLengthBy2]float32
	blockCounter                    int
	gainChangeHangover              int
	refinedFilterOutputLastSelected bool

	// Per-block scratch (the C++ stack/heap channel data).
	e                    [][FftLengthBy2]float32
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
	G                    [FftLengthBy2Plus1]float32
}

// NewEchoRemover mirrors EchoRemover::Create; the optimization is
// DetectOptimization().
func NewEchoRemover(config *EchoCanceller3Config, sampleRateHz, numRenderChannels, numCaptureChannels int) *EchoRemover {
	return NewEchoRemoverWithOptimization(config, sampleRateHz, numRenderChannels, numCaptureChannels,
		DetectOptimization())
}

// NewEchoRemoverWithOptimization creates an EchoRemover that runs the given
// optimization lane instead of DetectOptimization().
// The result keeps a reference to config, which must not change while it
// is in use (the C++ copies it).
func NewEchoRemoverWithOptimization(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, optimization Aec3Optimization) *EchoRemover {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate")
	}
	e := &EchoRemover{
		config:                          config,
		optimization:                    optimization,
		sampleRateHz:                    sampleRateHz,
		numRenderChannels:               numRenderChannels,
		numCaptureChannels:              numCaptureChannels,
		captureOutputUsed:               true,
		refinedFilterOutputLastSelected: true,
		metrics:                         NewEchoRemoverMetrics(),
		eOld:                            make([][FftLengthBy2]float32, numCaptureChannels),
		yOld:                            make([][FftLengthBy2]float32, numCaptureChannels),
		e:                               make([][FftLengthBy2]float32, numCaptureChannels),
		Y2:                              make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		E2:                              make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		R2:                              make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		R2Unbounded:                     make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		S2Linear:                        make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		Y:                               make([]FftData, numCaptureChannels),
		E:                               make([]FftData, numCaptureChannels),
		comfortNoise:                    make([]FftData, numCaptureChannels),
		highBandComfortNoise:            make([]FftData, numCaptureChannels),
		subtractorOutput:                make([]SubtractorOutput, numCaptureChannels),
	}
	cfg := e.config
	e.useCoarseFilterOutput = cfg.Filter.EnableCoarseFilterOutputUsage
	e.subtractor = NewSubtractor(cfg, numRenderChannels, numCaptureChannels, optimization)
	e.suppressionGain = NewSuppressionGain(cfg, optimization, sampleRateHz, numCaptureChannels)
	e.cng = NewComfortNoiseGenerator(cfg, optimization, numCaptureChannels)
	e.suppressionFilter = NewSuppressionFilter(optimization, sampleRateHz, numCaptureChannels)
	e.renderSignalAnalyzer = NewRenderSignalAnalyzer(cfg)
	e.residualEchoEstimator = NewResidualEchoEstimator(cfg, numRenderChannels)
	e.aecState = NewAecState(cfg, numCaptureChannels)
	return e
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
	e.subtractor.Process(renderBuffer, y, e.renderSignalAnalyzer, e.aecState, e.subtractorOutput)

	// Compute spectra.
	for ch := 0; ch < nch; ch++ {
		e.formLinearFilterOutput(&e.subtractorOutput[ch], &e.e[ch])
		erWindowedPaddedFft(&e.fft, y.View(0, ch), e.yOld[ch][:], &e.Y[ch])
		erWindowedPaddedFft(&e.fft, e.e[ch][:], e.eOld[ch][:], &e.E[ch])
		erLinearEchoPower(&e.E[ch], &e.Y[ch], &e.S2Linear[ch])
		e.Y[ch].Spectrum(e.optimization, e.Y2[ch][:])
		e.E[ch].Spectrum(e.optimization, e.E2[ch][:])
	}

	// Optionally return the linear filter output.
	if linearOutput != nil {
		for ch := 0; ch < nch; ch++ {
			copy(linearOutput.View(0, ch), e.e[ch][:])
		}
	}

	// Update the AEC state information.
	e.aecState.Update(externalDelay, e.subtractor.FilterFrequencyResponses(), e.subtractor.FilterImpulseResponses(),
		renderBuffer, e.E2, e.Y2, e.subtractorOutput)

	// Choose the linear output.
	YFft := e.Y
	if e.aecState.UseLinearFilterOutput() {
		YFft = e.E
	}

	// Estimate the comfort noise.
	e.cng.Compute(e.aecState.SaturatedCapture(), e.Y2, e.comfortNoise, e.highBandComfortNoise)

	// Only do the below processing if the output of the audio processing
	// module is used.
	G := &e.G
	if e.captureOutputUsed {
		// Estimate the residual echo power.
		e.residualEchoEstimator.Estimate(e.aecState, renderBuffer, e.S2Linear, e.Y2,
			e.suppressionGain.IsDominantNearend(), e.R2, e.R2Unbounded)

		// Suppressor nearend estimate.
		if e.aecState.UsableLinearEstimate() {
			// E2 is bound by Y2.
			for ch := 0; ch < nch; ch++ {
				kernStdMin(e.E2[ch][:], e.Y2[ch][:], e.E2[ch][:])
			}
		}
		nearendSpectrum := e.Y2
		if e.aecState.UsableLinearEstimate() {
			nearendSpectrum = e.E2
		}

		// Suppressor echo estimate.
		echoSpectrum := e.R2
		if e.aecState.UsableLinearEstimate() {
			echoSpectrum = e.S2Linear
		}

		// Determine if the suppressor should assume clock drift.
		clockDrift := e.config.EchoRemovalControl.HasClockDrift || echoPathVariability.ClockDrift

		// Compute preferred gains.
		var highBandsGain float32
		e.suppressionGain.GetGain(nearendSpectrum, echoSpectrum, e.R2, e.R2Unbounded, e.cng.NoiseSpectrum(),
			e.renderSignalAnalyzer, e.aecState, &x, clockDrift, &highBandsGain, G)

		e.suppressionFilter.ApplyGain(e.comfortNoise, e.highBandComfortNoise, G, highBandsGain, YFft, y)
	} else {
		*G = [FftLengthBy2Plus1]float32{}
	}

	// Update the metrics.
	e.metrics.Update(e.aecState, &e.cng.NoiseSpectrum()[0], G)
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

// erWindowedPaddedFft computes a windowed (square root Hanning) padded FFT
// and updates the related memory.
func erWindowedPaddedFft(fft *Aec3Fft, v, vOld []float32, V *FftData) {
	fft.PaddedFft(v, vOld, WindowSqrtHanning, V)
	copy(vOld, v)
}
