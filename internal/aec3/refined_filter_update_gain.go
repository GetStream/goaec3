package aec3

// Port of refined_filter_update_gain.{h,cc}.

const (
	afirHErrorInitial                float32 = 10000
	afirPoorExcitationCounterInitial         = 1000
)

// RefinedFilterUpdateGain computes the NLMS-style gain for the refined filter.
type RefinedFilterUpdateGain struct {
	configChangeDurationBlocks      int
	oneByConfigChangeDurationBlocks float32
	currentConfig                   RefinedConfiguration
	targetConfig                    RefinedConfiguration
	oldTargetConfig                 RefinedConfiguration
	hError                          [FftLengthBy2Plus1]float32
	poorExcitationCounter           int
	callCounter                     int
	configChangeCounter             int
}

func NewRefinedFilterUpdateGain(config RefinedConfiguration, configChangeDurationBlocks int) *RefinedFilterUpdateGain {
	g := &RefinedFilterUpdateGain{}
	g.init(config, configChangeDurationBlocks)
	return g
}

// init sets up g in place.
func (g *RefinedFilterUpdateGain) init(config RefinedConfiguration, configChangeDurationBlocks int) {
	*g = RefinedFilterUpdateGain{
		configChangeDurationBlocks: configChangeDurationBlocks,
		poorExcitationCounter:      afirPoorExcitationCounterInitial,
	}
	g.SetConfig(config, true)
	for k := range g.hError {
		g.hError[k] = afirHErrorInitial
	}
	g.oneByConfigChangeDurationBlocks = 1 / float32(g.configChangeDurationBlocks)
}

// HandleEchoPathChange reacts to a known echo path change.
func (g *RefinedFilterUpdateGain) HandleEchoPathChange(echoPathVariability EchoPathVariability) {
	if echoPathVariability.DelayChange != DelayAdjustmentNone {
		for k := range g.hError {
			g.hError[k] = afirHErrorInitial
		}
	}
	if !echoPathVariability.GainChange {
		g.poorExcitationCounter = afirPoorExcitationCounterInitial
		g.callCounter = 0
	}
}

// SetConfig sets a new config, immediately or gradually.
func (g *RefinedFilterUpdateGain) SetConfig(config RefinedConfiguration, immediateEffect bool) {
	if immediateEffect {
		g.oldTargetConfig, g.currentConfig, g.targetConfig = config, config, config
		g.configChangeCounter = 0
	} else {
		g.oldTargetConfig = g.currentConfig
		g.targetConfig = config
		g.configChangeCounter = g.configChangeDurationBlocks
	}
}

// Compute computes the gain gainFft. erl must hold FftLengthBy2Plus1 values.
func (g *RefinedFilterUpdateGain) Compute(renderPower *[FftLengthBy2Plus1]float32, renderSignalAnalyzer *RenderSignalAnalyzer, subtractorOutput *SubtractorOutput, erl []float32, sizePartitions int, saturatedCaptureSignal, disallowLeakageDiverged bool, gainFft *FftData) {
	ERefined := &subtractorOutput.ERefinedFft
	E2Refined := &subtractorOutput.E2Refined
	E2Coarse := &subtractorOutput.E2Coarse
	G := gainFft
	X2 := renderPower
	erl = erl[:FftLengthBy2Plus1]

	g.callCounter++
	g.updateCurrentConfig()

	if renderSignalAnalyzer.PoorSignalExcitation() {
		g.poorExcitationCounter = 0
	}

	// Do not update the filter if the render is not sufficiently excited.
	g.poorExcitationCounter++
	if g.poorExcitationCounter < sizePartitions || saturatedCaptureSignal || g.callCounter <= sizePartitions {
		G.Re = [FftLengthBy2Plus1]float32{}
		G.Im = [FftLengthBy2Plus1]float32{}
	} else {
		// mu = H_error / (0.5* H_error* X2 + n * E2).
		var mu [FftLengthBy2Plus1]float32
		kernRefinedMu(X2, &g.hError, E2Refined, g.currentConfig.NoiseGate, float32(sizePartitions), &mu)

		// Avoid updating the filter close to narrow bands in the render signals.
		renderSignalAnalyzer.MaskRegionsAroundNarrowBands(&mu)

		// H_error = H_error - 0.5 * mu * X2 * H_error.
		kernRefinedHError(&mu, X2, &g.hError)

		// G = mu * E.
		vectorMultiply(mu[:], ERefined.Re[:], G.Re[:])
		vectorMultiply(mu[:], ERefined.Im[:], G.Im[:])
	}

	// H_error = H_error + factor * erl.
	c := &g.currentConfig
	kernRefinedLeak(E2Refined, E2Coarse, erl, disallowLeakageDiverged, c.LeakageConverged, c.LeakageDiverged,
		c.ErrorFloor, c.ErrorCeil, &g.hError)
}

func (g *RefinedFilterUpdateGain) updateCurrentConfig() {
	if g.configChangeCounter > 0 {
		g.configChangeCounter--
		if g.configChangeCounter > 0 {
			changeFactor := float32(float32(g.configChangeCounter) * g.oneByConfigChangeDurationBlocks)
			c, o, t := &g.currentConfig, &g.oldTargetConfig, &g.targetConfig
			c.LeakageConverged = afirAverage(o.LeakageConverged, t.LeakageConverged, changeFactor)
			c.LeakageDiverged = afirAverage(o.LeakageDiverged, t.LeakageDiverged, changeFactor)
			c.ErrorFloor = afirAverage(o.ErrorFloor, t.ErrorFloor, changeFactor)
			c.ErrorCeil = afirAverage(o.ErrorCeil, t.ErrorCeil, changeFactor)
			c.NoiseGate = afirAverage(o.NoiseGate, t.NoiseGate, changeFactor)
		} else {
			g.currentConfig = g.targetConfig
			g.oldTargetConfig = g.targetConfig
		}
	}
}
