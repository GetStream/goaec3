package aec3

// Port of coarse_filter_update_gain.{h,cc}.

// CoarseFilterUpdateGain computes the fixed-step gain of the coarse filter.
type CoarseFilterUpdateGain struct {
	currentConfig                   CoarseConfiguration
	targetConfig                    CoarseConfiguration
	oldTargetConfig                 CoarseConfiguration
	configChangeDurationBlocks      int
	oneByConfigChangeDurationBlocks float32
	poorSignalExcitationCounter     int
	callCounter                     int
	configChangeCounter             int
}

func NewCoarseFilterUpdateGain(config CoarseConfiguration, configChangeDurationBlocks int) *CoarseFilterUpdateGain {
	g := &CoarseFilterUpdateGain{}
	g.init(config, configChangeDurationBlocks)
	return g
}

func (g *CoarseFilterUpdateGain) init(config CoarseConfiguration, configChangeDurationBlocks int) {
	*g = CoarseFilterUpdateGain{configChangeDurationBlocks: configChangeDurationBlocks}
	g.SetConfig(config, true)
	g.oneByConfigChangeDurationBlocks = 1 / float32(g.configChangeDurationBlocks)
}

// HandleEchoPathChange resets the adaptation counters.
func (g *CoarseFilterUpdateGain) HandleEchoPathChange() {
	g.poorSignalExcitationCounter = 0
	g.callCounter = 0
}

// SetConfig sets the config, immediately or gradually.
func (g *CoarseFilterUpdateGain) SetConfig(config CoarseConfiguration, immediateEffect bool) {
	if immediateEffect {
		g.oldTargetConfig, g.currentConfig, g.targetConfig = config, config, config
		g.configChangeCounter = 0
	} else {
		g.oldTargetConfig = g.currentConfig
		g.targetConfig = config
		g.configChangeCounter = g.configChangeDurationBlocks
	}
}

// Compute computes the gain G.
func (g *CoarseFilterUpdateGain) Compute(renderPower *[FftLengthBy2Plus1]float32, renderSignalAnalyzer *RenderSignalAnalyzer, ECoarse *FftData, sizePartitions int, saturatedCaptureSignal bool, G *FftData) {
	g.callCounter++
	g.updateCurrentConfig()

	if renderSignalAnalyzer.PoorSignalExcitation() {
		g.poorSignalExcitationCounter = 0
	}

	// No update while the render lacks excitation.
	g.poorSignalExcitationCounter++
	if g.poorSignalExcitationCounter < sizePartitions || saturatedCaptureSignal || g.callCounter <= sizePartitions {
		G.Re = [FftLengthBy2Plus1]float32{}
		G.Im = [FftLengthBy2Plus1]float32{}
		return
	}

	var mu [FftLengthBy2Plus1]float32
	X2 := renderPower
	kernCoarseMu(X2, g.currentConfig.NoiseGate, g.currentConfig.Rate, &mu)

	// No update close to narrow bands of the render signal.
	renderSignalAnalyzer.MaskRegionsAroundNarrowBands(&mu)

	// G = mu * E.
	vectorMultiply(mu[:], ECoarse.Re[:], G.Re[:])
	vectorMultiply(mu[:], ECoarse.Im[:], G.Im[:])
}

func (g *CoarseFilterUpdateGain) updateCurrentConfig() {
	if g.configChangeCounter > 0 {
		g.configChangeCounter--
		if g.configChangeCounter > 0 {
			changeFactor := float32(float32(g.configChangeCounter) * g.oneByConfigChangeDurationBlocks)
			g.currentConfig.Rate = afirAverage(g.oldTargetConfig.Rate, g.targetConfig.Rate, changeFactor)
			g.currentConfig.NoiseGate = afirAverage(g.oldTargetConfig.NoiseGate, g.targetConfig.NoiseGate, changeFactor)
		} else {
			g.currentConfig = g.targetConfig
			g.oldTargetConfig = g.targetConfig
		}
	}
}
