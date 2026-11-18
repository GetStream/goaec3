package aec3

const stDefaultTransparentModeGain float32 = 0.01

type stReverbType int

const (
	stReverbLinear stReverbType = iota
	stReverbNonLinear
)

// ResidualEchoEstimator mirrors residual_echo_estimator.h: estimates the
// residual echo power. Field trials are at their defaults.
type ResidualEchoEstimator struct {
	config                                 EchoCanceller3Config
	numRenderChannels                      int
	earlyReflectionsTransparentModeGain    float32
	lateReflectionsTransparentModeGain     float32
	earlyReflectionsGeneralGain            float32
	lateReflectionsGeneralGain             float32
	erleOnsetCompensationInDominantNearend bool
	X2NoiseFloor                           [FftLengthBy2Plus1]float32
	X2NoiseFloorCounter                    [FftLengthBy2Plus1]int
	echoReverb                             ReverbModel
}

func NewResidualEchoEstimator(config *EchoCanceller3Config, numRenderChannels int) *ResidualEchoEstimator {
	r := &ResidualEchoEstimator{
		config:                                 *config,
		numRenderChannels:                      numRenderChannels,
		earlyReflectionsTransparentModeGain:    stDefaultTransparentModeGain,
		lateReflectionsTransparentModeGain:     stDefaultTransparentModeGain,
		earlyReflectionsGeneralGain:            config.EpStrength.DefaultGain,
		lateReflectionsGeneralGain:             config.EpStrength.DefaultGain,
		erleOnsetCompensationInDominantNearend: config.EpStrength.ErleOnsetCompensationInDominantNearend,
	}
	r.reset()
	return r
}

func (r *ResidualEchoEstimator) reset() {
	r.echoReverb.Reset()
	for k := range r.X2NoiseFloorCounter {
		r.X2NoiseFloorCounter[k] = r.config.EchoModel.NoiseFloorHold
		r.X2NoiseFloor[k] = r.config.EchoModel.MinNoiseFloorPower
	}
}

func stGetRenderIndexesToAnalyze(spectrumBuffer *SpectrumBuffer, renderPreWindowSize, renderPostWindowSize,
	filterDelayBlocks int) (idxStart, idxStop int) {
	windowStart := max(0, filterDelayBlocks-renderPreWindowSize)
	windowEnd := filterDelayBlocks + renderPostWindowSize
	idxStart = spectrumBuffer.OffsetIndex(spectrumBuffer.Read, windowStart)
	idxStop = spectrumBuffer.OffsetIndex(spectrumBuffer.Read, windowEnd+1)
	return
}

func stLinearEstimate(S2Linear, erle, R2 [][FftLengthBy2Plus1]float32) {
	for ch := range R2 {
		kernDiv(S2Linear[ch][:], erle[ch][:], R2[ch][:])
	}
}

func stNonLinearEstimate(echoPathGain float32, X2 *[FftLengthBy2Plus1]float32, R2 [][FftLengthBy2Plus1]float32) {
	for ch := range R2 {
		for k := 0; k < FftLengthBy2Plus1; k++ {
			R2[ch][k] = X2[k] * echoPathGain
		}
	}
}

// Estimate estimates the residual echo power R2 and its unbounded variant.
func (r *ResidualEchoEstimator) Estimate(aecState *AecState, renderBuffer *RenderBuffer, S2Linear, Y2 [][FftLengthBy2Plus1]float32,
	dominantNearend bool, R2, R2Unbounded [][FftLengthBy2Plus1]float32) {
	numCaptureChannels := len(R2)
	r.updateRenderNoisePower(renderBuffer)

	if aecState.UsableLinearEstimate() {
		if aecState.SaturatedEcho() {
			for ch := 0; ch < numCaptureChannels; ch++ {
				R2[ch] = Y2[ch]
				R2Unbounded[ch] = Y2[ch]
			}
		} else {
			onsetCompensated := r.erleOnsetCompensationInDominantNearend || !dominantNearend
			stLinearEstimate(S2Linear, aecState.Erle(onsetCompensated), R2)
			stLinearEstimate(S2Linear, aecState.ErleUnbounded(), R2Unbounded)
		}
		r.updateReverb(stReverbLinear, aecState, renderBuffer, dominantNearend)
		r.addReverb(R2)
		r.addReverb(R2Unbounded)
	} else {
		echoPathGain := r.getEchoPathGain(aecState, true)
		if aecState.SaturatedEcho() {
			for ch := 0; ch < numCaptureChannels; ch++ {
				R2[ch] = Y2[ch]
				R2Unbounded[ch] = Y2[ch]
			}
		} else {
			var X2 [FftLengthBy2Plus1]float32
			r.echoGeneratingPower(renderBuffer.GetSpectrumBuffer(), aecState.MinDirectPathFilterDelay(), &X2)
			if !aecState.UseStationarityProperties() {
				em := &r.config.EchoModel
				for k := 0; k < FftLengthBy2Plus1; k++ {
					if em.NoiseGatePower > X2[k] {
						X2[k] = stMaxf(0, X2[k]-float32(em.NoiseGateSlope*(em.NoiseGatePower-X2[k])))
					}
				}
			}
			for k := 0; k < FftLengthBy2Plus1; k++ {
				X2[k] -= float32(r.config.EchoModel.StationaryGateSlope * r.X2NoiseFloor[k])
				X2[k] = stMaxf(0, X2[k])
			}
			stNonLinearEstimate(echoPathGain, &X2, R2)
			stNonLinearEstimate(echoPathGain, &X2, R2Unbounded)
		}
		if r.config.EchoModel.ModelReverbInNonlinearMode && !aecState.TransparentModeActive() {
			r.updateReverb(stReverbNonLinear, aecState, renderBuffer, dominantNearend)
			r.addReverb(R2)
			r.addReverb(R2Unbounded)
		}
	}

	if aecState.UseStationarityProperties() {
		var residualScaling [FftLengthBy2Plus1]float32
		aecState.GetResidualEchoScaling(residualScaling[:])
		for ch := 0; ch < numCaptureChannels; ch++ {
			vectorMultiply(R2[ch][:], residualScaling[:], R2[ch][:])
			vectorMultiply(R2Unbounded[ch][:], residualScaling[:], R2Unbounded[ch][:])
		}
	}
}

func (r *ResidualEchoEstimator) echoGeneratingPower(spectrumBuffer *SpectrumBuffer, filterDelayBlocks int, X2 *[FftLengthBy2Plus1]float32) {
	idxStart, idxStop := stGetRenderIndexesToAnalyze(spectrumBuffer, r.config.EchoModel.RenderPreWindowSize,
		r.config.EchoModel.RenderPostWindowSize, filterDelayBlocks)
	*X2 = [FftLengthBy2Plus1]float32{}
	if r.numRenderChannels == 1 {
		for k := idxStart; k != idxStop; k = spectrumBuffer.IncIndex(k) {
			kernStdMax(X2[:], spectrumBuffer.Buffer[k][0][:], X2[:])
		}
	} else {
		for k := idxStart; k != idxStop; k = spectrumBuffer.IncIndex(k) {
			var renderPower [FftLengthBy2Plus1]float32
			for ch := 0; ch < r.numRenderChannels; ch++ {
				channelPower := &spectrumBuffer.Buffer[k][ch]
				for j := 0; j < FftLengthBy2Plus1; j++ {
					renderPower[j] += channelPower[j]
				}
			}
			kernStdMax(X2[:], renderPower[:], X2[:])
		}
	}
}

func (r *ResidualEchoEstimator) renderPowerAt(renderBuffer *RenderBuffer, offset int, data *[FftLengthBy2Plus1]float32) *[FftLengthBy2Plus1]float32 {
	X2 := renderBuffer.Spectrum(offset)
	if r.numRenderChannels > 1 {
		*data = [FftLengthBy2Plus1]float32{}
		for ch := 0; ch < r.numRenderChannels; ch++ {
			channelPower := &X2[ch]
			for k := 0; k < FftLengthBy2Plus1; k++ {
				data[k] += channelPower[k]
			}
		}
		return data
	}
	return &X2[0]
}

func (r *ResidualEchoEstimator) updateRenderNoisePower(renderBuffer *RenderBuffer) {
	var renderPowerData [FftLengthBy2Plus1]float32
	renderPower := r.renderPowerAt(renderBuffer, 0, &renderPowerData)
	for k := 0; k < FftLengthBy2Plus1; k++ {
		if renderPower[k] < r.X2NoiseFloor[k] {
			r.X2NoiseFloor[k] = renderPower[k]
			r.X2NoiseFloorCounter[k] = 0
		} else {
			if r.X2NoiseFloorCounter[k] >= r.config.EchoModel.NoiseFloorHold {
				r.X2NoiseFloor[k] = stMaxf(r.X2NoiseFloor[k]*1.1, r.config.EchoModel.MinNoiseFloorPower)
			} else {
				r.X2NoiseFloorCounter[k]++
			}
		}
	}
}

func (r *ResidualEchoEstimator) updateReverb(reverbType stReverbType, aecState *AecState, renderBuffer *RenderBuffer, dominantNearend bool) {
	var firstReverbPartition int
	if reverbType == stReverbLinear {
		firstReverbPartition = aecState.FilterLengthBlocks() + 1
	} else {
		firstReverbPartition = aecState.MinDirectPathFilterDelay() + 1
	}
	var renderPowerData [FftLengthBy2Plus1]float32
	renderPower := r.renderPowerAt(renderBuffer, firstReverbPartition, &renderPowerData)
	reverbDecay := aecState.ReverbDecay(dominantNearend)
	if reverbType == stReverbLinear {
		r.echoReverb.UpdateReverb(renderPower[:], aecState.GetReverbFrequencyResponse()[:], reverbDecay)
	} else {
		echoPathGain := r.getEchoPathGain(aecState, false)
		r.echoReverb.UpdateReverbNoFreqShaping(renderPower[:], echoPathGain, reverbDecay)
	}
}

func (r *ResidualEchoEstimator) addReverb(R2 [][FftLengthBy2Plus1]float32) {
	reverbPower := r.echoReverb.Reverb()
	for ch := range R2 {
		vectorAccumulate(reverbPower[:], R2[ch][:])
	}
}

func (r *ResidualEchoEstimator) getEchoPathGain(aecState *AecState, gainForEarlyReflections bool) float32 {
	var gainAmplitude float32
	if aecState.TransparentModeActive() {
		if gainForEarlyReflections {
			gainAmplitude = r.earlyReflectionsTransparentModeGain
		} else {
			gainAmplitude = r.lateReflectionsTransparentModeGain
		}
	} else {
		if gainForEarlyReflections {
			gainAmplitude = r.earlyReflectionsGeneralGain
		} else {
			gainAmplitude = r.lateReflectionsGeneralGain
		}
	}
	return gainAmplitude * gainAmplitude
}
