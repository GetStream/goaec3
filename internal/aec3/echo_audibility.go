package aec3

// EchoAudibility mirrors echo_audibility.h: estimates the audibility of the
// echo from the render signal stationarity.
type EchoAudibility struct {
	renderSpectrumWritePrev      int
	renderSpectrumWritePrevValid bool
	renderBlockWritePrev         int
	nonZeroRenderSeen            bool
	useRenderStationarityAtInit  bool
	renderStationarity           StationarityEstimator
}

func NewEchoAudibility(useRenderStationarityAtInit bool) *EchoAudibility {
	e := &EchoAudibility{useRenderStationarityAtInit: useRenderStationarityAtInit}
	e.reset()
	return e
}

// Update feeds new render data to the audibility estimator.
func (e *EchoAudibility) Update(renderBuffer *RenderBuffer, averageReverb []float32, delayBlocks int, externalDelaySeen bool) {
	e.updateRenderNoiseEstimator(renderBuffer.GetSpectrumBuffer(), renderBuffer.GetBlockBuffer(), externalDelaySeen)
	if externalDelaySeen || e.useRenderStationarityAtInit {
		e.updateRenderStationarityFlags(renderBuffer, averageReverb, delayBlocks)
	}
}

// GetResidualEchoScaling returns the residual echo scaling per band.
func (e *EchoAudibility) GetResidualEchoScaling(filterHasHadTimeToConverge bool, residualScaling []float32) {
	for band := range residualScaling {
		if e.renderStationarity.IsBandStationary(band) && (filterHasHadTimeToConverge || e.useRenderStationarityAtInit) {
			residualScaling[band] = 0
		} else {
			residualScaling[band] = 1
		}
	}
}

// IsBlockStationary reports whether the render block is stationary.
func (e *EchoAudibility) IsBlockStationary() bool { return e.renderStationarity.IsBlockStationary() }

func (e *EchoAudibility) reset() {
	e.renderStationarity.Reset()
	e.nonZeroRenderSeen = false
	e.renderSpectrumWritePrevValid = false
}

func (e *EchoAudibility) updateRenderStationarityFlags(renderBuffer *RenderBuffer, averageReverb []float32, minChannelDelayBlocks int) {
	spectrumBuffer := renderBuffer.GetSpectrumBuffer()
	idxAtDelay := spectrumBuffer.OffsetIndex(spectrumBuffer.Read, minChannelDelayBlocks)
	numLookahead := renderBuffer.Headroom() - minChannelDelayBlocks + 1
	numLookahead = max(0, numLookahead)
	e.renderStationarity.UpdateStationarityFlags(spectrumBuffer, averageReverb, idxAtDelay, numLookahead)
}

func (e *EchoAudibility) updateRenderNoiseEstimator(spectrumBuffer *SpectrumBuffer, blockBuffer *BlockBuffer, externalDelaySeen bool) {
	if !e.renderSpectrumWritePrevValid {
		e.renderSpectrumWritePrev = spectrumBuffer.Write
		e.renderSpectrumWritePrevValid = true
		e.renderBlockWritePrev = blockBuffer.Write
		return
	}
	renderSpectrumWriteCurrent := spectrumBuffer.Write
	if !e.nonZeroRenderSeen && !externalDelaySeen {
		e.nonZeroRenderSeen = !e.isRenderTooLow(blockBuffer)
	}
	if e.nonZeroRenderSeen {
		for idx := e.renderSpectrumWritePrev; idx != renderSpectrumWriteCurrent; idx = spectrumBuffer.DecIndex(idx) {
			e.renderStationarity.UpdateNoiseEstimator(spectrumBuffer.Buffer[idx])
		}
	}
	e.renderSpectrumWritePrev = renderSpectrumWriteCurrent
}

func (e *EchoAudibility) isRenderTooLow(blockBuffer *BlockBuffer) bool {
	numRenderChannels := blockBuffer.Buffer[0].NumChannels()
	tooLow := false
	renderBlockWriteCurrent := blockBuffer.Write
	if renderBlockWriteCurrent == e.renderBlockWritePrev {
		tooLow = true
	} else {
		for idx := e.renderBlockWritePrev; idx != renderBlockWriteCurrent; idx = blockBuffer.IncIndex(idx) {
			var maxAbsOverChannels float32
			for ch := 0; ch < numRenderChannels; ch++ {
				// std::max(std::fabs(*r.first), std::fabs(*r.second)) with
				// r = std::minmax_element(block).
				maxAbsChannel := afirMaxAbsMinMax(blockBuffer.Buffer[idx].View(0, ch))
				maxAbsOverChannels = stdMax(maxAbsOverChannels, maxAbsChannel)
			}
			if maxAbsOverChannels < 10 {
				tooLow = true
				break
			}
		}
	}
	e.renderBlockWritePrev = renderBlockWriteCurrent
	return tooLow
}
