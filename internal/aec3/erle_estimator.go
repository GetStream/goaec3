package aec3

// ErleEstimator mirrors erle_estimator.h: combines the fullband, subband and
// (when erle.num_sections > 1) signal dependent ERLE estimators.
type ErleEstimator struct {
	signalDependentErleEstimator *SignalDependentErleEstimator
	fullbandErleEstimator        FullBandErleEstimator
	subbandErleEstimator         SubbandErleEstimator
	startupPhaseLengthBlocks     int
	blocksSinceReset             int
}

func NewErleEstimator(startupPhaseLengthBlocks int, config *EchoCanceller3Config, numCaptureChannels int) *ErleEstimator {
	e := &ErleEstimator{}
	e.init(startupPhaseLengthBlocks, config, numCaptureChannels)
	return e
}

// init sets up e in place.
func (e *ErleEstimator) init(startupPhaseLengthBlocks int, config *EchoCanceller3Config, numCaptureChannels int) {
	*e = ErleEstimator{startupPhaseLengthBlocks: startupPhaseLengthBlocks}
	e.fullbandErleEstimator.init(config, numCaptureChannels)
	e.subbandErleEstimator.init(config, numCaptureChannels)
	if config.Erle.NumSections > 1 {
		e.signalDependentErleEstimator = NewSignalDependentErleEstimator(config, numCaptureChannels)
	}
	e.Reset(true)
}

// Reset resets the estimators; delayChange also restarts the startup phase.
func (e *ErleEstimator) Reset(delayChange bool) {
	e.fullbandErleEstimator.Reset()
	e.subbandErleEstimator.Reset()
	if e.signalDependentErleEstimator != nil {
		e.signalDependentErleEstimator.Reset()
	}
	if delayChange {
		e.blocksSinceReset = 0
	}
}

// Update updates the ERLE estimates.
func (e *ErleEstimator) Update(renderBuffer *RenderBuffer, filterFrequencyResponses [][][FftLengthBy2Plus1]float32,
	avgRenderSpectrumWithReverb *[FftLengthBy2Plus1]float32, captureSpectra, subtractorSpectra [][FftLengthBy2Plus1]float32,
	convergedFilters []bool) {
	X2Reverb := avgRenderSpectrumWithReverb
	Y2 := captureSpectra
	E2 := subtractorSpectra
	e.blocksSinceReset++
	if e.blocksSinceReset < e.startupPhaseLengthBlocks {
		return
	}
	e.subbandErleEstimator.Update(X2Reverb, Y2, E2, convergedFilters)
	if e.signalDependentErleEstimator != nil {
		e.signalDependentErleEstimator.Update(renderBuffer, filterFrequencyResponses, X2Reverb, Y2, E2,
			e.subbandErleEstimator.Erle(false), e.subbandErleEstimator.Erle(true), convergedFilters)
	}
	e.fullbandErleEstimator.Update(X2Reverb[:], Y2, E2, convergedFilters)
}

// Erle returns the ERLE, onset compensated when requested.
func (e *ErleEstimator) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	if e.signalDependentErleEstimator != nil {
		return e.signalDependentErleEstimator.Erle(onsetCompensated)
	}
	return e.subbandErleEstimator.Erle(onsetCompensated)
}

// ErleUnbounded returns the non-capped ERLE.
func (e *ErleEstimator) ErleUnbounded() [][FftLengthBy2Plus1]float32 {
	if e.signalDependentErleEstimator == nil {
		return e.subbandErleEstimator.ErleUnbounded()
	}
	return e.signalDependentErleEstimator.Erle(false)
}

// ErleDuringOnsets returns the ERLE estimated during onsets.
func (e *ErleEstimator) ErleDuringOnsets() [][FftLengthBy2Plus1]float32 {
	return e.subbandErleEstimator.ErleDuringOnsets()
}

// FullbandErleLog2 returns the fullband ERLE in log2 units.
func (e *ErleEstimator) FullbandErleLog2() float32 { return e.fullbandErleEstimator.FullbandErleLog2() }

// GetInstLinearQualityEstimates returns the per-channel linear filter quality.
func (e *ErleEstimator) GetInstLinearQualityEstimates() []LinearFilterQuality {
	return e.fullbandErleEstimator.GetInstLinearQualityEstimates()
}
