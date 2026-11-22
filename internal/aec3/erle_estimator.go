package aec3

// ErleEstimator combines the fullband, the subband and, with
// erle.num_sections > 1, the signal dependent ERLE estimators
// (erle_estimator.h).
type ErleEstimator struct {
	*erleEstimatorState
	signalDependentErleEstimator *SignalDependentErleEstimator
	fullbandErleEstimator        FullBandErleEstimator
	subbandErleEstimator         SubbandErleEstimator
}

// erleEstimatorState is the state of an ErleEstimator except its pointers.
type erleEstimatorState struct {
	fullbandErleEstimator    fullBandErleEstimatorState
	subbandErleEstimator     subbandErleEstimatorState
	startupPhaseLengthBlocks int
	blocksSinceReset         int
}

func (e *ErleEstimator) bind(st *erleEstimatorState, b *binder) {
	e.erleEstimatorState = st
	if b != nil {
		bindPtr(b, &e.signalDependentErleEstimator, &b.x.signalDependentErleEstimator)
	}
	e.fullbandErleEstimator.bind(&st.fullbandErleEstimator, b)
	e.subbandErleEstimator.bind(&st.subbandErleEstimator, b)
}

func NewErleEstimator(startupPhaseLengthBlocks int, config *EchoCanceller3Config, numCaptureChannels int) *ErleEstimator {
	e := &ErleEstimator{}
	e.bind(&erleEstimatorState{}, bindAlone)
	e.init(startupPhaseLengthBlocks, config, numCaptureChannels)
	return e
}

func (e *ErleEstimator) init(startupPhaseLengthBlocks int, config *EchoCanceller3Config, numCaptureChannels int) {
	st := e.erleEstimatorState
	*st = erleEstimatorState{startupPhaseLengthBlocks: startupPhaseLengthBlocks}
	*e = ErleEstimator{}
	e.bind(st, bindAlone)
	e.fullbandErleEstimator.init(config, numCaptureChannels)
	e.subbandErleEstimator.init(config, numCaptureChannels)
	if config.Erle.NumSections > 1 {
		e.signalDependentErleEstimator = NewSignalDependentErleEstimator(config, numCaptureChannels)
	}
	e.Reset(true)
}

// Reset resets the estimators. With delayChange it restarts the startup
// phase.
func (e ErleEstimator) Reset(delayChange bool) {
	e.fullbandErleEstimator.Reset()
	e.subbandErleEstimator.Reset()
	if e.signalDependentErleEstimator != nil {
		e.signalDependentErleEstimator.Reset()
	}
	if delayChange {
		e.blocksSinceReset = 0
	}
}

// Update takes the spectra of one block.
func (e ErleEstimator) Update(renderBuffer *RenderBuffer, filterFrequencyResponses [][][FftLengthBy2Plus1]float32,
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

// Erle returns the ERLE, compensated for onsets if onsetCompensated.
func (e ErleEstimator) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	if e.signalDependentErleEstimator != nil {
		return e.signalDependentErleEstimator.Erle(onsetCompensated)
	}
	return e.subbandErleEstimator.Erle(onsetCompensated)
}

// ErleUnbounded returns the ERLE without its cap.
func (e ErleEstimator) ErleUnbounded() [][FftLengthBy2Plus1]float32 {
	if e.signalDependentErleEstimator == nil {
		return e.subbandErleEstimator.ErleUnbounded()
	}
	return e.signalDependentErleEstimator.Erle(false)
}

// ErleDuringOnsets returns the ERLE estimated during onsets.
func (e ErleEstimator) ErleDuringOnsets() [][FftLengthBy2Plus1]float32 {
	return e.subbandErleEstimator.ErleDuringOnsets()
}

// FullbandErleLog2 returns the fullband ERLE in log2 units.
func (e ErleEstimator) FullbandErleLog2() float32 { return e.fullbandErleEstimator.FullbandErleLog2() }

// GetInstLinearQualityEstimates returns the quality of the linear filter per
// channel.
func (e ErleEstimator) GetInstLinearQualityEstimates() []LinearFilterQuality {
	return e.fullbandErleEstimator.GetInstLinearQualityEstimates()
}
