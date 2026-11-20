package aec3

// EchoPathDelayEstimator estimates the delay of the echo path
// (echo_path_delay_estimator.h).
type EchoPathDelayEstimator struct {
	*echoPathDelayEstimatorState
	captureMixer AlignmentMixer
}

// echoPathDelayEstimatorState is the state of an EchoPathDelayEstimator but
// the slice of its mixer. The matched filter and the lag aggregator are
// part of it as they are, with their slices.
type echoPathDelayEstimatorState struct {
	downSamplingFactor         int
	subBlockSize               int
	captureMixer               alignmentMixerState
	captureDecimator           Decimator
	matchedFilter              MatchedFilter
	matchedFilterLagAggregator MatchedFilterLagAggregator
	oldAggregatedLag           DelayEstimate
	hasOldAggregatedLag        bool
	consistentEstimateCounter  int
	clockdriftDetector         ClockdriftDetector
}

// NewEchoPathDelayEstimator mirrors the C++ constructor (without the data
// dumper). The matched filter kernel follows DetectOptimization().
func NewEchoPathDelayEstimator(config *EchoCanceller3Config, numCaptureChannels int) *EchoPathDelayEstimator {
	e := &EchoPathDelayEstimator{}
	e.bind(&echoPathDelayEstimatorState{}, nil)
	e.init(config, numCaptureChannels)
	return e
}

// bind makes e the view of st.
func (e *EchoPathDelayEstimator) bind(st *echoPathDelayEstimatorState, b *binder) {
	e.echoPathDelayEstimatorState = st
	e.captureMixer.bind(&st.captureMixer, b)
}

func (e *EchoPathDelayEstimator) init(config *EchoCanceller3Config, numCaptureChannels int) {
	e.downSamplingFactor = config.Delay.DownSamplingFactor
	e.subBlockSize = BlockSize
	if e.downSamplingFactor != 0 {
		e.subBlockSize = BlockSize / e.downSamplingFactor
	}
	m := &config.Delay.CaptureAlignmentMixing
	e.captureMixer.init(numCaptureChannels, m.Downmix, m.AdaptiveSelection, m.ActivityPowerThreshold,
		m.PreferFirstTwoChannels)
	e.captureDecimator.init(e.downSamplingFactor)
	excitationLimit := config.RenderLevels.PoorExcitationRenderLimit
	if config.Delay.DownSamplingFactor == 8 {
		excitationLimit = config.RenderLevels.PoorExcitationRenderLimitDs8
	}
	e.matchedFilter.init(DetectOptimization(), e.subBlockSize, MatchedFilterWindowSizeSubBlocks,
		config.Delay.NumFilters, MatchedFilterAlignmentShiftSizeSubBlocks, excitationLimit,
		config.Delay.DelayEstimateSmoothing, config.Delay.DelayEstimateSmoothingDelayFound,
		config.Delay.DelayCandidateDetectionThreshold, config.Delay.DetectPreEcho)
	e.matchedFilterLagAggregator.init(e.matchedFilter.GetMaxFilterLag(), &config.Delay)
}

// Reset resets the estimation. If the delay confidence is reset, the reset
// behavior is as if the call is restarted.
func (e *EchoPathDelayEstimator) Reset(resetDelayConfidence bool) {
	e.reset(true, resetDelayConfidence)
}

// EstimateDelay produces a delay estimate (in samples) if one is available.
func (e *EchoPathDelayEstimator) EstimateDelay(renderBuffer *DownsampledRenderBuffer, capture *Block) (DelayEstimate, bool) {
	return e.EstimateDelayWide(renderBuffer, capture, nil)
}

// EstimateDelayWide is EstimateDelay with the room the matched filter can
// use for widened render samples (MatchedFilter.UpdateWide); wide may be nil.
func (e *EchoPathDelayEstimator) EstimateDelayWide(renderBuffer *DownsampledRenderBuffer, capture *Block,
	wide *dlyWide) (DelayEstimate, bool) {
	var downsampledCaptureData [BlockSize]float32
	downsampledCapture := downsampledCaptureData[:e.subBlockSize]

	var downmixedCapture [BlockSize]float32
	e.captureMixer.ProduceOutput(capture, downmixedCapture[:])
	e.captureDecimator.Decimate(downmixedCapture[:], downsampledCapture)
	e.matchedFilter.UpdateWide(renderBuffer, downsampledCapture, e.matchedFilterLagAggregator.ReliableDelayFound(),
		wide)

	aggregated, ok := e.matchedFilterLagAggregator.Aggregate(e.matchedFilter.GetBestLagEstimate())

	// Run clockdrift detection.
	if ok && aggregated.Quality == DelayQualityRefined {
		e.clockdriftDetector.Update(e.matchedFilterLagAggregator.GetDelayAtHighestPeak())
	}

	// Return the detected delay in samples as the aggregated matched filter lag
	// compensated by the down sampling factor for the signal being correlated.
	if ok {
		aggregated.Delay *= e.downSamplingFactor
	}

	if e.hasOldAggregatedLag && ok && e.oldAggregatedLag.Delay == aggregated.Delay {
		e.consistentEstimateCounter++
	} else {
		e.consistentEstimateCounter = 0
	}
	e.oldAggregatedLag, e.hasOldAggregatedLag = aggregated, ok
	const numBlocksPerSecondBy2 = NumBlocksPerSecond / 2
	if e.consistentEstimateCounter > numBlocksPerSecondBy2 {
		e.reset(false, false)
	}
	return aggregated, ok
}

// Clockdrift returns the level of detected clockdrift.
func (e *EchoPathDelayEstimator) Clockdrift() ClockdriftDetectorLevel {
	return e.clockdriftDetector.ClockdriftLevel()
}

func (e *EchoPathDelayEstimator) reset(resetLagAggregator, resetDelayConfidence bool) {
	if resetLagAggregator {
		e.matchedFilterLagAggregator.Reset(resetDelayConfidence)
	}
	e.matchedFilter.Reset(resetLagAggregator)
	e.hasOldAggregatedLag = false
	e.consistentEstimateCounter = 0
}
