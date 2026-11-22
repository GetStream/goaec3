package aec3

func dlyComputeBufferDelay(currentDelay DelayEstimate, hasCurrentDelay bool, hysteresisLimitBlocks int,
	estimatedDelay DelayEstimate) DelayEstimate {
	// The buffer delay that gives the desired latency, with hysteresis.
	newDelayBlocks := estimatedDelay.Delay >> BlockSizeLog2
	if hasCurrentDelay {
		currentDelayBlocks := currentDelay.Delay
		if newDelayBlocks > currentDelayBlocks && newDelayBlocks <= currentDelayBlocks+hysteresisLimitBlocks {
			newDelayBlocks = currentDelayBlocks
		}
	}
	newDelay := estimatedDelay
	newDelay.Delay = newDelayBlocks
	return newDelay
}

// RenderDelayController aligns the render with the capture signal through a
// RenderDelayBuffer (RenderDelayControllerImpl in
// render_delay_controller.cc). Metrics reporting is not ported.
type RenderDelayController struct {
	*renderDelayControllerState
	delayEstimator EchoPathDelayEstimator
}

// renderDelayControllerState is the state of a RenderDelayController except
// what the view of its delay estimator holds.
type renderDelayControllerState struct {
	hysteresisLimitBlocks    int
	delay                    DelayEstimate
	hasDelay                 bool
	delayEstimator           echoPathDelayEstimatorState
	delaySamples             DelayEstimate
	hasDelaySamples          bool
	captureCallCounter       int
	delayChangeCounter       int
	lastDelayEstimateQuality DelayEstimateQuality
}

// NewRenderDelayController mirrors RenderDelayController::Create.
func NewRenderDelayController(config *EchoCanceller3Config, sampleRateHz, numCaptureChannels int) *RenderDelayController {
	c := &RenderDelayController{}
	c.bind(&renderDelayControllerState{}, bindAlone)
	c.init(config, sampleRateHz, numCaptureChannels)
	return c
}

func (c *RenderDelayController) bind(st *renderDelayControllerState, b *binder) {
	c.renderDelayControllerState = st
	c.delayEstimator.bind(&st.delayEstimator, b)
}

func (c *RenderDelayController) init(config *EchoCanceller3Config, sampleRateHz, numCaptureChannels int) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for RenderDelayController")
	}
	st := c.renderDelayControllerState
	*st = renderDelayControllerState{
		hysteresisLimitBlocks:    config.Delay.HysteresisLimitBlocks,
		lastDelayEstimateQuality: DelayQualityCoarse,
	}
	*c = RenderDelayController{}
	c.bind(st, bindAlone)
	c.delayEstimator.init(config, numCaptureChannels)
}

// Reset resets the controller. With resetDelayConfidence it restarts as at
// the start of a call.
func (c *RenderDelayController) Reset(resetDelayConfidence bool) {
	c.hasDelay = false
	c.hasDelaySamples = false
	c.delayEstimator.Reset(resetDelayConfidence)
	c.delayChangeCounter = 0
	if resetDelayConfidence {
		c.lastDelayEstimateQuality = DelayQualityCoarse
	}
}

// LogRenderCall does nothing: metrics reporting is not ported.
func (c *RenderDelayController) LogRenderCall() {}

// GetDelay aligns the render buffer with the capture signal. It returns the
// delay of the render delay buffer in blocks, if it has one.
func (c *RenderDelayController) GetDelay(renderBuffer *DownsampledRenderBuffer, renderDelayBufferDelay int,
	capture *Block) (DelayEstimate, bool) {
	return c.GetDelayWide(renderBuffer, renderDelayBufferDelay, capture, nil)
}

// GetDelayWide is GetDelay with work space for MatchedFilter.UpdateWide.
// wide may be nil.
func (c *RenderDelayController) GetDelayWide(renderBuffer *DownsampledRenderBuffer, renderDelayBufferDelay int,
	capture *Block, wide *dlyWide) (DelayEstimate, bool) {
	_ = renderDelayBufferDelay
	c.captureCallCounter++
	delaySamples, ok := c.delayEstimator.EstimateDelayWide(renderBuffer, capture, wide)
	if ok {
		if !c.hasDelaySamples || delaySamples.Delay != c.delaySamples.Delay {
			c.delayChangeCounter = 0
		}
		if c.hasDelaySamples {
			if c.delaySamples.Delay == delaySamples.Delay {
				c.delaySamples.BlocksSinceLastChange++
			} else {
				c.delaySamples.BlocksSinceLastChange = 0
			}
			c.delaySamples.BlocksSinceLastUpdate = 0
			c.delaySamples.Delay = delaySamples.Delay
			c.delaySamples.Quality = delaySamples.Quality
		} else {
			c.delaySamples, c.hasDelaySamples = delaySamples, true
		}
	} else if c.hasDelaySamples {
		c.delaySamples.BlocksSinceLastChange++
		c.delaySamples.BlocksSinceLastUpdate++
	}

	if c.delayChangeCounter < 2*NumBlocksPerSecond {
		c.delayChangeCounter++
	}

	if c.hasDelaySamples {
		useHysteresis := c.lastDelayEstimateQuality == DelayQualityRefined &&
			c.delaySamples.Quality == DelayQualityRefined
		hysteresis := 0
		if useHysteresis {
			hysteresis = c.hysteresisLimitBlocks
		}
		c.delay = dlyComputeBufferDelay(c.delay, c.hasDelay, hysteresis, c.delaySamples)
		c.hasDelay = true
		c.lastDelayEstimateQuality = c.delaySamples.Quality
	}
	if !c.hasDelay {
		return DelayEstimate{}, false
	}
	return c.delay, true
}

// HasClockdrift reports whether clock drift is detected.
func (c *RenderDelayController) HasClockdrift() bool {
	return c.delayEstimator.Clockdrift() != ClockdriftLevelNone
}
