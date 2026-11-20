package aec3

func dlyComputeBufferDelay(currentDelay DelayEstimate, hasCurrentDelay bool, hysteresisLimitBlocks int,
	estimatedDelay DelayEstimate) DelayEstimate {
	// Compute the buffer delay increase required to achieve the desired
	// latency.
	newDelayBlocks := estimatedDelay.Delay >> BlockSizeLog2
	// Add hysteresis.
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

// RenderDelayController aligns the render and capture signal using a
// RenderDelayBuffer (render_delay_controller.h; this is the
// RenderDelayControllerImpl of render_delay_controller.cc). Metrics
// reporting is not ported.
type RenderDelayController struct {
	*renderDelayControllerState
	delayEstimator EchoPathDelayEstimator
}

// renderDelayControllerState is the state of a RenderDelayController but
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
	c.bind(&renderDelayControllerState{}, nil)
	c.init(config, sampleRateHz, numCaptureChannels)
	return c
}

// bind makes c the view of st.
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
	c.bind(st, nil)
	c.delayEstimator.init(config, numCaptureChannels)
}

// Reset resets the delay controller. If the delay confidence is reset, the
// reset behavior is as if the call is restarted.
func (c *RenderDelayController) Reset(resetDelayConfidence bool) {
	c.hasDelay = false
	c.hasDelaySamples = false
	c.delayEstimator.Reset(resetDelayConfidence)
	c.delayChangeCounter = 0
	if resetDelayConfidence {
		c.lastDelayEstimateQuality = DelayQualityCoarse
	}
}

// LogRenderCall logs a render call.
func (c *RenderDelayController) LogRenderCall() {}

// GetDelay aligns the render buffer content with the capture signal and
// returns the render delay buffer delay in blocks, if any.
func (c *RenderDelayController) GetDelay(renderBuffer *DownsampledRenderBuffer, renderDelayBufferDelay int,
	capture *Block) (DelayEstimate, bool) {
	return c.GetDelayWide(renderBuffer, renderDelayBufferDelay, capture, nil)
}

// GetDelayWide is GetDelay with the room the matched filter can use for
// widened render samples (MatchedFilter.UpdateWide); wide may be nil. The
// room is pointer-free and large, so the caller keeps it behind its last
// pointer.
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
		// Compute the render delay buffer delay.
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

// HasClockdrift returns true if clockdrift has been detected.
func (c *RenderDelayController) HasClockdrift() bool {
	return c.delayEstimator.Clockdrift() != ClockdriftLevelNone
}
