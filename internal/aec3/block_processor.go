package aec3

// BlockProcessorEchoRemover is the part of the EchoRemover interface that
// BlockProcessor drives. *EchoRemover implements it; the C++ test-only
// BlockProcessor::Create overload that injects an EchoRemover maps to
// NewBlockProcessorWithComponents.
type BlockProcessorEchoRemover interface {
	GetMetrics() EchoControlMetrics
	ProcessCapture(echoPathVariability EchoPathVariability, captureSignalSaturation bool,
		externalDelay *DelayEstimate, renderBuffer *RenderBuffer, linearOutput, capture *Block)
	UpdateEchoLeakageStatus(leakageDetected bool)
	SetCaptureOutputUsage(captureOutputUsed bool)
}

// BlockProcessor performs echo cancellation on 64 sample blocks of audio
// data (block_processor.h; this is the BlockProcessorImpl of
// block_processor.cc). Metrics reporting is not ported.
//
// The render delay buffer, the delay controller and the echo remover that
// NewBlockProcessor creates are part of the BlockProcessor, which must not
// be copied (its components refer to it).
type BlockProcessor struct {
	_ noCopy
	// The components of NewBlockProcessorWithComponents; nil otherwise.
	injected *blockProcessorComponents

	ownRenderBuffer    RenderDelayBuffer
	ownDelayController RenderDelayController
	ownEchoRemover     EchoRemover

	config                 EchoCanceller3Config
	captureProperlyStarted bool
	renderProperlyStarted  bool
	sampleRateHz           int
	hasDelayController     bool // false with an external delay estimator
	renderEvent            BufferingEvent
	captureCallCounter     int
	estimatedDelay         DelayEstimate
	hasEstimatedDelay      bool

	// delayWide is room for the matched filter of the delay controller
	// (RenderDelayController.GetDelayWide). It has no pointers and comes
	// last.
	delayWide dlyWide
}

// blockProcessorComponents are the components handed to
// NewBlockProcessorWithComponents.
type blockProcessorComponents struct {
	renderBuffer    *RenderDelayBuffer
	delayController *RenderDelayController
	echoRemover     BlockProcessorEchoRemover
}

// NewBlockProcessor mirrors BlockProcessor::Create(env, config,
// sample_rate_hz, num_render_channels, num_capture_channels).
func NewBlockProcessor(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int) *BlockProcessor {
	p := &BlockProcessor{}
	p.init(config, sampleRateHz, numRenderChannels, numCaptureChannels)
	return p
}

// init sets up p in place, as NewBlockProcessor does.
func (p *BlockProcessor) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for BlockProcessor")
	}
	*p = BlockProcessor{}
	p.config = *config
	config = &p.config
	p.sampleRateHz = sampleRateHz
	p.renderEvent = BufferingEventNone
	p.ownRenderBuffer.init(config, sampleRateHz, numRenderChannels)
	if !config.Delay.UseExternalDelayEstimator {
		p.hasDelayController = true
		p.ownDelayController.init(config, sampleRateHz, numCaptureChannels)
	}
	p.ownEchoRemover.init(config, sampleRateHz, numRenderChannels, numCaptureChannels, DetectOptimization())
}

// NewBlockProcessorWithComponents mirrors the BlockProcessor::Create
// overload that takes the render delay buffer, the delay controller (nil
// when the config uses an external delay estimator) and the echo remover.
func NewBlockProcessorWithComponents(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, renderBuffer *RenderDelayBuffer, delayController *RenderDelayController,
	echoRemover BlockProcessorEchoRemover) *BlockProcessor {
	_, _ = numRenderChannels, numCaptureChannels
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for BlockProcessor")
	}
	return &BlockProcessor{
		config:             *config,
		sampleRateHz:       sampleRateHz,
		hasDelayController: delayController != nil,
		renderEvent:        BufferingEventNone,
		injected: &blockProcessorComponents{renderBuffer: renderBuffer, delayController: delayController,
			echoRemover: echoRemover},
	}
}

// renderBuffer returns the render delay buffer in use.
func (p *BlockProcessor) renderBuffer() *RenderDelayBuffer {
	if p.injected != nil {
		return p.injected.renderBuffer
	}
	return &p.ownRenderBuffer
}

// delayController returns the delay controller in use; it is only valid
// with hasDelayController.
func (p *BlockProcessor) delayController() *RenderDelayController {
	if p.injected != nil {
		return p.injected.delayController
	}
	return &p.ownDelayController
}

// ZeroGainSkipsForTesting returns the number of filter adaptations the
// processor's own echo remover skipped because the gain was zero. Tests use
// it to check that a scenario reaches the skip.
func (p *BlockProcessor) ZeroGainSkipsForTesting() uint64 {
	return p.ownEchoRemover.subtractor.ZeroGainSkipsForTesting()
}

// ProcessCapture processes a block of capture data. linearOutput is nil
// when the linear filter output is not exported.
func (p *BlockProcessor) ProcessCapture(echoPathGainChange, captureSignalSaturation bool,
	linearOutput, captureBlock *Block) {
	if captureBlock.NumBands() != NumBandsForRate(p.sampleRateHz) {
		panic("aec3: BlockProcessor capture block has the wrong number of bands")
	}

	p.captureCallCounter++

	if p.renderProperlyStarted {
		if !p.captureProperlyStarted {
			p.captureProperlyStarted = true
			p.renderBuffer().Reset()
			if p.hasDelayController {
				p.delayController().Reset(true)
			}
		}
	} else {
		// If no render data has yet arrived, do not process the capture
		// signal.
		p.renderBuffer().HandleSkippedCaptureProcessing()
		return
	}

	echoPathVariability := EchoPathVariability{
		GainChange:  echoPathGainChange,
		DelayChange: DelayAdjustmentNone,
		ClockDrift:  false,
	}

	if p.renderEvent == BufferingEventRenderOverrun && p.renderProperlyStarted {
		echoPathVariability.DelayChange = DelayAdjustmentBufferFlush
		if p.hasDelayController {
			p.delayController().Reset(true)
		}
	}
	p.renderEvent = BufferingEventNone

	// Update the render buffers with any newly arrived render blocks and
	// prepare the render buffers for reading the render data corresponding
	// to the current capture block.
	bufferEvent := p.renderBuffer().PrepareCaptureProcessing()
	// Reset the delay controller at render buffer underrun.
	if bufferEvent == BufferingEventRenderUnderrun {
		if p.hasDelayController {
			p.delayController().Reset(false)
		}
	}

	hasDelayEstimator := !p.config.Delay.UseExternalDelayEstimator
	if hasDelayEstimator {
		// Compute and apply the render delay required to achieve proper
		// signal alignment.
		p.estimatedDelay, p.hasEstimatedDelay = p.delayController().GetDelayWide(
			p.renderBuffer().GetDownsampledRenderBuffer(), p.renderBuffer().Delay(), captureBlock, &p.delayWide)

		if p.hasEstimatedDelay {
			if p.renderBuffer().AlignFromDelay(p.estimatedDelay.Delay) {
				echoPathVariability.DelayChange = DelayAdjustmentNewDetectedDelay
			}
		}

		echoPathVariability.ClockDrift = p.delayController().HasClockdrift()
	} else {
		p.renderBuffer().AlignFromExternalDelay()
	}

	// Remove the echo from the capture signal.
	if hasDelayEstimator || p.renderBuffer().HasReceivedBufferDelay() {
		var externalDelay *DelayEstimate
		if p.hasEstimatedDelay {
			externalDelay = &p.estimatedDelay
		}
		if p.injected != nil {
			p.injected.echoRemover.ProcessCapture(echoPathVariability, captureSignalSaturation, externalDelay,
				p.renderBuffer().GetRenderBuffer(), linearOutput, captureBlock)
		} else {
			p.ownEchoRemover.ProcessCapture(echoPathVariability, captureSignalSaturation, externalDelay,
				p.ownRenderBuffer.GetRenderBuffer(), linearOutput, captureBlock)
		}
	}
}

// BufferRender buffers a block of render data supplied by a FrameBlocker.
func (p *BlockProcessor) BufferRender(block *Block) {
	if block.NumBands() != NumBandsForRate(p.sampleRateHz) {
		panic("aec3: BlockProcessor render block has the wrong number of bands")
	}
	p.renderEvent = p.renderBuffer().Insert(block)

	p.renderProperlyStarted = true
	if p.hasDelayController {
		p.delayController().LogRenderCall()
	}
}

// UpdateEchoLeakageStatus reports whether echo leakage has been detected in
// the echo canceller output.
func (p *BlockProcessor) UpdateEchoLeakageStatus(leakageDetected bool) {
	if p.injected != nil {
		p.injected.echoRemover.UpdateEchoLeakageStatus(leakageDetected)
		return
	}
	p.ownEchoRemover.UpdateEchoLeakageStatus(leakageDetected)
}

// GetMetrics returns the current metrics.
func (p *BlockProcessor) GetMetrics() EchoControlMetrics {
	var metrics EchoControlMetrics
	if p.injected != nil {
		metrics = p.injected.echoRemover.GetMetrics()
	} else {
		metrics = p.ownEchoRemover.GetMetrics()
	}
	const blockSizeMs = 4
	metrics.DelayMs = p.renderBuffer().Delay() * blockSizeMs
	return metrics
}

// SetAudioBufferDelay provides an optional external estimate of the audio
// buffer delay.
func (p *BlockProcessor) SetAudioBufferDelay(delayMs int) {
	p.renderBuffer().SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage specifies whether the capture output will be used.
// The block processor may then deactivate some of the processing when the
// output is anyway not used, for instance when the endpoint is muted.
func (p *BlockProcessor) SetCaptureOutputUsage(captureOutputUsed bool) {
	if p.injected != nil {
		p.injected.echoRemover.SetCaptureOutputUsage(captureOutputUsed)
		return
	}
	p.ownEchoRemover.SetCaptureOutputUsage(captureOutputUsed)
}
