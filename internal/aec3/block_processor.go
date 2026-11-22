package aec3

// BlockProcessorEchoRemover is the part of EchoRemover that BlockProcessor
// calls. NewBlockProcessorWithComponents takes one, like the test-only
// overload of BlockProcessor::Create.
type BlockProcessorEchoRemover interface {
	GetMetrics() EchoControlMetrics
	ProcessCapture(echoPathVariability EchoPathVariability, captureSignalSaturation bool,
		externalDelay *DelayEstimate, renderBuffer *RenderBuffer, linearOutput, capture *Block)
	UpdateEchoLeakageStatus(leakageDetected bool)
	SetCaptureOutputUsage(captureOutputUsed bool)
}

// BlockProcessor cancels the echo in blocks of 64 samples
// (BlockProcessorImpl in block_processor.cc). Metrics reporting is not
// ported.
//
// A BlockProcessor holds pointers to itself and is not copied once it is set
// up.
type BlockProcessor struct {
	*blockProcessorState
	// The components of NewBlockProcessorWithComponents, or nil.
	injected *blockProcessorComponents

	ownRenderBuffer    RenderDelayBuffer
	ownDelayController RenderDelayController
	ownEchoRemover     EchoRemover
}

// blockProcessorState is the state of a BlockProcessor except its pointers.
type blockProcessorState struct {
	_ noCopy

	ownRenderBuffer    renderDelayBufferState
	ownDelayController renderDelayControllerState
	ownEchoRemover     echoRemoverState

	config                 EchoCanceller3Config
	captureProperlyStarted bool
	renderProperlyStarted  bool
	sampleRateHz           int
	hasDelayController     bool // false with an external delay estimator
	renderEvent            BufferingEvent
	captureCallCounter     int
	estimatedDelay         DelayEstimate
	hasEstimatedDelay      bool

	// delayWide is work space of the matched filter. It has no pointers and
	// comes last.
	delayWide dlyWide
}

// blockProcessorComponents holds the components handed to
// NewBlockProcessorWithComponents.
type blockProcessorComponents struct {
	renderBuffer    *RenderDelayBuffer
	delayController *RenderDelayController
	echoRemover     BlockProcessorEchoRemover
}

func (p *BlockProcessor) bind(st *blockProcessorState, b *binder) {
	p.blockProcessorState = st
	p.ownRenderBuffer.bind(&st.ownRenderBuffer, b)
	p.ownDelayController.bind(&st.ownDelayController, b)
	p.ownEchoRemover.bind(&st.ownEchoRemover, b)
}

// init sets up p in place. It mirrors BlockProcessor::Create(env, config,
// sample_rate_hz, num_render_channels, num_capture_channels).
func (p *BlockProcessor) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for BlockProcessor")
	}
	st := p.blockProcessorState
	*st = blockProcessorState{}
	*p = BlockProcessor{}
	p.bind(st, bindAlone)
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
// overload that takes the components. delayController is nil when the config
// uses an external delay estimator.
func NewBlockProcessorWithComponents(config *EchoCanceller3Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int, renderBuffer *RenderDelayBuffer, delayController *RenderDelayController,
	echoRemover BlockProcessorEchoRemover) *BlockProcessor {
	_, _ = numRenderChannels, numCaptureChannels
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for BlockProcessor")
	}
	p := &BlockProcessor{
		injected: &blockProcessorComponents{renderBuffer: renderBuffer, delayController: delayController,
			echoRemover: echoRemover},
	}
	p.bind(&blockProcessorState{
		config:             *config,
		sampleRateHz:       sampleRateHz,
		hasDelayController: delayController != nil,
		renderEvent:        BufferingEventNone,
	}, bindAlone)
	return p
}

func (p *BlockProcessor) renderBuffer() *RenderDelayBuffer {
	if p.injected != nil {
		return p.injected.renderBuffer
	}
	return &p.ownRenderBuffer
}

// delayController returns the delay controller in use. It is valid only with
// hasDelayController.
func (p *BlockProcessor) delayController() *RenderDelayController {
	if p.injected != nil {
		return p.injected.delayController
	}
	return &p.ownDelayController
}

// ZeroGainSkipsForTesting returns the number of filter adaptations the
// zero-gain skip left out in the echo remover of p.
func (p *BlockProcessor) ZeroGainSkipsForTesting() uint64 {
	return p.ownEchoRemover.subtractor.ZeroGainSkipsForTesting()
}

// ProcessCapture processes a block of capture data in place. linearOutput is
// nil when the output of the linear filter is not wanted.
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
		// Without render data the capture signal passes unprocessed.
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

	// Take in the render blocks that arrived and align the render buffers
	// with the capture block.
	bufferEvent := p.renderBuffer().PrepareCaptureProcessing()
	if bufferEvent == BufferingEventRenderUnderrun {
		if p.hasDelayController {
			p.delayController().Reset(false)
		}
	}

	hasDelayEstimator := !p.config.Delay.UseExternalDelayEstimator
	if hasDelayEstimator {
		// Estimate the render delay and apply it.
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

	if hasDelayEstimator || p.renderBuffer().HasReceivedBufferDelay() {
		var externalDelay *DelayEstimate
		if p.hasEstimatedDelay {
			externalDelay = &p.estimatedDelay
		}
		if p.injected != nil {
			// A call through the interface would move the blocks of every
			// caller to the heap. The remover gets copies of the block
			// headers, which share the samples.
			capture := *captureBlock
			var linear *Block
			if linearOutput != nil {
				l := *linearOutput
				linear = &l
			}
			p.injected.echoRemover.ProcessCapture(echoPathVariability, captureSignalSaturation, externalDelay,
				p.injected.renderBuffer.GetRenderBuffer(), linear, &capture)
		} else {
			renderBuffer := p.ownRenderBuffer.renderBuffer()
			p.ownEchoRemover.ProcessCapture(echoPathVariability, captureSignalSaturation, externalDelay,
				&renderBuffer, linearOutput, captureBlock)
		}
	}
}

// BufferRender buffers a block of render data.
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

// UpdateEchoLeakageStatus tells the processor whether echo leaks into the
// output.
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

// SetAudioBufferDelay sets an external estimate of the audio buffer delay.
func (p *BlockProcessor) SetAudioBufferDelay(delayMs int) {
	p.renderBuffer().SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage tells the processor whether the capture output is
// used. It may skip work whose result nothing uses, as when the endpoint is
// muted.
func (p *BlockProcessor) SetCaptureOutputUsage(captureOutputUsed bool) {
	if p.injected != nil {
		p.injected.echoRemover.SetCaptureOutputUsage(captureOutputUsed)
		return
	}
	p.ownEchoRemover.SetCaptureOutputUsage(captureOutputUsed)
}
