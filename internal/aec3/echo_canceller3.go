package aec3

import "sync"

// SplitBandSize is the number of samples per band in a 10 ms frame
// (AudioBuffer::kSplitBandSize).
const SplitBandSize = 160

// AdjustConfig mirrors the function of the same name in echo_canceller3.cc,
// which adjusts config parameter dependencies through field trials. Every
// field trial is fixed at its default (disabled, no parameter string), so
// each adjustment keeps the incoming value: the float parameters make an
// exact float -> double -> float round trip and the range-checked overrides
// only apply when the parsed value differs from the current one. The
// adjusted config is therefore a copy of the input.
func AdjustConfig(config *EchoCanceller3Config) EchoCanceller3Config {
	adjusted := *config

	// WebRTC-Aec3UseNearendReverbLen: the parsed values are the current ones
	// and are only applied when both lie in (-1, 1).
	defaultLen := float32(float64(adjusted.EpStrength.DefaultLen))
	nearendLen := float32(float64(adjusted.EpStrength.NearendLen))
	if defaultLen > -1 && defaultLen < 1 && nearendLen > -1 && nearendLen < 1 {
		adjusted.EpStrength.DefaultLen = defaultLen
		adjusted.EpStrength.NearendLen = nearendLen
	}
	return adjusted
}

func ec3DetectSaturation(y []float32) bool {
	for _, v := range y {
		if v >= 32700.0 || v <= -32700.0 {
			return true
		}
	}
	return false
}

// ec3RenderTransferQueue is the SwapQueue that moves 10 ms render frames
// ([band][channel][SplitBandSize]) from AnalyzeRender to the capture side.
// Insert copies the caller's frame into the queue and Remove copies the
// oldest queued frame out; Insert fails (and drops the frame) when maxFrames
// frames are queued, Remove fails when the queue is empty. The frames live
// in one flat buffer that starts at ec3RenderTransferQueueInitialFrames
// frames and doubles, up to maxFrames, when a render burst fills it, so the
// queue holds as many frames as the deepest burst so far. It is safe for one
// inserting and one removing goroutine.
type ec3RenderTransferQueue struct {
	mu             sync.Mutex
	numBands       int
	numChannels    int
	maxFrames      int
	buffer         []float32 // [frame][band][channel][SplitBandSize]
	nextWriteIndex int
	nextReadIndex  int
	numElements    int
}

// ec3RenderTransferQueueInitialFrames is the initial queue capacity: render
// and capture alternate in steady state, so one frame is queued at a time.
const ec3RenderTransferQueueInitialFrames = 4

// ec3NewFrame returns a [numBands][numChannels][frameLength] frame whose
// samples share one backing array.
func ec3NewFrame(numBands, numChannels, frameLength int) [][][]float32 {
	return newFlat3(numBands, numChannels, frameLength, frameLength)
}

func (q *ec3RenderTransferQueue) init(maxFrames, numBands, numChannels int) {
	q.numBands, q.numChannels, q.maxFrames = numBands, numChannels, maxFrames
	q.buffer = make([]float32, min(maxFrames, ec3RenderTransferQueueInitialFrames)*q.frameSize())
}

func (q *ec3RenderTransferQueue) frameSize() int { return q.numBands * q.numChannels * SplitBandSize }

func (q *ec3RenderTransferQueue) capacity() int { return len(q.buffer) / q.frameSize() }

func (q *ec3RenderTransferQueue) slot(index int) []float32 {
	n := q.frameSize()
	return q.buffer[index*n : (index+1)*n : (index+1)*n]
}

// grow doubles the capacity (up to maxFrames), keeping the queued frames in
// order at the start of the new buffer.
func (q *ec3RenderTransferQueue) grow() {
	n := q.frameSize()
	capacity := q.capacity()
	buffer := make([]float32, min(q.maxFrames, 2*capacity)*n)
	for i := 0; i < q.numElements; i++ {
		copy(buffer[i*n:(i+1)*n], q.slot((q.nextReadIndex+i)%capacity))
	}
	q.buffer = buffer
	q.nextReadIndex = 0
	q.nextWriteIndex = q.numElements
}

func (q *ec3RenderTransferQueue) Insert(input [][][]float32) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.numElements == q.capacity() {
		if q.numElements == q.maxFrames {
			return false
		}
		q.grow()
	}
	dst := q.slot(q.nextWriteIndex)
	for band := range input {
		for ch := range input[band] {
			copy(dst[(band*q.numChannels+ch)*SplitBandSize:][:SplitBandSize], input[band][ch])
		}
	}
	q.numElements++
	q.nextWriteIndex++
	if q.nextWriteIndex == q.capacity() {
		q.nextWriteIndex = 0
	}
	return true
}

func (q *ec3RenderTransferQueue) Remove(output [][][]float32) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.numElements == 0 {
		return false
	}
	src := q.slot(q.nextReadIndex)
	for band := range output {
		for ch := range output[band] {
			copy(output[band][ch], src[(band*q.numChannels+ch)*SplitBandSize:][:SplitBandSize])
		}
	}
	q.numElements--
	q.nextReadIndex++
	if q.nextReadIndex == q.capacity() {
		q.nextReadIndex = 0
	}
	return true
}

// ec3RenderWriter is EchoCanceller3::RenderWriter: it copies the split-band
// render signal into the transfer queue, optionally high-pass filtering the
// lowest band.
type ec3RenderWriter struct {
	numBands              int
	numChannels           int
	highPassFilter        *HighPassFilter
	renderQueueInputFrame [][][]float32
	renderTransferQueue   *ec3RenderTransferQueue
}

func ec3NewRenderWriter(config *EchoCanceller3Config, renderTransferQueue *ec3RenderTransferQueue,
	numBands, numChannels int) *ec3RenderWriter {
	w := &ec3RenderWriter{
		numBands:              numBands,
		numChannels:           numChannels,
		renderQueueInputFrame: ec3NewFrame(numBands, numChannels, SplitBandSize),
		renderTransferQueue:   renderTransferQueue,
	}
	if config.Filter.HighPassFilterEchoReference {
		w.highPassFilter = NewHighPassFilter(16000, numChannels)
	}
	return w
}

func (w *ec3RenderWriter) Insert(input [][][]float32) {
	if w.numBands != len(input) {
		return
	}

	// CopyBufferIntoFrame.
	for band := 0; band < w.numBands; band++ {
		for channel := 0; channel < w.numChannels; channel++ {
			copy(w.renderQueueInputFrame[band][channel], input[band][channel][:SplitBandSize])
		}
	}
	if w.highPassFilter != nil {
		w.highPassFilter.Process(w.renderQueueInputFrame[0])
	}

	_ = w.renderTransferQueue.Insert(w.renderQueueInputFrame)
}

// EchoCanceller3 is the main AEC3 class (echo_canceller3.h). It receives
// 10 ms frames of band-split audio, provides the BlockProcessor with blocks
// of 64 samples and partially handles the jitter in the render and capture
// API call sequence.
//
// Audio is in FloatS16 scale. A split-band frame is [band][channel][160]
// with NumBandsForRate(sampleRateHz) bands, which is what
// AudioBuffer::split_bands exposes; at 16 kHz the single band is the
// full-band signal.
//
// AnalyzeRender may be called concurrently with the other methods; the
// other methods must not be called concurrently with each other.
type EchoCanceller3 struct {
	renderWriter *ec3RenderWriter

	config                      EchoCanceller3Config
	sampleRateHz                int
	numBands                    int
	numRenderInputChannels      int
	numRenderChannelsToAec      int
	numCaptureChannels          int
	configSelector              *ConfigSelector
	multichannelContentDetector *MultiChannelContentDetector
	linearOutputFramer          *BlockFramer
	outputFramer                *BlockFramer
	captureBlocker              *FrameBlocker
	renderBlocker               *FrameBlocker
	renderTransferQueue         ec3RenderTransferQueue
	blockProcessor              *BlockProcessor
	renderQueueOutputFrame      [][][]float32
	saturatedMicrophoneSignal   bool
	renderBlock                 Block
	linearOutputBlock           *Block
	captureBlock                Block
	renderSubFrameView          [][][]float32
	linearOutputSubFrameView    [][][]float32
	captureSubFrameView         [][][]float32
	blockDelayBuffer            *BlockDelayBuffer
}

func ec3NewSubFrameView(numBands, numChannels int) [][][]float32 {
	return newFlat3(numBands, numChannels, 0, 0)
}

// NewEchoCanceller3 creates an echo canceller. multichannelConfig is the
// optional config used once proper multichannel render content is detected
// (nil corresponds to std::nullopt).
func NewEchoCanceller3(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels int) *EchoCanceller3 {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for EchoCanceller3")
	}
	e := &EchoCanceller3{
		config:                 AdjustConfig(config),
		sampleRateHz:           sampleRateHz,
		numBands:               NumBandsForRate(sampleRateHz),
		numRenderInputChannels: numRenderChannels,
		numCaptureChannels:     numCaptureChannels,
	}
	e.configSelector = NewConfigSelector(&e.config, multichannelConfig, e.numRenderInputChannels)
	active := e.configSelector.ActiveConfig()
	e.multichannelContentDetector = NewMultiChannelContentDetector(
		active.MultiChannel.DetectStereoContent, e.numRenderInputChannels,
		active.MultiChannel.StereoDetectionThreshold,
		active.MultiChannel.StereoDetectionTimeoutThresholdSeconds,
		active.MultiChannel.StereoDetectionHysteresisSeconds)
	e.outputFramer = NewBlockFramer(e.numBands, e.numCaptureChannels)
	e.captureBlocker = NewFrameBlocker(e.numBands, e.numCaptureChannels)
	e.renderTransferQueue.init(RenderTransferQueueSizeFrames, e.numBands, e.numRenderInputChannels)
	e.renderQueueOutputFrame = ec3NewFrame(e.numBands, e.numRenderInputChannels, SplitBandSize)
	e.renderBlock = NewBlock(e.numBands, e.numRenderInputChannels)
	e.captureBlock = NewBlock(e.numBands, e.numCaptureChannels)
	e.captureSubFrameView = ec3NewSubFrameView(e.numBands, e.numCaptureChannels)

	if active.Delay.FixedCaptureDelaySamples > 0 {
		e.blockDelayBuffer = NewBlockDelayBuffer(e.numCaptureChannels, e.numBands, SplitBandSize,
			e.config.Delay.FixedCaptureDelaySamples)
	}

	e.renderWriter = ec3NewRenderWriter(active, &e.renderTransferQueue, e.numBands, e.numRenderInputChannels)

	if active.Filter.ExportLinearAecOutput {
		e.linearOutputFramer = NewBlockFramer(1, e.numCaptureChannels)
		b := NewBlock(1, e.numCaptureChannels)
		e.linearOutputBlock = &b
		e.linearOutputSubFrameView = ec3NewSubFrameView(1, e.numCaptureChannels)
	}

	e.initialize()
	return e
}

// initialize (re-)initializes the selected subset of the fields, at creation
// as well as during reconfiguration.
func (e *EchoCanceller3) initialize() {
	e.numRenderChannelsToAec = 1
	if e.multichannelContentDetector.IsProperMultiChannelContentDetected() {
		e.numRenderChannelsToAec = e.numRenderInputChannels
	}

	e.configSelector.Update(e.multichannelContentDetector.IsProperMultiChannelContentDetected())

	e.renderBlock.SetNumChannels(e.numRenderChannelsToAec)

	e.renderBlocker = NewFrameBlocker(e.numBands, e.numRenderChannelsToAec)

	e.blockProcessor = NewBlockProcessor(e.configSelector.ActiveConfig(), e.sampleRateHz,
		e.numRenderChannelsToAec, e.numCaptureChannels)

	e.renderSubFrameView = ec3NewSubFrameView(e.numBands, e.numRenderChannelsToAec)
}

// SetBlockProcessorForTesting replaces the internal block processor.
func (e *EchoCanceller3) SetBlockProcessorForTesting(blockProcessor *BlockProcessor) {
	e.blockProcessor = blockProcessor
}

// StereoRenderProcessingActiveForTesting reports whether stereo processing
// is active.
func (e *EchoCanceller3) StereoRenderProcessingActiveForTesting() bool {
	return e.multichannelContentDetector.IsProperMultiChannelContentDetected()
}

// GetActiveConfigForTesting returns the active config.
func (e *EchoCanceller3) GetActiveConfigForTesting() *EchoCanceller3Config {
	return e.configSelector.ActiveConfig()
}

// AnalyzeRender analyzes and stores an internal copy of the split-band
// render signal ([band][channel][SplitBandSize]).
func (e *EchoCanceller3) AnalyzeRender(render [][][]float32) {
	if len(render) > 0 && len(render[0]) != e.numRenderInputChannels {
		panic("aec3: EchoCanceller3.AnalyzeRender channel count mismatch")
	}
	e.renderWriter.Insert(render)
}

// AnalyzeCapture analyzes the full-band capture signal ([channel][samples])
// to detect signal saturation.
func (e *EchoCanceller3) AnalyzeCapture(capture [][]float32) {
	e.saturatedMicrophoneSignal = false
	for channel := range capture {
		e.saturatedMicrophoneSignal = e.saturatedMicrophoneSignal || ec3DetectSaturation(capture[channel])
		if e.saturatedMicrophoneSignal {
			break
		}
	}
}

// ProcessCapture removes the echo from the split-band capture signal
// ([band][channel][SplitBandSize]) in place. linearOutput, when non-nil,
// receives the linear filter output ([channel][SplitBandSize], 16 kHz); it
// requires Filter.ExportLinearAecOutput, and must then be passed on every
// call. levelChange flags an echo path gain change.
func (e *EchoCanceller3) ProcessCapture(capture [][][]float32, linearOutput [][]float32, levelChange bool) {
	if len(capture) != e.numBands || len(capture[0]) != e.numCaptureChannels {
		panic("aec3: EchoCanceller3.ProcessCapture capture shape mismatch")
	}
	if linearOutput != nil && e.linearOutputFramer == nil {
		panic("aec3: linear AEC output requested without Filter.ExportLinearAecOutput")
	}

	// Optionally delay the capture signal.
	if e.configSelector.ActiveConfig().Delay.FixedCaptureDelaySamples > 0 {
		if e.blockDelayBuffer == nil {
			panic("aec3: fixed capture delay enabled by a config switch")
		}
		e.blockDelayBuffer.DelaySignal(capture)
	}

	e.emptyRenderQueue()

	e.processCaptureFrameContent(linearOutput, capture, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 0)
	e.processCaptureFrameContent(linearOutput, capture, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 1)
	e.processRemainingCaptureFrameContent(levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected())
}

// fillSubFrameView points view[band][channel] at subframe subFrameIndex of
// frame[band][channel].
func ec3FillSubFrameView(frame [][][]float32, subFrameIndex int, view [][][]float32) {
	start := subFrameIndex * SubFrameLength
	for band := range view {
		for channel := range view[band] {
			view[band][channel] = frame[band][channel][start : start+SubFrameLength : start+SubFrameLength]
		}
	}
}

func ec3FillLinearSubFrameView(frame [][]float32, subFrameIndex int, view [][][]float32) {
	start := subFrameIndex * SubFrameLength
	for channel := range view[0] {
		view[0][channel] = frame[channel][start : start+SubFrameLength : start+SubFrameLength]
	}
}

// ec3FillRenderSubFrameView is the FillSubFrameView overload for the render
// frame: when the AEC runs on fewer channels than the frame has, it selects
// channel 0, after averaging all channels into it when a proper downmix is
// needed.
func ec3FillRenderSubFrameView(properDownmixNeeded bool, frame [][][]float32, subFrameIndex int,
	view [][][]float32) {
	frameNumChannels := len(frame[0])
	subFrameNumChannels := len(view[0])
	start := subFrameIndex * SubFrameLength
	if frameNumChannels > subFrameNumChannels {
		if properDownmixNeeded {
			// When a proper downmix is needed (which is the case when proper
			// stereo is present in the echo reference signal but the echo
			// canceller does the processing in mono) downmix the echo
			// reference by averaging the channel content (otherwise
			// downmixing is done by selecting channel 0).
			for band := range frame {
				dst := frame[band][0][start : start+SubFrameLength]
				for ch := 1; ch < frameNumChannels; ch++ {
					src := frame[band][ch][start : start+SubFrameLength]
					for k := range dst {
						dst[k] += src[k]
					}
				}
				oneByNumChannels := float32(1.0) / float32(frameNumChannels)
				for k := range dst {
					dst[k] *= oneByNumChannels
				}
			}
		}
		for band := range frame {
			view[band][0] = frame[band][0][start : start+SubFrameLength : start+SubFrameLength]
		}
	} else {
		ec3FillSubFrameView(frame, subFrameIndex, view)
	}
}

func (e *EchoCanceller3) processCaptureFrameContent(linearOutput [][]float32, capture [][][]float32,
	levelChange, aecReferenceIsDownmixedStereo bool, subFrameIndex int) {
	ec3FillSubFrameView(capture, subFrameIndex, e.captureSubFrameView)

	if linearOutput != nil {
		ec3FillLinearSubFrameView(linearOutput, subFrameIndex, e.linearOutputSubFrameView)
	}

	e.captureBlocker.InsertSubFrameAndExtractBlock(e.captureSubFrameView, &e.captureBlock)
	e.blockProcessor.ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, e.linearOutputBlock, &e.captureBlock)
	e.outputFramer.InsertBlockAndExtractSubFrame(&e.captureBlock, e.captureSubFrameView)

	if linearOutput != nil {
		e.linearOutputFramer.InsertBlockAndExtractSubFrame(e.linearOutputBlock, e.linearOutputSubFrameView)
	}
}

func (e *EchoCanceller3) processRemainingCaptureFrameContent(levelChange, aecReferenceIsDownmixedStereo bool) {
	if !e.captureBlocker.IsBlockAvailable() {
		return
	}

	e.captureBlocker.ExtractBlock(&e.captureBlock)
	e.blockProcessor.ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, e.linearOutputBlock, &e.captureBlock)
	e.outputFramer.InsertBlock(&e.captureBlock)

	if e.linearOutputFramer != nil {
		e.linearOutputFramer.InsertBlock(e.linearOutputBlock)
	}
}

func (e *EchoCanceller3) bufferRenderFrameContent(properDownmixNeeded bool, subFrameIndex int) {
	ec3FillRenderSubFrameView(properDownmixNeeded, e.renderQueueOutputFrame, subFrameIndex, e.renderSubFrameView)
	e.renderBlocker.InsertSubFrameAndExtractBlock(e.renderSubFrameView, &e.renderBlock)
	e.blockProcessor.BufferRender(&e.renderBlock)
}

func (e *EchoCanceller3) bufferRemainingRenderFrameContent() {
	if !e.renderBlocker.IsBlockAvailable() {
		return
	}
	e.renderBlocker.ExtractBlock(&e.renderBlock)
	e.blockProcessor.BufferRender(&e.renderBlock)
}

// emptyRenderQueue empties the render transfer queue.
func (e *EchoCanceller3) emptyRenderQueue() {
	frameToBuffer := e.renderTransferQueue.Remove(e.renderQueueOutputFrame)
	for frameToBuffer {
		if e.multichannelContentDetector.UpdateDetection(e.renderQueueOutputFrame) {
			// Reinitialize the AEC when proper stereo is detected.
			e.initialize()
		}

		// Buffer frame content.
		e.bufferRenderFrameContent(e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 0)
		e.bufferRenderFrameContent(e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 1)
		e.bufferRemainingRenderFrameContent()

		frameToBuffer = e.renderTransferQueue.Remove(e.renderQueueOutputFrame)
	}
}

// GetMetrics collects the current metrics from the echo canceller.
func (e *EchoCanceller3) GetMetrics() EchoControlMetrics {
	return e.blockProcessor.GetMetrics()
}

// SetAudioBufferDelay provides an optional external estimate of the audio
// buffer delay.
func (e *EchoCanceller3) SetAudioBufferDelay(delayMs int) {
	e.blockProcessor.SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage specifies whether the capture output will be used,
// allowing some processing to be deactivated when it is not, for instance
// when the endpoint is muted.
func (e *EchoCanceller3) SetCaptureOutputUsage(captureOutputUsed bool) {
	e.blockProcessor.SetCaptureOutputUsage(captureOutputUsed)
}

// ActiveProcessing reports whether the echo canceller is processing; AEC3
// always is.
func (e *EchoCanceller3) ActiveProcessing() bool { return true }

// UpdateEchoLeakageStatus signals whether an external detector has detected
// echo leakage from the echo canceller. Once flagged, leakage should be
// unflagged when it no longer occurs.
func (e *EchoCanceller3) UpdateEchoLeakageStatus(leakageDetected bool) {
	e.blockProcessor.UpdateEchoLeakageStatus(leakageDetected)
}

// SampleRateHz returns the full-band sample rate.
func (e *EchoCanceller3) SampleRateHz() int { return e.sampleRateHz }

// NumBands returns the number of split bands.
func (e *EchoCanceller3) NumBands() int { return e.numBands }
