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

// Frame is a 10 ms frame of split-band audio, [band][channel][SplitBandSize]
// in FloatS16 scale, stored contiguously in one pointer-free backing array:
// what AudioBuffer::split_bands exposes. View and SubFrame return views
// computed from the band and channel.
type Frame struct {
	numBands    int
	numChannels int
	data        []float32
}

// NewFrame returns a zero frame of numBands bands and numChannels channels.
func NewFrame(numBands, numChannels int) Frame {
	return ec3FrameOver(numBands, numChannels, make([]float32, numBands*numChannels*SplitBandSize))
}

// ec3FrameOver returns a frame that keeps its samples in storage.
func ec3FrameOver(numBands, numChannels int, storage []float32) Frame {
	n := numBands * numChannels * SplitBandSize
	return Frame{numBands: numBands, numChannels: numChannels, data: storage[:n:n]}
}

func (f Frame) NumBands() int    { return f.numBands }
func (f Frame) NumChannels() int { return f.numChannels }

// Samples returns all samples of the frame, [band][channel][SplitBandSize].
func (f Frame) Samples() []float32 { return f.data }

// View returns the SplitBandSize samples of one band and channel.
func (f Frame) View(band, channel int) []float32 {
	i := (band*f.numChannels + channel) * SplitBandSize
	return f.data[i : i+SplitBandSize : i+SplitBandSize]
}

// SubFrame returns the SubFrameLength samples of subframe subFrameIndex (0
// or 1) of one band and channel.
func (f Frame) SubFrame(band, channel, subFrameIndex int) []float32 {
	i := (band*f.numChannels+channel)*SplitBandSize + subFrameIndex*SubFrameLength
	return f.data[i : i+SubFrameLength : i+SubFrameLength]
}

// CopyFrom copies frame[band][channel] into f.
func (f Frame) CopyFrom(frame [][][]float32) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			copy(f.View(band, channel), frame[band][channel][:SplitBandSize])
		}
	}
}

// CopyTo copies f into frame[band][channel].
func (f Frame) CopyTo(frame [][][]float32) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			copy(frame[band][channel][:SplitBandSize], f.View(band, channel))
		}
	}
}

// ec3RenderTransferQueue is the SwapQueue that moves 10 ms render frames
// from AnalyzeRender to the capture side. Insert copies the caller's frame
// into the queue and Remove copies the oldest queued frame out; Insert fails
// (and drops the frame) when maxFrames frames are queued, Remove fails when
// the queue is empty. The frames live in one flat buffer that starts at
// ec3RenderTransferQueueInitialFrames frames and doubles, up to maxFrames,
// when a render burst fills it, so the queue holds as many frames as the
// deepest burst so far. It is safe for one inserting and one removing
// goroutine.
type ec3RenderTransferQueue struct {
	mu             sync.Mutex
	frameSize      int // samples per frame
	maxFrames      int
	buffer         []float32 // [frame][band][channel][SplitBandSize]
	nextWriteIndex int
	nextReadIndex  int
	numElements    int
}

// ec3RenderTransferQueueInitialFrames is the initial queue capacity: render
// and capture alternate in steady state, so one frame is queued at a time.
const ec3RenderTransferQueueInitialFrames = 4

func (q *ec3RenderTransferQueue) init(maxFrames, numBands, numChannels int) {
	q.frameSize, q.maxFrames = numBands*numChannels*SplitBandSize, maxFrames
	q.buffer = make([]float32, min(maxFrames, ec3RenderTransferQueueInitialFrames)*q.frameSize)
}

func (q *ec3RenderTransferQueue) capacity() int { return len(q.buffer) / q.frameSize }

func (q *ec3RenderTransferQueue) slot(index int) []float32 {
	n := q.frameSize
	return q.buffer[index*n : (index+1)*n : (index+1)*n]
}

// grow doubles the capacity (up to maxFrames), keeping the queued frames in
// order at the start of the new buffer.
func (q *ec3RenderTransferQueue) grow() {
	n := q.frameSize
	capacity := q.capacity()
	buffer := make([]float32, min(q.maxFrames, 2*capacity)*n)
	for i := 0; i < q.numElements; i++ {
		copy(buffer[i*n:(i+1)*n], q.slot((q.nextReadIndex+i)%capacity))
	}
	q.buffer = buffer
	q.nextReadIndex = 0
	q.nextWriteIndex = q.numElements
}

func (q *ec3RenderTransferQueue) Insert(input Frame) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.numElements == q.capacity() {
		if q.numElements == q.maxFrames {
			return false
		}
		q.grow()
	}
	copy(q.slot(q.nextWriteIndex), input.data)
	q.numElements++
	q.nextWriteIndex++
	if q.nextWriteIndex == q.capacity() {
		q.nextWriteIndex = 0
	}
	return true
}

func (q *ec3RenderTransferQueue) Remove(output Frame) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.numElements == 0 {
		return false
	}
	copy(output.data, q.slot(q.nextReadIndex))
	q.numElements--
	q.nextReadIndex++
	if q.nextReadIndex == q.capacity() {
		q.nextReadIndex = 0
	}
	return true
}

// ec3RenderWriter is EchoCanceller3::RenderWriter: it holds the split-band
// render frame that goes into the transfer queue, optionally high-pass
// filtering its lowest band first.
type ec3RenderWriter struct {
	hasHighPassFilter     bool
	highPassFilter        HighPassFilter
	renderQueueInputFrame Frame
}

// init sets up w; the frame keeps its samples in storage.
func (w *ec3RenderWriter) init(config *EchoCanceller3Config, numBands, numChannels int, storage []float32) {
	*w = ec3RenderWriter{renderQueueInputFrame: ec3FrameOver(numBands, numChannels, storage)}
	if config.Filter.HighPassFilterEchoReference {
		w.hasHighPassFilter = true
		w.highPassFilter.init(16000, numChannels)
	}
}

// Insert moves renderQueueInputFrame into the transfer queue.
func (w *ec3RenderWriter) Insert(renderTransferQueue *ec3RenderTransferQueue) {
	if w.hasHighPassFilter {
		for channel := 0; channel < w.renderQueueInputFrame.numChannels; channel++ {
			w.highPassFilter.ProcessChannel(channel, w.renderQueueInputFrame.View(0, channel))
		}
	}

	_ = renderTransferQueue.Insert(w.renderQueueInputFrame)
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
// The canceller owns the frames it works on: the caller fills RenderFrame
// and calls AnalyzeRenderFrame, and fills CaptureFrame and calls
// ProcessCaptureFrame, which processes it in place. AnalyzeRender and
// ProcessCapture do the same for frames of the caller, which they copy.
//
// The render-side methods (RenderFrame, AnalyzeRenderFrame, AnalyzeRender)
// may be called concurrently with the other methods; the other methods must
// not be called concurrently with each other.
//
// The frames, blocks and framing buffers are views of one pointer-free
// array, and the framing components and the block processor are held by
// value, so the garbage collector follows few pointers per canceller.
type EchoCanceller3 struct {
	// The fields that hold pointers come first, the block processor (whose
	// own come first too) after them and the pointer-free state last: the
	// garbage collector reads a canceller up to its last pointer.
	renderWriter             ec3RenderWriter
	linearOutputFramer       BlockFramer
	outputFramer             BlockFramer
	captureBlocker           FrameBlocker
	renderBlocker            FrameBlocker
	renderTransferQueue      ec3RenderTransferQueue
	renderQueueOutputFrame   Frame
	captureFrame             Frame
	linearOutputFrame        Frame // [1][channel][SplitBandSize]
	renderBlock              Block
	linearOutputBlock        Block
	captureBlock             Block
	blockDelayBuffer         BlockDelayBuffer
	blockProcessorForTesting *BlockProcessor // set by SetBlockProcessorForTesting
	blockProcessor           BlockProcessor

	config                      EchoCanceller3Config
	sampleRateHz                int
	numBands                    int
	numRenderInputChannels      int
	numRenderChannelsToAec      int
	numCaptureChannels          int
	configSelector              ConfigSelector
	multichannelContentDetector MultiChannelContentDetector
	hasLinearOutput             bool
	saturatedMicrophoneSignal   bool
	hasBlockDelayBuffer         bool
}

// NewEchoCanceller3 creates an echo canceller. multichannelConfig is the
// optional config used once proper multichannel render content is detected
// (nil corresponds to std::nullopt).
func NewEchoCanceller3(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels int) *EchoCanceller3 {
	e := &EchoCanceller3{}
	e.Init(config, multichannelConfig, sampleRateHz, numRenderChannels, numCaptureChannels)
	return e
}

// Init sets up e, which must be the zero value, as NewEchoCanceller3 does. It
// serves an EchoCanceller3 embedded by value; e must not be copied
// afterwards.
func (e *EchoCanceller3) Init(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels int) {
	if !ValidFullBandRate(sampleRateHz) {
		panic("aec3: invalid sample rate for EchoCanceller3")
	}
	e.config = AdjustConfig(config)
	e.sampleRateHz = sampleRateHz
	e.numBands = NumBandsForRate(sampleRateHz)
	e.numRenderInputChannels = numRenderChannels
	e.numCaptureChannels = numCaptureChannels
	e.configSelector.init(&e.config, multichannelConfig, e.numRenderInputChannels)
	active := e.configSelector.ActiveConfig()
	e.multichannelContentDetector.init(
		active.MultiChannel.DetectStereoContent, e.numRenderInputChannels,
		active.MultiChannel.StereoDetectionThreshold,
		active.MultiChannel.StereoDetectionTimeoutThresholdSeconds,
		active.MultiChannel.StereoDetectionHysteresisSeconds)
	e.hasLinearOutput = active.Filter.ExportLinearAecOutput

	// One backing array for every sample buffer but the transfer queue. The
	// render frame, which the render side writes, comes last.
	renderViews := e.numBands * e.numRenderInputChannels
	captureViews := e.numBands * e.numCaptureChannels
	n := renderViews*(2*SplitBandSize+2*BlockSize) + captureViews*(SplitBandSize+3*BlockSize)
	if e.hasLinearOutput {
		n += e.numCaptureChannels * (SplitBandSize + 2*BlockSize)
	}
	storage := make([]float32, n)
	take := func(n int) []float32 {
		s := storage[:n:n]
		storage = storage[n:]
		return s
	}

	e.outputFramer.init(e.numBands, e.numCaptureChannels, take(captureViews*BlockSize))
	e.captureBlocker.init(e.numBands, e.numCaptureChannels, take(captureViews*BlockSize))
	e.renderTransferQueue.init(RenderTransferQueueSizeFrames, e.numBands, e.numRenderInputChannels)
	e.renderQueueOutputFrame = ec3FrameOver(e.numBands, e.numRenderInputChannels, take(renderViews*SplitBandSize))
	e.captureFrame = ec3FrameOver(e.numBands, e.numCaptureChannels, take(captureViews*SplitBandSize))
	e.renderBlock = ec3BlockOver(e.numBands, e.numRenderInputChannels, take(renderViews*BlockSize))
	e.captureBlock = ec3BlockOver(e.numBands, e.numCaptureChannels, take(captureViews*BlockSize))
	// The render blocker serves numRenderChannelsToAec channels, at most
	// all of them.
	e.renderBlocker.init(e.numBands, e.numRenderInputChannels, take(renderViews*BlockSize))

	if active.Delay.FixedCaptureDelaySamples > 0 {
		e.hasBlockDelayBuffer = true
		e.blockDelayBuffer.init(e.numCaptureChannels, e.numBands, SplitBandSize,
			e.config.Delay.FixedCaptureDelaySamples)
	}

	if e.hasLinearOutput {
		e.linearOutputFramer.init(1, e.numCaptureChannels, take(e.numCaptureChannels*BlockSize))
		e.linearOutputBlock = ec3BlockOver(1, e.numCaptureChannels, take(e.numCaptureChannels*BlockSize))
		e.linearOutputFrame = ec3FrameOver(1, e.numCaptureChannels, take(e.numCaptureChannels*SplitBandSize))
	}

	e.renderWriter.init(active, e.numBands, e.numRenderInputChannels, take(renderViews*SplitBandSize))

	e.initialize()
}

// ec3BlockOver returns a block that keeps its samples in storage.
func ec3BlockOver(numBands, numChannels int, storage []float32) Block {
	n := numBands * numChannels * BlockSize
	return Block{numBands: numBands, numChannels: numChannels, data: storage[:n:n]}
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

	// An empty blocker over the same storage.
	e.renderBlocker.init(e.numBands, e.numRenderChannelsToAec, e.renderBlocker.buffer[:cap(e.renderBlocker.buffer)])

	e.blockProcessor.init(e.configSelector.ActiveConfig(), e.sampleRateHz,
		e.numRenderChannelsToAec, e.numCaptureChannels)
}

// processor returns the block processor in use.
func (e *EchoCanceller3) processor() *BlockProcessor {
	if e.blockProcessorForTesting != nil {
		return e.blockProcessorForTesting
	}
	return &e.blockProcessor
}

// SetBlockProcessorForTesting replaces the internal block processor.
// ZeroGainSkipsForTesting returns the number of filter adaptations skipped
// because the gain was zero (BlockProcessor.ZeroGainSkipsForTesting).
func (e *EchoCanceller3) ZeroGainSkipsForTesting() uint64 {
	return e.blockProcessor.ZeroGainSkipsForTesting()
}

func (e *EchoCanceller3) SetBlockProcessorForTesting(blockProcessor *BlockProcessor) {
	e.blockProcessorForTesting = blockProcessor
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

// RenderFrame returns the split-band render frame that AnalyzeRenderFrame
// consumes. The caller fills every band and channel before each call.
func (e *EchoCanceller3) RenderFrame() Frame { return e.renderWriter.renderQueueInputFrame }

// AnalyzeRenderFrame analyzes and stores an internal copy of RenderFrame,
// whose contents are unspecified afterwards.
func (e *EchoCanceller3) AnalyzeRenderFrame() {
	e.renderWriter.Insert(&e.renderTransferQueue)
}

// AnalyzeRender analyzes and stores an internal copy of the split-band
// render signal ([band][channel][SplitBandSize]).
func (e *EchoCanceller3) AnalyzeRender(render [][][]float32) {
	if len(render) > 0 && len(render[0]) != e.numRenderInputChannels {
		panic("aec3: EchoCanceller3.AnalyzeRender channel count mismatch")
	}
	if e.numBands != len(render) {
		return
	}
	e.renderWriter.renderQueueInputFrame.CopyFrom(render)
	e.AnalyzeRenderFrame()
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

// AnalyzeCaptureSamples is AnalyzeCapture for a full-band capture signal
// whose channels are stored one after the other.
func (e *EchoCanceller3) AnalyzeCaptureSamples(capture []float32) {
	e.saturatedMicrophoneSignal = ec3DetectSaturation(capture)
}

// CaptureFrame returns the split-band capture frame that ProcessCaptureFrame
// processes in place.
func (e *EchoCanceller3) CaptureFrame() Frame { return e.captureFrame }

// LinearOutputFrame returns the frame ([1][channel][SplitBandSize], 16 kHz)
// that receives the linear filter output. It is the zero Frame without
// Filter.ExportLinearAecOutput.
func (e *EchoCanceller3) LinearOutputFrame() Frame { return e.linearOutputFrame }

// ProcessCapture removes the echo from the split-band capture signal
// ([band][channel][SplitBandSize]) in place. linearOutput, when non-nil,
// receives the linear filter output ([channel][SplitBandSize], 16 kHz); it
// requires Filter.ExportLinearAecOutput, and must then be passed on every
// call. levelChange flags an echo path gain change.
func (e *EchoCanceller3) ProcessCapture(capture [][][]float32, linearOutput [][]float32, levelChange bool) {
	if len(capture) != e.numBands || len(capture[0]) != e.numCaptureChannels {
		panic("aec3: EchoCanceller3.ProcessCapture capture shape mismatch")
	}
	e.captureFrame.CopyFrom(capture)
	e.ProcessCaptureFrame(linearOutput != nil, levelChange)
	e.captureFrame.CopyTo(capture)
	if linearOutput != nil {
		for channel := range linearOutput {
			copy(linearOutput[channel][:SplitBandSize], e.linearOutputFrame.View(0, channel))
		}
	}
}

// ProcessCaptureFrame removes the echo from CaptureFrame in place. With
// linearOutput, LinearOutputFrame receives the linear filter output; that
// requires Filter.ExportLinearAecOutput, and linearOutput must then be set on
// every call. levelChange flags an echo path gain change.
func (e *EchoCanceller3) ProcessCaptureFrame(linearOutput, levelChange bool) {
	if linearOutput && !e.hasLinearOutput {
		panic("aec3: linear AEC output requested without Filter.ExportLinearAecOutput")
	}

	// Optionally delay the capture signal.
	if e.configSelector.ActiveConfig().Delay.FixedCaptureDelaySamples > 0 {
		if !e.hasBlockDelayBuffer {
			panic("aec3: fixed capture delay enabled by a config switch")
		}
		e.blockDelayBuffer.DelaySignal(e.captureFrame)
	}

	e.emptyRenderQueue()

	e.processCaptureFrameContent(linearOutput, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 0)
	e.processCaptureFrameContent(linearOutput, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 1)
	e.processRemainingCaptureFrameContent(levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected())
}

// ec3DownmixRenderSubFrame prepares subframe subFrameIndex of the render
// frame for an AEC that runs on numChannelsToAec channels (the
// FillSubFrameView overload for the render frame): when that is fewer than
// the frame has, the AEC reads channel 0, after all channels are averaged
// into it when a proper downmix is needed.
func ec3DownmixRenderSubFrame(properDownmixNeeded bool, frame Frame, subFrameIndex, numChannelsToAec int) {
	frameNumChannels := frame.numChannels
	if frameNumChannels > numChannelsToAec && properDownmixNeeded {
		// When a proper downmix is needed (which is the case when proper
		// stereo is present in the echo reference signal but the echo
		// canceller does the processing in mono) downmix the echo
		// reference by averaging the channel content (otherwise
		// downmixing is done by selecting channel 0).
		for band := 0; band < frame.numBands; band++ {
			dst := frame.SubFrame(band, 0, subFrameIndex)
			for ch := 1; ch < frameNumChannels; ch++ {
				src := frame.SubFrame(band, ch, subFrameIndex)
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
}

// linearBlock returns the block that receives the linear filter output, nil
// without Filter.ExportLinearAecOutput.
func (e *EchoCanceller3) linearBlock() *Block {
	if !e.hasLinearOutput {
		return nil
	}
	return &e.linearOutputBlock
}

func (e *EchoCanceller3) processCaptureFrameContent(linearOutput, levelChange, aecReferenceIsDownmixedStereo bool,
	subFrameIndex int) {
	e.captureBlocker.InsertSubFrameAndExtractBlock(e.captureFrame, subFrameIndex, &e.captureBlock)
	e.processor().ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, e.linearBlock(), &e.captureBlock)
	e.outputFramer.InsertBlockAndExtractSubFrame(&e.captureBlock, e.captureFrame, subFrameIndex)

	if linearOutput {
		e.linearOutputFramer.InsertBlockAndExtractSubFrame(&e.linearOutputBlock, e.linearOutputFrame, subFrameIndex)
	}
}

func (e *EchoCanceller3) processRemainingCaptureFrameContent(levelChange, aecReferenceIsDownmixedStereo bool) {
	if !e.captureBlocker.IsBlockAvailable() {
		return
	}

	e.captureBlocker.ExtractBlock(&e.captureBlock)
	e.processor().ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, e.linearBlock(), &e.captureBlock)
	e.outputFramer.InsertBlock(&e.captureBlock)

	if e.hasLinearOutput {
		e.linearOutputFramer.InsertBlock(&e.linearOutputBlock)
	}
}

func (e *EchoCanceller3) bufferRenderFrameContent(properDownmixNeeded bool, subFrameIndex int) {
	ec3DownmixRenderSubFrame(properDownmixNeeded, e.renderQueueOutputFrame, subFrameIndex, e.numRenderChannelsToAec)
	e.renderBlocker.InsertSubFrameAndExtractBlock(e.renderQueueOutputFrame, subFrameIndex, &e.renderBlock)
	e.processor().BufferRender(&e.renderBlock)
}

func (e *EchoCanceller3) bufferRemainingRenderFrameContent() {
	if !e.renderBlocker.IsBlockAvailable() {
		return
	}
	e.renderBlocker.ExtractBlock(&e.renderBlock)
	e.processor().BufferRender(&e.renderBlock)
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
	return e.processor().GetMetrics()
}

// SetAudioBufferDelay provides an optional external estimate of the audio
// buffer delay.
func (e *EchoCanceller3) SetAudioBufferDelay(delayMs int) {
	e.processor().SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage specifies whether the capture output will be used,
// allowing some processing to be deactivated when it is not, for instance
// when the endpoint is muted.
func (e *EchoCanceller3) SetCaptureOutputUsage(captureOutputUsed bool) {
	e.processor().SetCaptureOutputUsage(captureOutputUsed)
}

// ActiveProcessing reports whether the echo canceller is processing; AEC3
// always is.
func (e *EchoCanceller3) ActiveProcessing() bool { return true }

// UpdateEchoLeakageStatus signals whether an external detector has detected
// echo leakage from the echo canceller. Once flagged, leakage should be
// unflagged when it no longer occurs.
func (e *EchoCanceller3) UpdateEchoLeakageStatus(leakageDetected bool) {
	e.processor().UpdateEchoLeakageStatus(leakageDetected)
}

// SampleRateHz returns the full-band sample rate.
func (e *EchoCanceller3) SampleRateHz() int { return e.sampleRateHz }

// NumBands returns the number of split bands.
func (e *EchoCanceller3) NumBands() int { return e.numBands }
