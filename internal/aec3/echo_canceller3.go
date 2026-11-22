package aec3

import "sync"

// SplitBandSize is the number of samples per band of a 10 ms frame
// (AudioBuffer::kSplitBandSize).
const SplitBandSize = 160

// AdjustConfig mirrors AdjustConfig of echo_canceller3.cc, which applies
// field trials to the config. With every trial at its default it returns a
// copy: a float parameter survives the round trip through double, and an
// override applies only when the parsed value differs from the current one.
func AdjustConfig(config *EchoCanceller3Config) EchoCanceller3Config {
	adjusted := *config

	// WebRTC-Aec3UseNearendReverbLen: the parsed values are the current
	// ones, applied only when both lie in (-1, 1).
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

// Frame is a 10 ms frame of split-band audio in FloatS16 scale, as
// AudioBuffer::split_bands exposes it: SplitBandSize samples per band and
// channel in one pointer-free array.
type Frame struct {
	numBands    int
	numChannels int
	data        []float32
}

// NewFrame returns a frame of zeros.
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

// Samples returns the samples of f, band-major.
func (f Frame) Samples() []float32 { return f.data }

// View returns the samples of one band and channel.
func (f Frame) View(band, channel int) []float32 {
	i := (band*f.numChannels + channel) * SplitBandSize
	return f.data[i : i+SplitBandSize : i+SplitBandSize]
}

// SubFrame returns the SubFrameLength samples of subframe subFrameIndex, 0
// or 1, of one band and channel.
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

// ec3RenderTransferQueue is the SwapQueue that carries render frames to the
// capture side. Insert copies a frame in and fails when maxFrames frames are
// queued; Remove copies the oldest frame out and fails when the queue is
// empty. The buffer doubles, up to maxFrames, when a render burst fills it.
// The queue is safe for one inserting and one removing goroutine.
type ec3RenderTransferQueue struct {
	buffer         []float32 // [frame][band][channel][SplitBandSize]
	mu             sync.Mutex
	frameSize      int // samples per frame
	maxFrames      int
	nextWriteIndex int
	nextReadIndex  int
	numElements    int
}

// ec3RenderTransferQueueInitialFrames is the initial capacity of the queue.
// Render and capture alternate in steady state, which queues one frame.
const ec3RenderTransferQueueInitialFrames = 4

// ec3RenderTransferQueueInitialSize returns the number of samples the queue
// starts with.
func ec3RenderTransferQueueInitialSize(maxFrames, numBands, numChannels int) int {
	return min(maxFrames, ec3RenderTransferQueueInitialFrames) * numBands * numChannels * SplitBandSize
}

// init sets up q over the ec3RenderTransferQueueInitialSize samples of
// buffer. With a nil buffer the queue starts with the samples the caller
// passes to insert and remove.
func (q *ec3RenderTransferQueue) init(maxFrames, numBands, numChannels int, buffer []float32) {
	q.frameSize, q.maxFrames = numBands*numChannels*SplitBandSize, maxFrames
	q.buffer = buffer
}

// buf returns the buffer of the queue, or initial before it has one.
func (q *ec3RenderTransferQueue) buf(initial []float32) []float32 {
	if q.buffer != nil {
		return q.buffer
	}
	return initial
}

func ec3QueueSlot(buffer []float32, n, index int) []float32 {
	return buffer[index*n : (index+1)*n : (index+1)*n]
}

// grow doubles the capacity, up to maxFrames, and moves the queued frames to
// the start of the new buffer in order.
func (q *ec3RenderTransferQueue) grow(old []float32) {
	n := q.frameSize
	capacity := len(old) / n
	buffer := make([]float32, min(q.maxFrames, 2*capacity)*n)
	for i := 0; i < q.numElements; i++ {
		copy(buffer[i*n:(i+1)*n], ec3QueueSlot(old, n, (q.nextReadIndex+i)%capacity))
	}
	q.buffer = buffer
	q.nextReadIndex = 0
	q.nextWriteIndex = q.numElements
}

func (q *ec3RenderTransferQueue) Insert(input Frame) bool { return q.insert(input, nil) }

func (q *ec3RenderTransferQueue) Remove(output Frame) bool { return q.remove(output, nil) }

// insert is Insert for a queue that starts with the frames of initial.
func (q *ec3RenderTransferQueue) insert(input Frame, initial []float32) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	buffer := q.buf(initial)
	if q.numElements == len(buffer)/q.frameSize {
		if q.numElements == q.maxFrames {
			return false
		}
		q.grow(buffer)
		buffer = q.buffer
	}
	copy(ec3QueueSlot(buffer, q.frameSize, q.nextWriteIndex), input.data)
	q.numElements++
	q.nextWriteIndex++
	if q.nextWriteIndex == len(buffer)/q.frameSize {
		q.nextWriteIndex = 0
	}
	return true
}

// remove is Remove for a queue that starts with the frames of initial.
func (q *ec3RenderTransferQueue) remove(output Frame, initial []float32) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.numElements == 0 {
		return false
	}
	buffer := q.buf(initial)
	copy(output.data, ec3QueueSlot(buffer, q.frameSize, q.nextReadIndex))
	q.numElements--
	q.nextReadIndex++
	if q.nextReadIndex == len(buffer)/q.frameSize {
		q.nextReadIndex = 0
	}
	return true
}

// ec3RenderWriter moves the render frame into the transfer queue, after
// high-pass filtering its lowest band if configured
// (EchoCanceller3::RenderWriter).
//
// The filter has one section per channel. Those of more than inlineChannels
// channels are the canceller's (Ec3Core.hpMore).
type ec3RenderWriter struct {
	hasHighPassFilter bool
	sections          [inlineChannels]BiQuad
}

// init sets up w. It returns the sections of more than inlineChannels
// channels, or nil.
func (w *ec3RenderWriter) init(config *EchoCanceller3Config, numChannels int) (more []BiQuad) {
	*w = ec3RenderWriter{}
	if config.Filter.HighPassFilterEchoReference {
		w.hasHighPassFilter = true
		more = moreChannels[BiQuad](numChannels)
		sections := channelState(&w.sections, numChannels, more)
		c := ec3ChooseHighPassCoefficients(16000)[0]
		for k := range sections {
			sections[k] = BiQuad{Coefficients: c}
		}
	}
	return more
}

// Insert moves renderQueueInputFrame into the transfer queue. more is the
// result of init.
func (w *ec3RenderWriter) Insert(renderTransferQueue *ec3RenderTransferQueue, renderQueueInputFrame Frame, more []BiQuad,
	initial []float32) {
	if w.hasHighPassFilter {
		sections := channelState(&w.sections, renderQueueInputFrame.numChannels, more)
		for channel := range sections {
			x := renderQueueInputFrame.View(0, channel)
			dlyApplyBiQuadCascade(x, x, sections[channel:channel+1])
		}
	}

	_ = renderTransferQueue.insert(renderQueueInputFrame, initial)
}

// EchoCanceller3 is the echo canceller (echo_canceller3.h). It takes 10 ms
// frames of split-band audio in FloatS16 scale, feeds the BlockProcessor
// blocks of 64 samples and absorbs jitter between the render and the capture
// calls. A frame has NumBandsForRate(sampleRateHz) bands.
//
// The canceller owns the frames it works on. The caller fills RenderFrame
// and calls AnalyzeRenderFrame, fills CaptureFrame and calls
// ProcessCaptureFrame. AnalyzeRender and ProcessCapture copy frames of the
// caller instead.
//
// RenderFrame, AnalyzeRenderFrame and AnalyzeRender may run concurrently
// with the other methods. The other methods must not run concurrently with
// each other.
//
// An EchoCanceller3 is a view of an Ec3Core. NewEchoCanceller3 returns one
// that stays bound; Ec3Core.Bind binds one for a call.
type EchoCanceller3 struct {
	*ec3State
	core                     *Ec3Core
	storage                  []float32 // every sample buffer but the transfer queue
	renderTransferQueue      *ec3RenderTransferQueue
	hpMore                   []BiQuad               // of the render writer
	alloc                    func(ArenaSizes) Arena // what init hands to the first adopt
	blockProcessorForTesting *BlockProcessor        // set by SetBlockProcessorForTesting
	blockProcessor           BlockProcessor
}

// Ec3Core is what a canceller keeps between calls: a few pointer words, then
// the pointer-free state of every component. The garbage collector reads the
// pointer words only.
//
// BindRender and the render-side methods of its view may run concurrently
// with the capture side. They read arena and hpMore, which are set once, and
// the queue.
type Ec3Core struct {
	// arena holds the storage of the canceller, then the arrays of the block
	// processor it started with.
	arena Arena
	// bpArena holds the arrays of a block processor that was set up again
	// and does not fit arena, or nil.
	bpArena Arena
	queue   ec3RenderTransferQueue
	hpMore  []BiQuad
	bpExtra *bpExtra
	st      ec3State
}

// Arena is one object that holds the arrays of a canceller.
type Arena interface {
	// Views returns the arrays. Every call returns the same ones.
	Views() ArenaViews
	// Alloc returns another arena that holds at least the given sizes.
	Alloc(ArenaSizes) Arena
}

// ArenaViews holds the arrays of an Arena, one per element type.
type ArenaViews struct {
	F32  []float32
	Fft  []FftData
	Spec [][FftLengthBy2Plus1]float32
	I32  []int32
}

// ArenaSizes holds the lengths a canceller needs of the arrays of an Arena.
type ArenaSizes struct{ F32, Fft, Spec, I32 int }

// sliceArena is an Arena of four allocations.
type sliceArena struct{ v ArenaViews }

func (a *sliceArena) Views() ArenaViews { return a.v }

func (a *sliceArena) Alloc(n ArenaSizes) Arena { return NewSliceArena(n) }

// NewSliceArena returns an Arena whose arrays are separate allocations.
func NewSliceArena(n ArenaSizes) Arena {
	return &sliceArena{v: ArenaViews{
		F32:  make([]float32, n.F32),
		Fft:  make([]FftData, n.Fft),
		Spec: make([][FftLengthBy2Plus1]float32, n.Spec),
		I32:  make([]int32, n.I32),
	}}
}

// Init sets up c, which is the zero value. The storage has room for
// extraSamples samples of the caller (EchoCanceller3.ExtraSamples). alloc
// returns an arena of at least the given sizes; nil is NewSliceArena.
func (c *Ec3Core) Init(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels, extraSamples int, alloc func(ArenaSizes) Arena) {
	var e EchoCanceller3
	c.BindRender(&e)
	e.blockProcessor.bind(&c.st.blockProcessor, bindAlone)
	if alloc == nil {
		alloc = NewSliceArena
	}
	e.init(config, multichannelConfig, sampleRateHz, numRenderChannels, numCaptureChannels, extraSamples, alloc)
}

// Arena returns the arena Init allocated.
func (c *Ec3Core) Arena() Arena { return c.arena }

// storage returns the samples of the canceller itself.
func (c *Ec3Core) storage() []float32 {
	if c.arena == nil {
		return nil
	}
	n := c.st.offsets.size
	return c.arena.Views().F32[:n:n]
}

// bp returns the arenas of the block processor.
func (c *Ec3Core) bp() bpArenas {
	if c.bpArena != nil {
		v := c.bpArena.Views()
		return bpArenas{f32: v.F32, fft: v.Fft, spec: v.Spec, i32: v.I32}
	}
	v := c.arena.Views()
	return bpArenas{f32: v.F32[c.st.offsets.size:], fft: v.Fft, spec: v.Spec, i32: v.I32}
}

// BindRender makes e the view of c for RenderFrame, AnalyzeRenderFrame and
// AnalyzeRender.
func (c *Ec3Core) BindRender(e *EchoCanceller3) {
	e.ec3State = &c.st
	e.core = c
	e.storage = c.storage()
	e.renderTransferQueue = &c.queue
	e.hpMore = c.hpMore
}

// Bind makes e the view of c for the capture-side methods.
func (c *Ec3Core) Bind(e *EchoCanceller3) {
	v := c.arena.Views()
	n := c.st.offsets.size
	e.ec3State = &c.st
	e.core = c
	e.storage = v.F32[:n:n]
	e.renderTransferQueue = &c.queue
	e.hpMore = c.hpMore
	b := binder{mode: bindView, a: bpArenas{f32: v.F32[n:], fft: v.Fft, spec: v.Spec, i32: v.I32}, x: c.bpExtra}
	if c.bpArena != nil {
		b.a = c.bp()
	}
	if b.x == nil {
		b.x = &bpNoExtra
	}
	e.blockProcessor.bind(&c.st.blockProcessor, &b)
}

// measure records the lengths of the slices the init of p allocated in the
// state and returns their sums per arena.
func (c *Ec3Core) measure(p *BlockProcessor) arenaSizes {
	b := binder{mode: bindMeasure, notView: -1, x: &bpNoExtra}
	p.bind(&c.st.blockProcessor, &b)
	return b.n
}

// adopt moves the slices the init of p allocated into the arena of c and its
// pointers into the extras. The first call allocates the arena with alloc. A
// later one, when stereo content starts or ends and the block processor is
// set up at another size, uses the arena again if the slices fit and an
// arena of their own otherwise.
func (c *Ec3Core) adopt(p *BlockProcessor, alloc func(ArenaSizes) Arena) {
	n := c.measure(p)
	need := ArenaSizes{F32: int(n.f32), Fft: int(n.fft), Spec: int(n.spec), I32: int(n.i32)}
	if c.arena == nil {
		need.F32 += c.st.offsets.size
		c.arena = alloc(need)
	} else {
		c.bpArena = nil
		if a := c.bp(); len(a.f32) < need.F32 || len(a.fft) < need.Fft || len(a.spec) < need.Spec || len(a.i32) < need.I32 {
			c.bpArena = c.arena.Alloc(need)
		}
	}
	b := binder{mode: bindMove, notView: -1, a: c.bp(), x: &bpExtra{}}
	p.bind(&c.st.blockProcessor, &b)
	c.bpExtra = nil
	if b.hasExtra {
		c.bpExtra = b.x
	}
}

// ec3State is the state of an EchoCanceller3 except its pointers.
type ec3State struct {
	renderWriter   ec3RenderWriter
	blockProcessor blockProcessorState

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

	// The buffers in storage, and what the framing components keep besides
	// their samples.
	offsets                ec3Offsets
	outputFramerFill       int
	captureBlockerFill     int
	renderBlockerFill      int
	linearOutputFramerFill int
	blockDelay             int
	blockDelayLastInsert   int
}

// ec3Offsets holds the start of each buffer of an EchoCanceller3 in its
// storage.
type ec3Offsets struct {
	outputFramer           int // [band][capture channel][BlockSize]
	captureBlocker         int // [band][capture channel][BlockSize]
	renderQueueOutputFrame int // [band][render channel][SplitBandSize]
	captureFrame           int // [band][capture channel][SplitBandSize]
	renderBlock            int // [band][render channel][BlockSize]
	captureBlock           int // [band][capture channel][BlockSize]
	renderBlocker          int // [band][render channel][BlockSize]
	blockDelayBuffer       int // [capture channel][band][delay]
	linearOutputFramer     int // [capture channel][BlockSize]
	linearOutputBlock      int // [capture channel][BlockSize]
	linearOutputFrame      int // [capture channel][SplitBandSize]
	renderQueueInputFrame  int // [band][render channel][SplitBandSize]
	extra                  int // the caller's samples
	queue, queueSize       int // the first frames of the render transfer queue
	size                   int // of the storage
}

// The frames, blocks and framing components of e are views over its storage.
// A framing component is a copy: its fill is stored back after use.

func (e *EchoCanceller3) renderQueueInputFrame() Frame {
	return ec3FrameOver(e.numBands, e.numRenderInputChannels, e.storage[e.offsets.renderQueueInputFrame:])
}

func (e *EchoCanceller3) renderQueueOutputFrame() Frame {
	return ec3FrameOver(e.numBands, e.numRenderInputChannels, e.storage[e.offsets.renderQueueOutputFrame:])
}

func (e *EchoCanceller3) captureFrame() Frame {
	return ec3FrameOver(e.numBands, e.numCaptureChannels, e.storage[e.offsets.captureFrame:])
}

func (e *EchoCanceller3) linearOutputFrame() Frame {
	if !e.hasLinearOutput {
		return Frame{}
	}
	return ec3FrameOver(1, e.numCaptureChannels, e.storage[e.offsets.linearOutputFrame:])
}

func (e *EchoCanceller3) renderBlock() Block {
	return ec3BlockOver(e.numBands, e.numRenderChannelsToAec, e.storage[e.offsets.renderBlock:])
}

func (e *EchoCanceller3) captureBlock() Block {
	return ec3BlockOver(e.numBands, e.numCaptureChannels, e.storage[e.offsets.captureBlock:])
}

// linearOutputBlock returns the block that receives the output of the linear
// filter. ok is false without Filter.ExportLinearAecOutput.
func (e *EchoCanceller3) linearOutputBlock() (block Block, ok bool) {
	if !e.hasLinearOutput {
		return Block{}, false
	}
	return ec3BlockOver(1, e.numCaptureChannels, e.storage[e.offsets.linearOutputBlock:]), true
}

func (e *EchoCanceller3) outputFramer() BlockFramer {
	return BlockFramer{numBands: e.numBands, numChannels: e.numCaptureChannels, fill: e.outputFramerFill,
		buffer: e.storage[e.offsets.outputFramer:][:e.numBands*e.numCaptureChannels*BlockSize]}
}

func (e *EchoCanceller3) linearOutputFramer() BlockFramer {
	return BlockFramer{numBands: 1, numChannels: e.numCaptureChannels, fill: e.linearOutputFramerFill,
		buffer: e.storage[e.offsets.linearOutputFramer:][:e.numCaptureChannels*BlockSize]}
}

func (e *EchoCanceller3) captureBlocker() FrameBlocker {
	return FrameBlocker{numBands: e.numBands, numChannels: e.numCaptureChannels, fill: e.captureBlockerFill,
		buffer: e.storage[e.offsets.captureBlocker:][:e.numBands*e.numCaptureChannels*BlockSize]}
}

// renderBlocker serves numRenderChannelsToAec channels, at most all of them.
func (e *EchoCanceller3) renderBlocker() FrameBlocker {
	return FrameBlocker{numBands: e.numBands, numChannels: e.numRenderChannelsToAec, fill: e.renderBlockerFill,
		buffer: e.storage[e.offsets.renderBlocker:][:e.numBands*e.numRenderChannelsToAec*BlockSize]}
}

func (e *EchoCanceller3) blockDelayBuffer() BlockDelayBuffer {
	return BlockDelayBuffer{numChannels: e.numCaptureChannels, numBands: e.numBands, frameLength: SplitBandSize,
		delay: e.blockDelay, lastInsert: e.blockDelayLastInsert,
		buf: e.storage[e.offsets.blockDelayBuffer:][:e.numCaptureChannels*e.numBands*e.blockDelay]}
}

// NewEchoCanceller3 returns an echo canceller. multichannelConfig is the
// config for multichannel render content, or nil for std::nullopt.
func NewEchoCanceller3(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels int) *EchoCanceller3 {
	c := &Ec3Core{}
	c.Init(config, multichannelConfig, sampleRateHz, numRenderChannels, numCaptureChannels, 0, nil)
	e := &EchoCanceller3{}
	c.Bind(e)
	return e
}

// init sets up the core of e, which is the zero value.
func (e *EchoCanceller3) init(config, multichannelConfig *EchoCanceller3Config, sampleRateHz,
	numRenderChannels, numCaptureChannels, extraSamples int, alloc func(ArenaSizes) Arena) {
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

	// One array for every sample buffer except the transfer queue. The
	// render frame, which the render side writes, comes last.
	renderViews := e.numBands * e.numRenderInputChannels
	captureViews := e.numBands * e.numCaptureChannels
	n := 0
	take := func(size int) int {
		offset := n
		n += size
		return offset
	}
	e.offsets.outputFramer = take(captureViews * BlockSize)
	e.offsets.captureBlocker = take(captureViews * BlockSize)
	e.offsets.renderQueueOutputFrame = take(renderViews * SplitBandSize)
	e.offsets.captureFrame = take(captureViews * SplitBandSize)
	e.offsets.renderBlock = take(renderViews * BlockSize)
	e.offsets.captureBlock = take(captureViews * BlockSize)
	e.offsets.renderBlocker = take(renderViews * BlockSize)
	if active.Delay.FixedCaptureDelaySamples > 0 {
		e.hasBlockDelayBuffer = true
		e.blockDelay = e.config.Delay.FixedCaptureDelaySamples
		e.offsets.blockDelayBuffer = take(captureViews * e.blockDelay)
	}
	if e.hasLinearOutput {
		e.offsets.linearOutputFramer = take(e.numCaptureChannels * BlockSize)
		e.offsets.linearOutputBlock = take(e.numCaptureChannels * BlockSize)
		e.offsets.linearOutputFrame = take(e.numCaptureChannels * SplitBandSize)
	}
	queueSize := ec3RenderTransferQueueInitialSize(RenderTransferQueueSizeFrames, e.numBands, e.numRenderInputChannels)
	e.offsets.queue, e.offsets.queueSize = take(queueSize), queueSize
	e.offsets.renderQueueInputFrame = take(renderViews * SplitBandSize)
	e.offsets.extra = take(extraSamples)
	e.offsets.size = n

	e.alloc = alloc

	// initialize allocates the arena.
	e.initialize()

	// The framers start with BlockSize zeros, the blockers empty.
	e.outputFramerFill = BlockSize
	e.linearOutputFramerFill = BlockSize
	e.renderTransferQueue.init(RenderTransferQueueSizeFrames, e.numBands, e.numRenderInputChannels, nil)
	e.hpMore = e.renderWriter.init(active, e.numRenderInputChannels)
	e.core.hpMore = e.hpMore
}

// ec3BlockOver returns a block that keeps its samples in storage.
func ec3BlockOver(numBands, numChannels int, storage []float32) Block {
	n := numBands * numChannels * BlockSize
	return Block{numBands: numBands, numChannels: numChannels, data: storage[:n:n]}
}

// initialize sets up the fields that depend on the active config, at
// creation and when the config changes.
func (e *EchoCanceller3) initialize() {
	e.numRenderChannelsToAec = 1
	if e.multichannelContentDetector.IsProperMultiChannelContentDetected() {
		e.numRenderChannelsToAec = e.numRenderInputChannels
	}

	e.configSelector.Update(e.multichannelContentDetector.IsProperMultiChannelContentDetected())

	e.blockProcessor.init(e.configSelector.ActiveConfig(), e.sampleRateHz,
		e.numRenderChannelsToAec, e.numCaptureChannels)
	e.core.adopt(&e.blockProcessor, e.alloc)
	e.storage = e.core.storage()

	renderBlock := e.renderBlock()
	clear(renderBlock.data)

	// An empty blocker over the same storage.
	e.renderBlockerFill = 0
}

func (e *EchoCanceller3) processor() *BlockProcessor {
	if e.blockProcessorForTesting != nil {
		return e.blockProcessorForTesting
	}
	return &e.blockProcessor
}

// ZeroGainSkipsForTesting returns BlockProcessor.ZeroGainSkipsForTesting of
// the block processor.
func (e *EchoCanceller3) ZeroGainSkipsForTesting() uint64 {
	return e.blockProcessor.ZeroGainSkipsForTesting()
}

// SetBlockProcessorForTesting replaces the block processor of e.
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

// queueInitial returns the frames the render transfer queue starts with.
func (e *EchoCanceller3) queueInitial() []float32 {
	i, n := e.offsets.queue, e.offsets.queueSize
	return e.storage[i : i+n : i+n]
}

// ExtraSamples returns the samples Ec3Core.Init set aside for the caller.
func (e *EchoCanceller3) ExtraSamples() []float32 { return e.storage[e.offsets.extra:] }

// RenderFrame returns the render frame AnalyzeRenderFrame consumes. The
// caller fills every band and channel before each call.
func (e *EchoCanceller3) RenderFrame() Frame { return e.renderQueueInputFrame() }

// AnalyzeRenderFrame queues a copy of RenderFrame for the capture side. The
// contents of RenderFrame are unspecified afterwards.
func (e *EchoCanceller3) AnalyzeRenderFrame() {
	e.renderWriter.Insert(e.renderTransferQueue, e.renderQueueInputFrame(), e.hpMore, e.queueInitial())
}

// AnalyzeRender queues a copy of the split-band render signal,
// render[band][channel], for the capture side.
func (e *EchoCanceller3) AnalyzeRender(render [][][]float32) {
	if len(render) > 0 && len(render[0]) != e.numRenderInputChannels {
		panic("aec3: EchoCanceller3.AnalyzeRender channel count mismatch")
	}
	if e.numBands != len(render) {
		return
	}
	e.renderQueueInputFrame().CopyFrom(render)
	e.AnalyzeRenderFrame()
}

// AnalyzeCapture detects saturation in the full-band capture signal,
// capture[channel].
func (e *EchoCanceller3) AnalyzeCapture(capture [][]float32) {
	e.saturatedMicrophoneSignal = false
	for channel := range capture {
		e.saturatedMicrophoneSignal = e.saturatedMicrophoneSignal || ec3DetectSaturation(capture[channel])
		if e.saturatedMicrophoneSignal {
			break
		}
	}
}

// AnalyzeCaptureSamples is AnalyzeCapture for channels stored one after the
// other.
func (e *EchoCanceller3) AnalyzeCaptureSamples(capture []float32) {
	e.saturatedMicrophoneSignal = ec3DetectSaturation(capture)
}

// CaptureFrame returns the capture frame ProcessCaptureFrame processes in
// place.
func (e *EchoCanceller3) CaptureFrame() Frame { return e.captureFrame() }

// LinearOutputFrame returns the one-band frame at 16 kHz that receives the
// output of the linear filter, or the zero Frame without
// Filter.ExportLinearAecOutput.
func (e *EchoCanceller3) LinearOutputFrame() Frame { return e.linearOutputFrame() }

// ProcessCapture removes the echo from the split-band capture signal,
// capture[band][channel], in place. A linearOutput that is not nil receives
// the output of the linear filter at 16 kHz; it needs
// Filter.ExportLinearAecOutput and is then passed on every call. levelChange
// reports a gain change of the echo path.
func (e *EchoCanceller3) ProcessCapture(capture [][][]float32, linearOutput [][]float32, levelChange bool) {
	if len(capture) != e.numBands || len(capture[0]) != e.numCaptureChannels {
		panic("aec3: EchoCanceller3.ProcessCapture capture shape mismatch")
	}
	captureFrame := e.captureFrame()
	captureFrame.CopyFrom(capture)
	e.ProcessCaptureFrame(linearOutput != nil, levelChange)
	captureFrame.CopyTo(capture)
	if linearOutput != nil {
		linearOutputFrame := e.linearOutputFrame()
		for channel := range linearOutput {
			copy(linearOutput[channel][:SplitBandSize], linearOutputFrame.View(0, channel))
		}
	}
}

// ProcessCaptureFrame removes the echo from CaptureFrame in place. With
// linearOutput, LinearOutputFrame receives the output of the linear filter;
// that needs Filter.ExportLinearAecOutput and is then set on every call.
// levelChange reports a gain change of the echo path.
func (e *EchoCanceller3) ProcessCaptureFrame(linearOutput, levelChange bool) {
	if linearOutput && !e.hasLinearOutput {
		panic("aec3: linear AEC output requested without Filter.ExportLinearAecOutput")
	}

	if e.configSelector.ActiveConfig().Delay.FixedCaptureDelaySamples > 0 {
		if !e.hasBlockDelayBuffer {
			panic("aec3: fixed capture delay enabled by a config switch")
		}
		blockDelayBuffer := e.blockDelayBuffer()
		blockDelayBuffer.DelaySignal(e.captureFrame())
		e.blockDelayLastInsert = blockDelayBuffer.lastInsert
	}

	e.emptyRenderQueue()

	e.processCaptureFrameContent(linearOutput, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 0)
	e.processCaptureFrameContent(linearOutput, levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 1)
	e.processRemainingCaptureFrameContent(levelChange,
		e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected())
}

// ec3DownmixRenderSubFrame prepares a subframe of the render frame for an
// AEC that runs on numChannelsToAec channels. With fewer than the frame has,
// the AEC reads channel 0, which first takes the average of all channels if
// properDownmixNeeded.
func ec3DownmixRenderSubFrame(properDownmixNeeded bool, frame Frame, subFrameIndex, numChannelsToAec int) {
	frameNumChannels := frame.numChannels
	if frameNumChannels > numChannelsToAec && properDownmixNeeded {
		// Stereo content that the AEC processes in mono is averaged into
		// channel 0; otherwise channel 0 is the downmix.
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

func (e *EchoCanceller3) processCaptureFrameContent(linearOutput, levelChange, aecReferenceIsDownmixedStereo bool,
	subFrameIndex int) {
	captureFrame, captureBlock := e.captureFrame(), e.captureBlock()
	linearOutputBlock, hasLinearBlock := e.linearOutputBlock()
	var linearBlock *Block
	if hasLinearBlock {
		linearBlock = &linearOutputBlock
	}
	captureBlocker := e.captureBlocker()
	captureBlocker.InsertSubFrameAndExtractBlock(captureFrame, subFrameIndex, &captureBlock)
	e.captureBlockerFill = captureBlocker.fill
	e.processor().ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, linearBlock, &captureBlock)
	outputFramer := e.outputFramer()
	outputFramer.InsertBlockAndExtractSubFrame(&captureBlock, captureFrame, subFrameIndex)
	e.outputFramerFill = outputFramer.fill

	if linearOutput {
		linearOutputFramer := e.linearOutputFramer()
		linearOutputFramer.InsertBlockAndExtractSubFrame(&linearOutputBlock, e.linearOutputFrame(), subFrameIndex)
		e.linearOutputFramerFill = linearOutputFramer.fill
	}
}

func (e *EchoCanceller3) processRemainingCaptureFrameContent(levelChange, aecReferenceIsDownmixedStereo bool) {
	captureBlocker := e.captureBlocker()
	if !captureBlocker.IsBlockAvailable() {
		return
	}

	captureBlock := e.captureBlock()
	linearOutputBlock, hasLinearBlock := e.linearOutputBlock()
	var linearBlock *Block
	if hasLinearBlock {
		linearBlock = &linearOutputBlock
	}
	captureBlocker.ExtractBlock(&captureBlock)
	e.captureBlockerFill = captureBlocker.fill
	e.processor().ProcessCapture(levelChange || aecReferenceIsDownmixedStereo,
		e.saturatedMicrophoneSignal, linearBlock, &captureBlock)
	outputFramer := e.outputFramer()
	outputFramer.InsertBlock(&captureBlock)
	e.outputFramerFill = outputFramer.fill

	if e.hasLinearOutput {
		linearOutputFramer := e.linearOutputFramer()
		linearOutputFramer.InsertBlock(&linearOutputBlock)
		e.linearOutputFramerFill = linearOutputFramer.fill
	}
}

func (e *EchoCanceller3) bufferRenderFrameContent(properDownmixNeeded bool, subFrameIndex int) {
	renderQueueOutputFrame, renderBlock := e.renderQueueOutputFrame(), e.renderBlock()
	ec3DownmixRenderSubFrame(properDownmixNeeded, renderQueueOutputFrame, subFrameIndex, e.numRenderChannelsToAec)
	renderBlocker := e.renderBlocker()
	renderBlocker.InsertSubFrameAndExtractBlock(renderQueueOutputFrame, subFrameIndex, &renderBlock)
	e.renderBlockerFill = renderBlocker.fill
	e.processor().BufferRender(&renderBlock)
}

func (e *EchoCanceller3) bufferRemainingRenderFrameContent() {
	renderBlocker := e.renderBlocker()
	if !renderBlocker.IsBlockAvailable() {
		return
	}
	renderBlock := e.renderBlock()
	renderBlocker.ExtractBlock(&renderBlock)
	e.renderBlockerFill = renderBlocker.fill
	e.processor().BufferRender(&renderBlock)
}

func (e *EchoCanceller3) emptyRenderQueue() {
	renderQueueOutputFrame := e.renderQueueOutputFrame()
	queueInitial := e.queueInitial()
	frameToBuffer := e.renderTransferQueue.remove(renderQueueOutputFrame, queueInitial)
	for frameToBuffer {
		if e.multichannelContentDetector.UpdateDetection(renderQueueOutputFrame) {
			// Stereo content reinitializes the AEC.
			e.initialize()
		}

		e.bufferRenderFrameContent(e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 0)
		e.bufferRenderFrameContent(e.multichannelContentDetector.IsTemporaryMultiChannelContentDetected(), 1)
		e.bufferRemainingRenderFrameContent()

		frameToBuffer = e.renderTransferQueue.remove(renderQueueOutputFrame, queueInitial)
	}
}

// GetMetrics returns the current metrics.
func (e *EchoCanceller3) GetMetrics() EchoControlMetrics {
	return e.processor().GetMetrics()
}

// SetAudioBufferDelay sets an external estimate of the audio buffer delay.
func (e *EchoCanceller3) SetAudioBufferDelay(delayMs int) {
	e.processor().SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage tells the canceller whether the capture output is
// used. It may skip work whose result nothing uses, as when the endpoint is
// muted.
func (e *EchoCanceller3) SetCaptureOutputUsage(captureOutputUsed bool) {
	e.processor().SetCaptureOutputUsage(captureOutputUsed)
}

// ActiveProcessing reports whether the canceller processes the capture
// signal. AEC3 always does.
func (e *EchoCanceller3) ActiveProcessing() bool { return true }

// UpdateEchoLeakageStatus tells the canceller whether an external detector
// sees echo leak into its output.
func (e *EchoCanceller3) UpdateEchoLeakageStatus(leakageDetected bool) {
	e.processor().UpdateEchoLeakageStatus(leakageDetected)
}

// SampleRateHz returns the full-band sample rate.
func (e *EchoCanceller3) SampleRateHz() int { return e.sampleRateHz }

// NumBands returns the number of split bands.
func (e *EchoCanceller3) NumBands() int { return e.numBands }

// ArenasForTesting returns the arena of c and the one of the block
// processor, which is nil unless it has its own.
func (c *Ec3Core) ArenasForTesting() (arena, blockProcessor Arena) { return c.arena, c.bpArena }
