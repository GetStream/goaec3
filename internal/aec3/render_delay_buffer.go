package aec3

// BufferingEvent mirrors RenderDelayBuffer::BufferingEvent.
type BufferingEvent int

const (
	BufferingEventNone BufferingEvent = iota
	BufferingEventRenderUnderrun
	BufferingEventRenderOverrun
	BufferingEventApiCallSkew
)

// RenderDelayBuffer buffers the render blocks and returns them with a given
// delay (RenderDelayBufferImpl in render_delay_buffer.cc).
type RenderDelayBuffer struct {
	*renderDelayBufferState
	blocks      BlockBuffer
	spectra     SpectrumBuffer
	ffts        FftBuffer
	lowRate     DownsampledRenderBuffer
	renderMixer AlignmentMixer
}

// renderDelayBufferState is the state of a RenderDelayBuffer except the
// slices of its rings and its mixer.
type renderDelayBufferState struct {
	blocks      blockBufferState
	spectra     spectrumBufferState
	ffts        fftBufferState
	lowRate     downsampledRenderBufferState
	renderMixer alignmentMixerState

	optimization                               Aec3Optimization
	config                                     EchoCanceller3Config
	renderLinearAmplitudeGain                  float32
	downSamplingFactor                         int
	subBlockSize                               int
	delay                                      int
	hasDelay                                   bool
	echoRemoverRenderActivity                  bool // of the render buffer for the echo remover
	renderDecimator                            Decimator
	fft                                        Aec3Fft
	fftPair                                    aec3FftPair
	bufferHeadroom                             int
	lastCallWasRender                          bool
	numApiCallsInARow                          int
	maxObservedJitter                          int
	captureCallCounter                         int64
	renderCallCounter                          int64
	renderActivity                             bool
	renderActivityCounter                      int
	externalAudioBufferDelay                   int
	hasExternalAudioBufferDelay                bool
	externalAudioBufferDelayVerifiedAfterReset bool
	minLatencyBlocks                           int
	excessRenderDetectionCounter               int
}

// NewRenderDelayBuffer mirrors RenderDelayBuffer::Create.
func NewRenderDelayBuffer(config *EchoCanceller3Config, sampleRateHz, numRenderChannels int) *RenderDelayBuffer {
	r := &RenderDelayBuffer{}
	r.bind(&renderDelayBufferState{}, bindAlone)
	r.init(config, sampleRateHz, numRenderChannels)
	return r
}

func (r *RenderDelayBuffer) bind(st *renderDelayBufferState, b *binder) {
	r.renderDelayBufferState = st
	r.blocks.bind(&st.blocks, b)
	r.spectra.bind(&st.spectra, b)
	r.ffts.bind(&st.ffts, b)
	r.lowRate.bind(&st.lowRate, b)
	r.renderMixer.bind(&st.renderMixer, b)
}

func (r *RenderDelayBuffer) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels int) {
	st := r.renderDelayBufferState
	*st = renderDelayBufferState{
		optimization:              DetectOptimization(),
		config:                    *config,
		renderLinearAmplitudeGain: Exp10f(config.RenderLevels.RenderPowerGainDb / 20),
		downSamplingFactor:        config.Delay.DownSamplingFactor,
		maxObservedJitter:         1,
	}
	*r = RenderDelayBuffer{}
	r.bind(st, bindAlone)
	r.subBlockSize = BlockSize
	if r.downSamplingFactor > 0 {
		r.subBlockSize = BlockSize / r.downSamplingFactor
	}
	size := GetRenderDelayBufferSize(r.downSamplingFactor, config.Delay.NumFilters, config.Filter.Refined.LengthBlocks)
	// The block ring and the ring of decimated samples share one array.
	numBands := NumBandsForRate(sampleRateHz)
	blockSamples := size * numBands * numRenderChannels * BlockSize
	lowRateSize := GetDownSampledBufferSize(r.downSamplingFactor, config.Delay.NumFilters)
	samples := make([]float32, blockSamples+lowRateSize)
	r.blocks.init(size, numBands, numRenderChannels, samples[:blockSamples:blockSamples])
	r.spectra.init(size, numRenderChannels)
	r.ffts.init(size, numRenderChannels)
	r.delay, r.hasDelay = config.Delay.DefaultDelay, true
	r.lowRate.init(samples[blockSamples:])
	m := &config.Delay.RenderAlignmentMixing
	r.renderMixer.init(numRenderChannels, m.Downmix, m.AdaptiveSelection, m.ActivityPowerThreshold,
		m.PreferFirstTwoChannels)
	r.renderDecimator.init(r.downSamplingFactor)
	r.bufferHeadroom = config.Filter.Refined.LengthBlocks
	r.Reset()
}

// Reset resets the alignment of the buffers and clears the reported delays.
func (r *RenderDelayBuffer) Reset() {
	r.lastCallWasRender = false
	r.numApiCallsInARow = 1
	r.minLatencyBlocks = 0
	r.excessRenderDetectionCounter = 0

	// The read index starts one sub-block before the write index.
	r.lowRate.Read = r.lowRate.OffsetIndex(r.lowRate.Write, r.subBlockSize)

	// An external audio buffer delay applies if the headroom allows it.
	if r.hasExternalAudioBufferDelay {
		const headroom = 2
		var audioBufferDelayToSet int
		// The delay is at least 1, as in the low-rate render buffer.
		if r.externalAudioBufferDelay <= headroom {
			audioBufferDelayToSet = 1
		} else {
			audioBufferDelayToSet = r.externalAudioBufferDelay - headroom
		}
		audioBufferDelayToSet = min(audioBufferDelayToSet, r.MaxDelay())

		r.applyTotalDelay(audioBufferDelayToSet)
		r.delay, r.hasDelay = r.computeDelay(), true

		r.externalAudioBufferDelayVerifiedAfterReset = false
	} else {
		// Without an external estimate the initial delay is the default.
		r.applyTotalDelay(r.config.Delay.DefaultDelay)

		// Unset the delays AlignFromDelay sets.
		r.hasDelay = false
	}
}

// Insert adds a block to the buffer.
func (r *RenderDelayBuffer) Insert(block *Block) BufferingEvent {
	r.renderCallCounter++
	if r.hasDelay {
		if !r.lastCallWasRender {
			r.lastCallWasRender = true
			r.numApiCallsInARow = 1
		} else {
			r.numApiCallsInARow++
			if r.numApiCallsInARow > r.maxObservedJitter {
				r.maxObservedJitter = r.numApiCallsInARow
			}
		}
	}

	previousWrite := r.blocks.Write
	r.incrementWriteIndices()

	// More render than capture data overruns the buffer, which resets it.
	event := BufferingEventNone
	if r.renderOverrun() {
		event = BufferingEventRenderOverrun
	}

	if !r.renderActivity {
		if r.detectActiveRender(block.View(0, 0)) {
			r.renderActivityCounter++
		}
		r.renderActivity = r.renderActivityCounter >= 20
	}

	r.insertBlock(block, previousWrite)

	if event != BufferingEventNone {
		r.Reset()
	}
	return event
}

// HandleSkippedCaptureProcessing is called for a capture block that
// PrepareCaptureProcessing is not called for.
func (r *RenderDelayBuffer) HandleSkippedCaptureProcessing() {
	r.captureCallCounter++
}

// PrepareCaptureProcessing advances the buffers by one capture block.
func (r *RenderDelayBuffer) PrepareCaptureProcessing() BufferingEvent {
	event := BufferingEventNone
	r.captureCallCounter++

	if r.hasDelay {
		if r.lastCallWasRender {
			r.lastCallWasRender = false
			r.numApiCallsInARow = 1
		} else {
			r.numApiCallsInARow++
			if r.numApiCallsInARow > r.maxObservedJitter {
				r.maxObservedJitter = r.numApiCallsInARow
			}
		}
	}

	if r.detectExcessRenderBlocks() {
		// Too many render blocks for the capture blocks: the delay may end
		// up before the filter of the delay estimator.
		r.Reset()
		event = BufferingEventRenderOverrun
	} else if r.renderUnderrun() {
		// An underrun does not advance the read indices of the low rate
		// buffer.
		r.incrementReadIndices()
		// Advancing the buffer index alone reduces the delay by one.
		if r.hasDelay && r.delay > 0 {
			r.delay--
		}
		event = BufferingEventRenderUnderrun
	} else {
		r.incrementLowRateReadIndices()
		r.incrementReadIndices()
	}

	r.echoRemoverRenderActivity = r.renderActivity
	if r.renderActivity {
		r.renderActivityCounter = 0
		r.renderActivity = false
	}
	return event
}

// AlignFromDelay sets the delay of the buffer and reports whether it
// changed.
func (r *RenderDelayBuffer) AlignFromDelay(delay int) bool {
	if !r.externalAudioBufferDelayVerifiedAfterReset && r.hasExternalAudioBufferDelay && r.hasDelay {
		r.externalAudioBufferDelayVerifiedAfterReset = true
	}
	if r.hasDelay && r.delay == delay {
		return false
	}
	r.delay, r.hasDelay = delay, true

	totalDelay := r.mapDelayToTotalDelay(r.delay)
	totalDelay = min(r.MaxDelay(), max(totalDelay, 0))

	r.applyTotalDelay(totalDelay)
	return true
}

// SetAudioBufferDelay sets an external estimate of the audio buffer delay.
func (r *RenderDelayBuffer) SetAudioBufferDelay(delayMs int) {
	// Milliseconds to blocks, rounded down.
	r.externalAudioBufferDelay = int(int32(delayMs) / 4)
	r.hasExternalAudioBufferDelay = true
}

// HasReceivedBufferDelay reports whether SetAudioBufferDelay was called.
func (r *RenderDelayBuffer) HasReceivedBufferDelay() bool { return r.hasExternalAudioBufferDelay }

// AlignFromExternalDelay sets the delay of the buffer from the external
// estimate.
func (r *RenderDelayBuffer) AlignFromExternalDelay() {
	if r.hasExternalAudioBufferDelay {
		delay := r.renderCallCounter - r.captureCallCounter + int64(r.externalAudioBufferDelay)
		delayWithHeadroom := delay - int64(r.config.Delay.DelayHeadroomSamples/BlockSize)
		r.applyTotalDelay(int(int32(delayWithHeadroom)))
	}
}

// Delay returns the delay of the buffer. A negative value is the wrapped
// size_t of the C++.
func (r *RenderDelayBuffer) Delay() int { return r.computeDelay() }

// MaxDelay returns the largest delay of the buffer.
func (r *RenderDelayBuffer) MaxDelay() int { return r.blocks.Size - 1 - r.bufferHeadroom }

// GetRenderBuffer returns the render buffer for the echo remover: a view of
// the rings as they are, to fetch again after r changes.
func (r *RenderDelayBuffer) GetRenderBuffer() *RenderBuffer {
	b := r.renderBuffer()
	return &b
}

// renderBuffer returns a RenderBuffer over the rings of r. insertBlock
// writes an FFT and its spectrum to the same slot, and the indices of the
// two rings move together (SpectraMatchFfts).
func (r *RenderDelayBuffer) renderBuffer() RenderBuffer {
	return RenderBuffer{blockBuffer: &r.blocks, spectrumBuffer: &r.spectra, fftBuffer: &r.ffts,
		renderActivity: r.echoRemoverRenderActivity, spectraMatchFfts: true}
}

// GetDownsampledRenderBuffer returns the downsampled render buffer.
func (r *RenderDelayBuffer) GetDownsampledRenderBuffer() *DownsampledRenderBuffer { return &r.lowRate }

// BufferLatency returns the number of unread sub-blocks.
func (r *RenderDelayBuffer) BufferLatency() int {
	l := &r.lowRate
	latencySamples := (len(l.Buffer) + l.Read - l.Write) % len(l.Buffer)
	return latencySamples / r.subBlockSize
}

// mapDelayToTotalDelay maps an external delay to the delay of the buffers.
func (r *RenderDelayBuffer) mapDelayToTotalDelay(externalDelayBlocks int) int {
	return r.BufferLatency() + externalDelayBlocks
}

// computeDelay returns the delay without the call jitter.
func (r *RenderDelayBuffer) computeDelay() int {
	latencyBlocks := r.BufferLatency()
	var internalDelay int
	if r.spectra.Read >= r.spectra.Write {
		internalDelay = r.spectra.Read - r.spectra.Write
	} else {
		internalDelay = r.spectra.Size + r.spectra.Read - r.spectra.Write
	}
	return internalDelay - latencyBlocks
}

// applyTotalDelay sets the read indices for delay.
func (r *RenderDelayBuffer) applyTotalDelay(delay int) {
	r.blocks.Read = r.blocks.OffsetIndex(r.blocks.Write, -delay)
	r.spectra.Read = r.spectra.OffsetIndex(r.spectra.Write, delay)
	r.ffts.Read = r.ffts.OffsetIndex(r.ffts.Write, delay)
}

// insertBlock writes block and what derives from it to the rings.
func (r *RenderDelayBuffer) insertBlock(block *Block, previousWrite int) {
	b := &r.blocks
	lr := &r.lowRate
	var renderDs [BlockSize]float32
	ds := renderDs[:r.subBlockSize]
	f := &r.ffts
	s := &r.spectra
	dst := b.At(b.Write)
	numBands := dst.NumBands()
	numRenderChannels := dst.NumChannels()
	for band := 0; band < numBands; band++ {
		for ch := 0; ch < numRenderChannels; ch++ {
			copy(dst.View(band, ch), block.View(band, ch))
		}
	}

	if r.renderLinearAmplitudeGain != 1 {
		for band := 0; band < numBands; band++ {
			for ch := 0; ch < numRenderChannels; ch++ {
				v := dst.View(band, ch)
				for i := range v {
					v[i] *= r.renderLinearAmplitudeGain
				}
			}
		}
	}

	var downmixedRender [BlockSize]float32
	r.renderMixer.ProduceOutput(&dst, downmixedRender[:])
	r.renderDecimator.Decimate(downmixedRender[:], ds)
	out := lr.Buffer[lr.Write : lr.Write+len(ds)]
	for i := range ds {
		out[i] = ds[len(ds)-1-i]
	}
	// Two channels per transform pair, then a last one alone.
	prev := b.At(previousWrite)
	X := f.At(f.Write)
	channel := 0
	for ; channel+1 < numRenderChannels; channel += 2 {
		r.fftPair.PaddedFftRectangular(dst.View(0, channel), prev.View(0, channel),
			dst.View(0, channel+1), prev.View(0, channel+1), &X[channel], &X[channel+1])
		X[channel].Spectrum(r.optimization, s.At(s.Write)[channel][:])
		X[channel+1].Spectrum(r.optimization, s.At(s.Write)[channel+1][:])
	}
	if channel < numRenderChannels {
		r.fft.PaddedFft(dst.View(0, channel), prev.View(0, channel), WindowRectangular, &X[channel])
		X[channel].Spectrum(r.optimization, s.At(s.Write)[channel][:])
	}
}

func (r *RenderDelayBuffer) detectActiveRender(x []float32) bool {
	var xEnergy float32
	for _, v := range x {
		xEnergy += float32(v * v)
	}
	limit := r.config.RenderLevels.ActiveRenderLimit
	return xEnergy > float32(float32(limit*limit)*FftLengthBy2)
}

func (r *RenderDelayBuffer) detectExcessRenderBlocks() bool {
	excessRenderDetected := false
	latencyBlocks := r.BufferLatency()
	// The smallest latency seen lately, in blocks. It stays close to 0.
	r.minLatencyBlocks = min(r.minLatencyBlocks, latencyBlocks)
	r.excessRenderDetectionCounter++
	if r.excessRenderDetectionCounter >= r.config.Buffering.ExcessRenderDetectionIntervalBlocks {
		// A minimum latency above the threshold means more render than
		// capture frames.
		excessRenderDetected = r.minLatencyBlocks > r.config.Buffering.MaxAllowedExcessRenderBlocks
		r.minLatencyBlocks = latencyBlocks
		r.excessRenderDetectionCounter = 0
	}
	return excessRenderDetected
}

func (r *RenderDelayBuffer) incrementWriteIndices() {
	r.lowRate.UpdateWriteIndex(-r.subBlockSize)
	r.blocks.IncWriteIndex()
	r.spectra.DecWriteIndex()
	r.ffts.DecWriteIndex()
}

func (r *RenderDelayBuffer) incrementLowRateReadIndices() {
	r.lowRate.UpdateReadIndex(-r.subBlockSize)
}

func (r *RenderDelayBuffer) incrementReadIndices() {
	if r.blocks.Read != r.blocks.Write {
		r.blocks.IncReadIndex()
		r.spectra.DecReadIndex()
		r.ffts.DecReadIndex()
	}
}

func (r *RenderDelayBuffer) renderOverrun() bool {
	return r.lowRate.Read == r.lowRate.Write || r.blocks.Read == r.blocks.Write
}

func (r *RenderDelayBuffer) renderUnderrun() bool {
	return r.lowRate.Read == r.lowRate.Write
}
