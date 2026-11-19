package aec3

// BufferingEvent mirrors RenderDelayBuffer::BufferingEvent.
type BufferingEvent int

const (
	BufferingEventNone BufferingEvent = iota
	BufferingEventRenderUnderrun
	BufferingEventRenderOverrun
	BufferingEventApiCallSkew
)

// RenderDelayBuffer buffers the incoming render blocks such that they may be
// extracted with a specified delay (render_delay_buffer.h; this is the
// RenderDelayBufferImpl of render_delay_buffer.cc).
type RenderDelayBuffer struct {
	optimization                               Aec3Optimization
	config                                     *EchoCanceller3Config // shared, never modified
	renderLinearAmplitudeGain                  float32
	downSamplingFactor                         int
	subBlockSize                               int
	blocks                                     BlockBuffer
	spectra                                    SpectrumBuffer
	ffts                                       FftBuffer
	delay                                      int
	hasDelay                                   bool
	echoRemoverBuffer                          RenderBuffer // over blocks, spectra and ffts
	lowRate                                    DownsampledRenderBuffer
	renderMixer                                AlignmentMixer
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
// The result keeps a reference to config, which must not change while it
// is in use (the C++ copies it).
func NewRenderDelayBuffer(config *EchoCanceller3Config, sampleRateHz, numRenderChannels int) *RenderDelayBuffer {
	r := &RenderDelayBuffer{}
	r.init(config, sampleRateHz, numRenderChannels)
	return r
}

// init sets up r in place: its render buffer points at its own rings, so a
// RenderDelayBuffer must not be copied.
func (r *RenderDelayBuffer) init(config *EchoCanceller3Config, sampleRateHz, numRenderChannels int) {
	*r = RenderDelayBuffer{
		optimization:              DetectOptimization(),
		config:                    config,
		renderLinearAmplitudeGain: Exp10f(config.RenderLevels.RenderPowerGainDb / 20),
		downSamplingFactor:        config.Delay.DownSamplingFactor,
		maxObservedJitter:         1,
	}
	r.subBlockSize = BlockSize
	if r.downSamplingFactor > 0 {
		r.subBlockSize = BlockSize / r.downSamplingFactor
	}
	size := GetRenderDelayBufferSize(r.downSamplingFactor, config.Delay.NumFilters, config.Filter.Refined.LengthBlocks)
	// The block ring and the decimated render ring share one array.
	numBands := NumBandsForRate(sampleRateHz)
	blockSamples := size * numBands * numRenderChannels * BlockSize
	lowRateSize := GetDownSampledBufferSize(r.downSamplingFactor, config.Delay.NumFilters)
	samples := make([]float32, blockSamples+lowRateSize)
	r.blocks = BlockBuffer{ringIndex: ringIndex{Size: size}, numBands: numBands, numChannels: numRenderChannels,
		data: samples[:blockSamples:blockSamples]}
	r.spectra = *NewSpectrumBuffer(size, numRenderChannels)
	r.ffts = *NewFftBuffer(size, numRenderChannels)
	r.delay, r.hasDelay = config.Delay.DefaultDelay, true
	r.echoRemoverBuffer = *NewRenderBuffer(&r.blocks, &r.spectra, &r.ffts)
	r.lowRate = DownsampledRenderBuffer{ringIndex: ringIndex{Size: lowRateSize}, Buffer: samples[blockSamples:]}
	m := &config.Delay.RenderAlignmentMixing
	r.renderMixer.init(numRenderChannels, m.Downmix, m.AdaptiveSelection, m.ActivityPowerThreshold,
		m.PreferFirstTwoChannels)
	r.renderDecimator.init(r.downSamplingFactor)
	r.bufferHeadroom = config.Filter.Refined.LengthBlocks
	r.Reset()
}

// Reset resets the buffer alignment and clears the reported delays.
func (r *RenderDelayBuffer) Reset() {
	r.lastCallWasRender = false
	r.numApiCallsInARow = 1
	r.minLatencyBlocks = 0
	r.excessRenderDetectionCounter = 0

	// Initialize the read index to one sub-block before the write index.
	r.lowRate.Read = r.lowRate.OffsetIndex(r.lowRate.Write, r.subBlockSize)

	// Check for any external audio buffer delay and whether it is feasible.
	if r.hasExternalAudioBufferDelay {
		const headroom = 2
		var audioBufferDelayToSet int
		// Minimum delay is 1 (like the low-rate render buffer).
		if r.externalAudioBufferDelay <= headroom {
			audioBufferDelayToSet = 1
		} else {
			audioBufferDelayToSet = r.externalAudioBufferDelay - headroom
		}
		audioBufferDelayToSet = min(audioBufferDelayToSet, r.MaxDelay())

		// When an external delay estimate is available, use that delay as the
		// initial render buffer delay.
		r.applyTotalDelay(audioBufferDelayToSet)
		r.delay, r.hasDelay = r.computeDelay(), true

		r.externalAudioBufferDelayVerifiedAfterReset = false
	} else {
		// If an external delay estimate is not available, use that delay as
		// the initial delay. Set the render buffer delays to the default delay.
		r.applyTotalDelay(r.config.Delay.DefaultDelay)

		// Unset the delays which are set by AlignFromDelay.
		r.hasDelay = false
	}
}

// Insert inserts a block into the buffer.
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

	// Increase the write indices to where the new blocks should be written.
	previousWrite := r.blocks.Write
	r.incrementWriteIndices()

	// Allow overrun and do a reset when render overrun occurrs due to more
	// render data being inserted than capture data is received.
	event := BufferingEventNone
	if r.renderOverrun() {
		event = BufferingEventRenderOverrun
	}

	// Detect and update render activity.
	if !r.renderActivity {
		if r.detectActiveRender(block.View(0, 0)) {
			r.renderActivityCounter++
		}
		r.renderActivity = r.renderActivityCounter >= 20
	}

	// Insert the new render block into the specified position.
	r.insertBlock(block, previousWrite)

	if event != BufferingEventNone {
		r.Reset()
	}
	return event
}

// HandleSkippedCaptureProcessing is called on capture blocks where
// PrepareCaptureProcessing is not called.
func (r *RenderDelayBuffer) HandleSkippedCaptureProcessing() {
	r.captureCallCounter++
}

// PrepareCaptureProcessing updates the buffers one step for processing
// another capture block.
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
		// Too many render blocks compared to capture blocks. Risk of delay
		// ending up before the filter used by the delay estimator.
		r.Reset()
		event = BufferingEventRenderOverrun
	} else if r.renderUnderrun() {
		// Don't increment the read indices of the low rate buffer if there is
		// a render underrun.
		r.incrementReadIndices()
		// Incrementing the buffer index without increasing the low rate buffer
		// index means that the delay is reduced by one.
		if r.hasDelay && r.delay > 0 {
			r.delay--
		}
		event = BufferingEventRenderUnderrun
	} else {
		// Increment the read indices in the render buffers to point to the
		// most recent block to use in the capture processing.
		r.incrementLowRateReadIndices()
		r.incrementReadIndices()
	}

	r.echoRemoverBuffer.SetRenderActivity(r.renderActivity)
	if r.renderActivity {
		r.renderActivityCounter = 0
		r.renderActivity = false
	}
	return event
}

// AlignFromDelay sets the buffer delay and returns whether it changed.
func (r *RenderDelayBuffer) AlignFromDelay(delay int) bool {
	if !r.externalAudioBufferDelayVerifiedAfterReset && r.hasExternalAudioBufferDelay && r.hasDelay {
		r.externalAudioBufferDelayVerifiedAfterReset = true
	}
	if r.hasDelay && r.delay == delay {
		return false
	}
	r.delay, r.hasDelay = delay, true

	// Compute the total delay and limit the delay to the allowed range.
	totalDelay := r.mapDelayToTotalDelay(r.delay)
	totalDelay = min(r.MaxDelay(), max(totalDelay, 0))

	// Apply the delay to the buffers.
	r.applyTotalDelay(totalDelay)
	return true
}

// SetAudioBufferDelay provides an external estimate of the audio buffer
// delay in milliseconds.
func (r *RenderDelayBuffer) SetAudioBufferDelay(delayMs int) {
	// Convert delay from milliseconds to blocks (rounded down).
	r.externalAudioBufferDelay = int(int32(delayMs) / 4)
	r.hasExternalAudioBufferDelay = true
}

// HasReceivedBufferDelay reports whether SetAudioBufferDelay has been called.
func (r *RenderDelayBuffer) HasReceivedBufferDelay() bool { return r.hasExternalAudioBufferDelay }

// AlignFromExternalDelay sets the buffer delay from the most recently
// reported external delay.
func (r *RenderDelayBuffer) AlignFromExternalDelay() {
	if r.hasExternalAudioBufferDelay {
		delay := r.renderCallCounter - r.captureCallCounter + int64(r.externalAudioBufferDelay)
		delayWithHeadroom := delay - int64(r.config.Delay.DelayHeadroomSamples/BlockSize)
		r.applyTotalDelay(int(int32(delayWithHeadroom)))
	}
}

// Delay returns the buffer delay. The C++ returns it as size_t; a negative
// value here corresponds to the wrapped C++ value.
func (r *RenderDelayBuffer) Delay() int { return r.computeDelay() }

// MaxDelay returns the maximum buffer delay.
func (r *RenderDelayBuffer) MaxDelay() int { return r.blocks.Size - 1 - r.bufferHeadroom }

// GetRenderBuffer returns the render buffer for the echo remover.
func (r *RenderDelayBuffer) GetRenderBuffer() *RenderBuffer { return &r.echoRemoverBuffer }

// GetDownsampledRenderBuffer returns the downsampled render buffer.
func (r *RenderDelayBuffer) GetDownsampledRenderBuffer() *DownsampledRenderBuffer { return &r.lowRate }

// BufferLatency returns the number of unread sub-blocks in the buffer.
func (r *RenderDelayBuffer) BufferLatency() int {
	l := &r.lowRate
	latencySamples := (len(l.Buffer) + l.Read - l.Write) % len(l.Buffer)
	return latencySamples / r.subBlockSize
}

// mapDelayToTotalDelay maps the externally computed delay to the delay used
// internally.
func (r *RenderDelayBuffer) mapDelayToTotalDelay(externalDelayBlocks int) int {
	return r.BufferLatency() + externalDelayBlocks
}

// computeDelay returns the delay (not including call jitter).
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

// applyTotalDelay sets the read indices according to the delay.
func (r *RenderDelayBuffer) applyTotalDelay(delay int) {
	r.blocks.Read = r.blocks.OffsetIndex(r.blocks.Write, -delay)
	r.spectra.Read = r.spectra.OffsetIndex(r.spectra.Write, delay)
	r.ffts.Read = r.ffts.OffsetIndex(r.ffts.Write, delay)
}

// insertBlock inserts a block into the render buffers.
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
	// The channels are transformed two at a time, then a last one alone.
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
	// The recently seen minimum latency in blocks. Should be close to 0.
	r.minLatencyBlocks = min(r.minLatencyBlocks, latencyBlocks)
	// After processing a configurable number of blocks the minimum latency is
	// checked.
	r.excessRenderDetectionCounter++
	if r.excessRenderDetectionCounter >= r.config.Buffering.ExcessRenderDetectionIntervalBlocks {
		// If the minimum latency is not lower than the threshold there have
		// been more render than capture frames.
		excessRenderDetected = r.minLatencyBlocks > r.config.Buffering.MaxAllowedExcessRenderBlocks
		// Reset the counter and let the minimum latency be the current latency.
		r.minLatencyBlocks = latencyBlocks
		r.excessRenderDetectionCounter = 0
	}
	return excessRenderDetected
}

// incrementWriteIndices increments the write indices for the render buffers.
func (r *RenderDelayBuffer) incrementWriteIndices() {
	r.lowRate.UpdateWriteIndex(-r.subBlockSize)
	r.blocks.IncWriteIndex()
	r.spectra.DecWriteIndex()
	r.ffts.DecWriteIndex()
}

// incrementLowRateReadIndices increments the read indices of the low rate
// render buffers.
func (r *RenderDelayBuffer) incrementLowRateReadIndices() {
	r.lowRate.UpdateReadIndex(-r.subBlockSize)
}

// incrementReadIndices increments the read indices for the render buffers.
func (r *RenderDelayBuffer) incrementReadIndices() {
	if r.blocks.Read != r.blocks.Write {
		r.blocks.IncReadIndex()
		r.spectra.DecReadIndex()
		r.ffts.DecReadIndex()
	}
}

// renderOverrun checks for a render buffer overrun.
func (r *RenderDelayBuffer) renderOverrun() bool {
	return r.lowRate.Read == r.lowRate.Write || r.blocks.Read == r.blocks.Write
}

// renderUnderrun checks for a render buffer underrun.
func (r *RenderDelayBuffer) renderUnderrun() bool {
	return r.lowRate.Read == r.lowRate.Write
}
