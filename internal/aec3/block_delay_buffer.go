package aec3

// BlockDelayBuffer applies a fixed delay to the samples of a signal
// partitioned with the audio buffer band-splitting scheme
// (block_delay_buffer.h). The C++ operates on an AudioBuffer; here it
// operates on a Frame, of which it delays the first frameLength samples of
// every band and channel. The delay lines share one pointer-free buffer.
type BlockDelayBuffer struct {
	numChannels int
	numBands    int
	frameLength int
	delay       int
	buf         []float32 // [channel][band][delay]
	lastInsert  int
}

// NewBlockDelayBuffer mirrors the C++ constructor.
func NewBlockDelayBuffer(numChannels, numBands, frameLength, delaySamples int) *BlockDelayBuffer {
	b := &BlockDelayBuffer{}
	b.init(numChannels, numBands, frameLength, delaySamples)
	return b
}

func (b *BlockDelayBuffer) init(numChannels, numBands, frameLength, delaySamples int) {
	*b = BlockDelayBuffer{numChannels: numChannels, numBands: numBands, frameLength: frameLength,
		delay: delaySamples, buf: make([]float32, numChannels*numBands*delaySamples)}
}

// DelaySignal delays the first frameLength samples of every band and channel
// of frame in place by the configured number of samples.
func (b *BlockDelayBuffer) DelaySignal(frame Frame) {
	if b.delay == 0 {
		return
	}
	delay := b.delay
	iStart := b.lastInsert
	i := 0
	for ch := 0; ch < b.numChannels; ch++ {
		for band := 0; band < b.numBands; band++ {
			i = iStart
			k0 := (ch*b.numBands + band) * delay
			bufChBand := b.buf[k0 : k0+delay]
			frameChBand := frame.View(band, ch)[:b.frameLength]
			for k := range frameChBand {
				tmp := bufChBand[i]
				bufChBand[i] = frameChBand[k]
				frameChBand[k] = tmp
				if i < delay-1 {
					i++
				} else {
					i = 0
				}
			}
		}
	}
	b.lastInsert = i
}
