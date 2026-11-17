package aec3

// BlockDelayBuffer applies a fixed delay to the samples of a signal
// partitioned with the audio buffer band-splitting scheme
// (block_delay_buffer.h). The C++ operates on an AudioBuffer; here a frame is
// [band][channel][sample] with frameLength samples per band.
type BlockDelayBuffer struct {
	frameLength int
	delay       int
	buf         [][][]float32 // [channel][band][delay]
	lastInsert  int
}

// NewBlockDelayBuffer mirrors the C++ constructor.
func NewBlockDelayBuffer(numChannels, numBands, frameLength, delaySamples int) *BlockDelayBuffer {
	b := &BlockDelayBuffer{frameLength: frameLength, delay: delaySamples, buf: make([][][]float32, numChannels)}
	for ch := range b.buf {
		b.buf[ch] = make([][]float32, numBands)
		for band := range b.buf[ch] {
			b.buf[ch][band] = make([]float32, delaySamples)
		}
	}
	return b
}

// DelaySignal delays frame[band][channel][0:frameLength] in place by the
// configured number of samples.
func (b *BlockDelayBuffer) DelaySignal(frame [][][]float32) {
	if b.delay == 0 {
		return
	}
	numBands := len(b.buf[0])
	numChannels := len(b.buf)
	iStart := b.lastInsert
	i := 0
	for ch := 0; ch < numChannels; ch++ {
		delay := b.delay
		for band := 0; band < numBands; band++ {
			i = iStart
			bufChBand := b.buf[ch][band]
			frameChBand := frame[band][ch][:b.frameLength]
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
