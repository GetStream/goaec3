package aec3

// BlockDelayBuffer delays the bands of a signal by a fixed number of samples
// (block_delay_buffer.h). It works on the first frameLength samples of every
// band and channel of a Frame, where the C++ takes an AudioBuffer.
type BlockDelayBuffer struct {
	numChannels int
	numBands    int
	frameLength int
	delay       int
	buf         []float32 // [channel][band][delay]
	lastInsert  int
}

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
// of frame in place.
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
