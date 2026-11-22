package aec3

// BlockFramer produces frames of two subframes of 80 samples from blocks of
// 64 samples, the reverse of FrameBlocker (block_framer.h). Any other rate
// of insertion overruns its buffers.
type BlockFramer struct {
	numBands    int
	numChannels int
	fill        int       // buffered samples per band and channel
	buffer      []float32 // [band][channel][BlockSize]
}

// NewBlockFramer returns a framer whose buffers hold BlockSize zeros.
func NewBlockFramer(numBands, numChannels int) *BlockFramer {
	f := &BlockFramer{}
	f.init(numBands, numChannels, make([]float32, numBands*numChannels*BlockSize))
	return f
}

// init sets up f in place. storage holds at least
// numBands*numChannels*BlockSize zeros.
func (f *BlockFramer) init(numBands, numChannels int, storage []float32) {
	*f = BlockFramer{numBands: numBands, numChannels: numChannels, fill: BlockSize,
		buffer: storage[:numBands*numChannels*BlockSize]}
}

// InsertBlock adds a block to the samples of the next frame. The buffers
// must be empty.
func (f *BlockFramer) InsertBlock(block *Block) {
	if f.fill != 0 {
		panic("aec3: BlockFramer.InsertBlock called with a non-empty buffer")
	}
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			i := (band*f.numChannels + channel) * BlockSize
			copy(f.buffer[i:i+BlockSize], block.View(band, channel))
		}
	}
	f.fill = BlockSize
}

// InsertBlockAndExtractSubFrame adds a block and extracts subframe
// subFrameIndex of frame.
func (f *BlockFramer) InsertBlockAndExtractSubFrame(block *Block, frame Frame, subFrameIndex int) {
	fill := f.fill
	if fill+BlockSize < SubFrameLength || fill > BlockSize {
		panic("aec3: BlockFramer buffer overrun")
	}
	samplesToFrame := SubFrameLength - fill
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			i := (band*f.numChannels + channel) * BlockSize
			buf := f.buffer[i : i+BlockSize]
			out := frame.SubFrame(band, channel, subFrameIndex)
			src := block.View(band, channel)
			copy(out, buf[:fill])
			copy(out[fill:], src[:samplesToFrame])
			copy(buf, src[samplesToFrame:])
		}
	}
	f.fill = BlockSize - samplesToFrame
}
