package aec3

// BlockFramer produces frames consisting of 2 subframes of 80 samples each
// from 64 sample blocks; it is the reverse of FrameBlocker (block_framer.h).
// The internal buffers overrun if any other rate of insertion is used. The
// samples left over between calls are kept in one pointer-free buffer; every
// band and channel holds the same number of them.
type BlockFramer struct {
	numBands    int
	numChannels int
	fill        int       // buffered samples per band and channel
	buffer      []float32 // [band][channel][BlockSize]
}

// NewBlockFramer returns a framer whose buffers start with BlockSize zeros.
func NewBlockFramer(numBands, numChannels int) *BlockFramer {
	f := &BlockFramer{}
	f.init(numBands, numChannels, make([]float32, numBands*numChannels*BlockSize))
	return f
}

// init sets up a framer that keeps its samples in storage, which holds at
// least numBands*numChannels*BlockSize zeros.
func (f *BlockFramer) init(numBands, numChannels int, storage []float32) {
	*f = BlockFramer{numBands: numBands, numChannels: numChannels, fill: BlockSize,
		buffer: storage[:numBands*numChannels*BlockSize]}
}

// InsertBlock adds a 64 sample block into the data that will form the next
// output frame. The buffers must be empty.
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

// InsertBlockAndExtractSubFrame adds a 64 sample block and extracts an 80
// sample subframe into subframe subFrameIndex of frame.
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
