package aec3

// BlockFramer produces frames consisting of 2 subframes of 80 samples each
// from 64 sample blocks; it is the reverse of FrameBlocker (block_framer.h).
// The internal buffers overrun if any other rate of insertion is used.
type BlockFramer struct {
	numBands    int
	numChannels int
	buffer      [][][]float32 // [band][channel], capacity BlockSize
}

// NewBlockFramer returns a framer whose buffers start with BlockSize zeros.
func NewBlockFramer(numBands, numChannels int) *BlockFramer {
	return &BlockFramer{numBands: numBands, numChannels: numChannels,
		buffer: newFlat3(numBands, numChannels, BlockSize, BlockSize)}
}

// InsertBlock adds a 64 sample block into the data that will form the next
// output frame. The buffers must be empty.
func (f *BlockFramer) InsertBlock(block *Block) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			if len(f.buffer[band][channel]) != 0 {
				panic("aec3: BlockFramer.InsertBlock called with a non-empty buffer")
			}
			f.buffer[band][channel] = append(f.buffer[band][channel][:0], block.View(band, channel)...)
		}
	}
}

// InsertBlockAndExtractSubFrame adds a 64 sample block and extracts an 80
// sample subframe into subFrame[band][channel].
func (f *BlockFramer) InsertBlockAndExtractSubFrame(block *Block, subFrame [][][]float32) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			buf := f.buffer[band][channel]
			if len(buf)+BlockSize < SubFrameLength || len(buf) > BlockSize {
				panic("aec3: BlockFramer buffer overrun")
			}
			out := subFrame[band][channel][:SubFrameLength]
			src := block.View(band, channel)
			samplesToFrame := SubFrameLength - len(buf)
			copy(out, buf)
			copy(out[len(buf):], src[:samplesToFrame])
			f.buffer[band][channel] = append(buf[:0], src[samplesToFrame:]...)
		}
	}
}
