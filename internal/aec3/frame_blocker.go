package aec3

// FrameBlocker produces 64 sample multiband blocks from frames consisting of
// 2 subframes of 80 samples (frame_blocker.h).
type FrameBlocker struct {
	numBands    int
	numChannels int
	buffer      [][][]float32 // [band][channel], capacity BlockSize
}

// NewFrameBlocker returns a blocker for numBands bands of numChannels.
func NewFrameBlocker(numBands, numChannels int) *FrameBlocker {
	f := &FrameBlocker{numBands: numBands, numChannels: numChannels, buffer: make([][][]float32, numBands)}
	for band := range f.buffer {
		f.buffer[band] = make([][]float32, numChannels)
		for ch := range f.buffer[band] {
			f.buffer[band][ch] = make([]float32, 0, BlockSize)
		}
	}
	return f
}

// InsertSubFrameAndExtractBlock inserts one 80 sample multiband subframe
// (subFrame[band][channel]) and extracts one 64 sample multiband block.
func (f *FrameBlocker) InsertSubFrameAndExtractBlock(subFrame [][][]float32, block *Block) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			buf := f.buffer[band][channel]
			sub := subFrame[band][channel][:SubFrameLength]
			samplesToBlock := BlockSize - len(buf)
			dst := block.View(band, channel)
			copy(dst, buf)
			copy(dst[BlockSize-samplesToBlock:], sub[:samplesToBlock])
			f.buffer[band][channel] = append(buf[:0], sub[samplesToBlock:]...)
		}
	}
}

// IsBlockAvailable reports whether a multiband block of 64 samples is
// available for extraction.
func (f *FrameBlocker) IsBlockAvailable() bool {
	return len(f.buffer[0][0]) == BlockSize
}

// ExtractBlock extracts a multiband block of 64 samples.
func (f *FrameBlocker) ExtractBlock(block *Block) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			copy(block.View(band, channel), f.buffer[band][channel])
			f.buffer[band][channel] = f.buffer[band][channel][:0]
		}
	}
}
