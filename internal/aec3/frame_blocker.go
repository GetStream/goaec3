package aec3

// FrameBlocker produces 64 sample multiband blocks from frames consisting of
// 2 subframes of 80 samples (frame_blocker.h). The samples left over between
// calls are kept in one pointer-free buffer; every band and channel holds
// the same number of them.
type FrameBlocker struct {
	numBands    int
	numChannels int
	fill        int       // buffered samples per band and channel
	buffer      []float32 // [band][channel][BlockSize]
}

// NewFrameBlocker returns a blocker for numBands bands of numChannels.
func NewFrameBlocker(numBands, numChannels int) *FrameBlocker {
	f := &FrameBlocker{}
	f.init(numBands, numChannels, make([]float32, numBands*numChannels*BlockSize))
	return f
}

// init sets up an empty blocker that keeps its samples in storage, which
// holds at least numBands*numChannels*BlockSize samples.
func (f *FrameBlocker) init(numBands, numChannels int, storage []float32) {
	*f = FrameBlocker{numBands: numBands, numChannels: numChannels,
		buffer: storage[:numBands*numChannels*BlockSize]}
}

// InsertSubFrameAndExtractBlock inserts the 80 sample multiband subframe
// subFrameIndex of frame and extracts one 64 sample multiband block. The
// blocker reads the first numChannels channels of the frame.
func (f *FrameBlocker) InsertSubFrameAndExtractBlock(frame Frame, subFrameIndex int, block *Block) {
	fill := f.fill
	samplesToBlock := BlockSize - fill
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			i := (band*f.numChannels + channel) * BlockSize
			buf := f.buffer[i : i+BlockSize]
			sub := frame.SubFrame(band, channel, subFrameIndex)
			dst := block.View(band, channel)
			copy(dst, buf[:fill])
			copy(dst[fill:], sub[:samplesToBlock])
			copy(buf, sub[samplesToBlock:])
		}
	}
	f.fill = SubFrameLength - samplesToBlock
}

// IsBlockAvailable reports whether a multiband block of 64 samples is
// available for extraction.
func (f *FrameBlocker) IsBlockAvailable() bool {
	return f.fill == BlockSize
}

// ExtractBlock extracts a multiband block of 64 samples.
func (f *FrameBlocker) ExtractBlock(block *Block) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			i := (band*f.numChannels + channel) * BlockSize
			copy(block.View(band, channel), f.buffer[i:i+BlockSize])
		}
	}
	f.fill = 0
}
