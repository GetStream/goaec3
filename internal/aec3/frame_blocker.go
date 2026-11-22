package aec3

// FrameBlocker produces blocks of 64 samples from frames of two subframes of
// 80 samples (frame_blocker.h).
type FrameBlocker struct {
	numBands    int
	numChannels int
	fill        int       // buffered samples per band and channel
	buffer      []float32 // [band][channel][BlockSize]
}

// NewFrameBlocker returns an empty blocker.
func NewFrameBlocker(numBands, numChannels int) *FrameBlocker {
	f := &FrameBlocker{}
	f.init(numBands, numChannels, make([]float32, numBands*numChannels*BlockSize))
	return f
}

// init sets up an empty blocker in place. storage holds at least
// numBands*numChannels*BlockSize samples.
func (f *FrameBlocker) init(numBands, numChannels int, storage []float32) {
	*f = FrameBlocker{numBands: numBands, numChannels: numChannels,
		buffer: storage[:numBands*numChannels*BlockSize]}
}

// InsertSubFrameAndExtractBlock inserts subframe subFrameIndex of the first
// numChannels channels of frame and extracts one block.
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

// IsBlockAvailable reports whether the blocker holds a whole block.
func (f *FrameBlocker) IsBlockAvailable() bool {
	return f.fill == BlockSize
}

// ExtractBlock extracts one block.
func (f *FrameBlocker) ExtractBlock(block *Block) {
	for band := 0; band < f.numBands; band++ {
		for channel := 0; channel < f.numChannels; channel++ {
			i := (band*f.numChannels + channel) * BlockSize
			copy(block.View(band, channel), f.buffer[i:i+BlockSize])
		}
	}
	f.fill = 0
}
