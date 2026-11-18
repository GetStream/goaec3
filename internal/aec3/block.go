package aec3

// Block is a [band][channel][BlockSize] float32 block, stored contiguously.
type Block struct {
	numBands    int
	numChannels int
	data        []float32
}

func NewBlock(numBands, numChannels int) Block {
	return Block{numBands: numBands, numChannels: numChannels, data: make([]float32, numBands*numChannels*BlockSize)}
}

// NewBlockFilled returns a block with every sample set to v.
func NewBlockFilled(numBands, numChannels int, v float32) Block {
	b := NewBlock(numBands, numChannels)
	for i := range b.data {
		b.data[i] = v
	}
	return b
}

func (b Block) NumBands() int    { return b.numBands }
func (b Block) NumChannels() int { return b.numChannels }

func (b *Block) SetNumChannels(numChannels int) {
	b.numChannels = numChannels
	n := b.numBands * numChannels * BlockSize
	if cap(b.data) >= n {
		b.data = b.data[:n]
	} else {
		b.data = make([]float32, n)
	}
	clear(b.data)
}

// View returns the BlockSize samples of one band/channel.
func (b Block) View(band, channel int) []float32 {
	i := (band*b.numChannels + channel) * BlockSize
	return b.data[i : i+BlockSize : i+BlockSize]
}

// CopyFrom copies o into b (same shape required).
func (b *Block) CopyFrom(o *Block) {
	b.numBands, b.numChannels = o.numBands, o.numChannels
	if cap(b.data) < len(o.data) {
		b.data = make([]float32, len(o.data))
	}
	b.data = b.data[:len(o.data)]
	copy(b.data, o.data)
}

func (b *Block) Swap(o *Block) { *b, *o = *o, *b }
