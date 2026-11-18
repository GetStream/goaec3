package aec3

// ringIndex holds the read/write positions shared by the AEC3 ring buffers.
type ringIndex struct {
	Size  int
	Write int
	Read  int
}

func (r *ringIndex) IncIndex(index int) int {
	if index < r.Size-1 {
		return index + 1
	}
	return 0
}

func (r *ringIndex) DecIndex(index int) int {
	if index > 0 {
		return index - 1
	}
	return r.Size - 1
}

// OffsetIndex returns (Size + index + offset) mod Size. The C++ uses int %,
// which yields a negative position when offset < -(Size + index); indexing
// with it is undefined behavior there (reachable in external-delay mode with
// long render bursts). The port wraps it into [0, Size) instead.
func (r *ringIndex) OffsetIndex(index, offset int) int {
	p := (r.Size + index + offset) % r.Size
	if p < 0 {
		p += r.Size
	}
	return p
}
func (r *ringIndex) UpdateWriteIndex(offset int) { r.Write = r.OffsetIndex(r.Write, offset) }
func (r *ringIndex) IncWriteIndex()              { r.Write = r.IncIndex(r.Write) }
func (r *ringIndex) DecWriteIndex()              { r.Write = r.DecIndex(r.Write) }
func (r *ringIndex) UpdateReadIndex(offset int)  { r.Read = r.OffsetIndex(r.Read, offset) }
func (r *ringIndex) IncReadIndex()               { r.Read = r.IncIndex(r.Read) }
func (r *ringIndex) DecReadIndex()               { r.Read = r.DecIndex(r.Read) }

// The rings below keep all their slots in one backing array (one heap
// object per ring rather than one per slot); each slot is a full slice
// expression of its own part, so a slot never grows into its neighbour.

// newFlat3 returns [n0][n1] slices of length length and capacity capacity,
// all in one backing array.
func newFlat3(n0, n1, length, capacity int) [][][]float32 {
	var data []float32
	if capacity > 0 {
		data = make([]float32, n0*n1*capacity)
	}
	views := make([][]float32, n0*n1)
	v := make([][][]float32, n0)
	for i := range v {
		v[i] = views[i*n1 : (i+1)*n1 : (i+1)*n1]
		for j := range v[i] {
			k := (i*n1 + j) * capacity
			v[i][j] = data[k : k+length : k+capacity]
		}
	}
	return v
}

// BlockBuffer is a ring of Blocks (block_buffer.h).
type BlockBuffer struct {
	ringIndex
	Buffer []Block
}

func NewBlockBuffer(size, numBands, numChannels int) *BlockBuffer {
	b := &BlockBuffer{ringIndex: ringIndex{Size: size}, Buffer: make([]Block, size)}
	n := numBands * numChannels * BlockSize
	data := make([]float32, size*n)
	for i := range b.Buffer {
		b.Buffer[i] = Block{numBands: numBands, numChannels: numChannels, data: data[i*n : (i+1)*n : (i+1)*n]}
	}
	return b
}

// SpectrumBuffer is a ring of per-channel power spectra (spectrum_buffer.h).
type SpectrumBuffer struct {
	ringIndex
	Buffer [][][FftLengthBy2Plus1]float32
}

func NewSpectrumBuffer(size, numChannels int) *SpectrumBuffer {
	b := &SpectrumBuffer{ringIndex: ringIndex{Size: size}, Buffer: make([][][FftLengthBy2Plus1]float32, size)}
	data := make([][FftLengthBy2Plus1]float32, size*numChannels)
	for i := range b.Buffer {
		b.Buffer[i] = data[i*numChannels : (i+1)*numChannels : (i+1)*numChannels]
	}
	return b
}

// FftBuffer is a ring of per-channel FFTs (fft_buffer.h).
type FftBuffer struct {
	ringIndex
	Buffer [][]FftData
}

func NewFftBuffer(size, numChannels int) *FftBuffer {
	b := &FftBuffer{ringIndex: ringIndex{Size: size}, Buffer: make([][]FftData, size)}
	data := make([]FftData, size*numChannels)
	for i := range b.Buffer {
		b.Buffer[i] = data[i*numChannels : (i+1)*numChannels : (i+1)*numChannels]
	}
	return b
}

// DownsampledRenderBuffer is a ring of decimated samples
// (downsampled_render_buffer.h).
type DownsampledRenderBuffer struct {
	ringIndex
	Buffer []float32
}

func NewDownsampledRenderBuffer(size int) *DownsampledRenderBuffer {
	return &DownsampledRenderBuffer{ringIndex: ringIndex{Size: size}, Buffer: make([]float32, size)}
}
