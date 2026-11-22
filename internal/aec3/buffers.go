package aec3

// ringIndex holds the read and write positions of a ring buffer.
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

// OffsetIndex returns index+offset wrapped into [0, Size). The C++ computes
// (Size + index + offset) % Size on ints, which is negative for offset <
// -(Size + index) and then indexes out of bounds; that is reachable with an
// external delay estimator and long render bursts.
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

// The rings below keep their slots in one pointer-free array and return a
// slot as a view computed from its index. A ring is a view (see arena.go).

// inlineChannels is the number of capture channels whose per-channel state a
// component keeps in arrays of its own struct. It allocates the state of
// more channels.
const inlineChannels = 2

// inlineChannelLimit is inlineChannels. Tests lower it to make a component
// with few channels allocate its per-channel state.
var inlineChannelLimit = inlineChannels

// channelState returns the n elements of per-channel state of a component:
// inline[:n] for up to inlineChannels elements, more otherwise (see
// moreChannels). The component holds no pointer to inline, so the garbage
// collector reads nothing for the state of few channels.
func channelState[T any](inline *[inlineChannels]T, n int, more []T) []T {
	if more != nil {
		return more
	}
	return inline[:n:n]
}

// moreChannels returns n elements for channelState, or nil when they fit
// inline.
func moreChannels[T any](n int) []T {
	if n > inlineChannelLimit {
		return make([]T, n)
	}
	return nil
}

// noCopy marks a struct that holds pointers to itself. go vet reports its
// copies.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// BlockBuffer is a ring of Blocks (block_buffer.h).
//
// The view copies the dimensions, which never change, so that At reads
// nothing of the state.
type BlockBuffer struct {
	*blockBufferState
	data        []float32
	numBands    int
	numChannels int
}

// blockBufferState is the state of a BlockBuffer except its slots.
type blockBufferState struct {
	ringIndex
	numBands    int
	numChannels int
	dataSpan    span
}

func NewBlockBuffer(size, numBands, numChannels int) *BlockBuffer {
	b := &BlockBuffer{}
	b.bind(&blockBufferState{}, bindAlone)
	b.init(size, numBands, numChannels, make([]float32, size*numBands*numChannels*BlockSize))
	return b
}

func (b *BlockBuffer) bind(st *blockBufferState, bd *binder) {
	b.blockBufferState = st
	b.numBands, b.numChannels = st.numBands, st.numChannels
	bd.f32(&b.data, &st.dataSpan)
}

// init sets up b in place. data holds the samples of size slots.
func (b *BlockBuffer) init(size, numBands, numChannels int, data []float32) {
	*b.blockBufferState = blockBufferState{ringIndex: ringIndex{Size: size}, numBands: numBands,
		numChannels: numChannels}
	b.numBands, b.numChannels = numBands, numChannels
	b.data = data
}

// At returns slot i. The Block aliases the ring.
func (b BlockBuffer) At(i int) Block {
	n := b.numBands * b.numChannels * BlockSize
	return Block{numBands: b.numBands, numChannels: b.numChannels, data: b.data[i*n : (i+1)*n : (i+1)*n]}
}

func (b BlockBuffer) NumBands() int    { return b.numBands }
func (b BlockBuffer) NumChannels() int { return b.numChannels }

// SpectrumBuffer is a ring of per-channel power spectra (spectrum_buffer.h).
type SpectrumBuffer struct {
	*spectrumBufferState
	data        [][FftLengthBy2Plus1]float32
	numChannels int // a copy of the state's, for At
}

// spectrumBufferState is the state of a SpectrumBuffer except its slots.
type spectrumBufferState struct {
	ringIndex
	numChannels int
	dataSpan    span
}

func NewSpectrumBuffer(size, numChannels int) *SpectrumBuffer {
	b := &SpectrumBuffer{}
	b.bind(&spectrumBufferState{}, bindAlone)
	b.init(size, numChannels)
	return b
}

func (b *SpectrumBuffer) bind(st *spectrumBufferState, bd *binder) {
	b.spectrumBufferState = st
	b.numChannels = st.numChannels
	bd.spec(&b.data, &st.dataSpan)
}

func (b *SpectrumBuffer) init(size, numChannels int) {
	*b.spectrumBufferState = spectrumBufferState{ringIndex: ringIndex{Size: size}, numChannels: numChannels}
	b.numChannels = numChannels
	b.data = make([][FftLengthBy2Plus1]float32, size*numChannels)
}

// At returns the spectra of slot i, one per channel.
func (b SpectrumBuffer) At(i int) [][FftLengthBy2Plus1]float32 {
	n := b.numChannels
	return b.data[i*n : (i+1)*n : (i+1)*n]
}

func (b SpectrumBuffer) NumChannels() int { return b.numChannels }

// FftBuffer is a ring of per-channel FFTs (fft_buffer.h).
type FftBuffer struct {
	*fftBufferState
	data        []FftData
	numChannels int // a copy of the state's, for At
}

// fftBufferState is the state of an FftBuffer except its slots.
type fftBufferState struct {
	ringIndex
	numChannels int
	dataSpan    span
}

func NewFftBuffer(size, numChannels int) *FftBuffer {
	b := &FftBuffer{}
	b.bind(&fftBufferState{}, bindAlone)
	b.init(size, numChannels)
	return b
}

func (b *FftBuffer) bind(st *fftBufferState, bd *binder) {
	b.fftBufferState = st
	b.numChannels = st.numChannels
	bd.fft(&b.data, &st.dataSpan)
}

func (b *FftBuffer) init(size, numChannels int) {
	*b.fftBufferState = fftBufferState{ringIndex: ringIndex{Size: size}, numChannels: numChannels}
	b.numChannels = numChannels
	b.data = make([]FftData, size*numChannels)
}

// At returns the FFTs of slot i, one per channel.
func (b FftBuffer) At(i int) []FftData {
	n := b.numChannels
	return b.data[i*n : (i+1)*n : (i+1)*n]
}

func (b FftBuffer) NumChannels() int { return b.numChannels }

// DownsampledRenderBuffer is a ring of decimated samples
// (downsampled_render_buffer.h).
type DownsampledRenderBuffer struct {
	*downsampledRenderBufferState
	Buffer []float32
}

// downsampledRenderBufferState is the state of a DownsampledRenderBuffer
// except its samples.
type downsampledRenderBufferState struct {
	ringIndex
	bufferSpan span
}

func NewDownsampledRenderBuffer(size int) *DownsampledRenderBuffer {
	b := &DownsampledRenderBuffer{}
	b.bind(&downsampledRenderBufferState{}, bindAlone)
	b.init(make([]float32, size))
	return b
}

func (b *DownsampledRenderBuffer) bind(st *downsampledRenderBufferState, bd *binder) {
	b.downsampledRenderBufferState = st
	bd.f32(&b.Buffer, &st.bufferSpan)
}

func (b *DownsampledRenderBuffer) init(buffer []float32) {
	*b.downsampledRenderBufferState = downsampledRenderBufferState{ringIndex: ringIndex{Size: len(buffer)}}
	b.Buffer = buffer
}
