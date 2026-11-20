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

// The rings below keep all their slots in one pointer-free backing array
// and hand out slots as views computed from the slot index. A ring is a view
// (see arena.go): its indices and dimensions are a pointer-free state, which
// the methods reach through the embedded pointer, next to the slice of the
// slots.

// inlineChannels is the number of capture channels whose per-channel state a
// component keeps in arrays of its own struct. With more channels the state
// is allocated.
const inlineChannels = 2

// inlineChannelLimit is inlineChannels. Tests lower it, so that a component
// with few channels allocates its per-channel state as one with many does.
var inlineChannelLimit = inlineChannels

// channelState returns n elements of per-channel state of a component:
// inline[:n], an array of the component's own struct, for up to
// inlineChannels elements, and otherwise more, which the component
// allocated (see moreChannels). The component holds no pointer to inline, so
// with up to inlineChannels channels its per-channel state is nothing the
// garbage collector reads, and the component stays a plain value. A
// component keeps what it allocates for more channels behind one pointer at
// its start: the garbage collector reads every pointer word of a live
// object, nil or not, so pointer words scattered over a large struct each
// cost a cache line.
func channelState[T any](inline *[inlineChannels]T, n int, more []T) []T {
	if more != nil {
		return more
	}
	return inline[:n:n]
}

// moreChannels returns the n elements that channelState hands out when they
// do not fit inline, and nil when they do.
func moreChannels[T any](n int) []T {
	if n > inlineChannelLimit {
		return make([]T, n)
	}
	return nil
}

// noCopy marks a struct that is not copied once it is set up, because
// it holds pointers to itself: go vet (copylocks) reports the copies.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// BlockBuffer is a ring of Blocks (block_buffer.h). The slots share one
// pointer-free backing array; At returns a view of one slot.
type BlockBuffer struct {
	*blockBufferState
	data []float32
}

// blockBufferState is the state of a BlockBuffer but its slots.
type blockBufferState struct {
	ringIndex
	numBands    int
	numChannels int
	dataSpan    span
}

func NewBlockBuffer(size, numBands, numChannels int) *BlockBuffer {
	b := &BlockBuffer{}
	b.bind(&blockBufferState{}, nil)
	b.init(size, numBands, numChannels, make([]float32, size*numBands*numChannels*BlockSize))
	return b
}

// bind makes b the view of st.
func (b *BlockBuffer) bind(st *blockBufferState, bd *binder) {
	b.blockBufferState = st
	bd.f32(&b.data, &st.dataSpan)
}

// init sets up b in place over data, which holds the samples of size slots.
func (b *BlockBuffer) init(size, numBands, numChannels int, data []float32) {
	*b.blockBufferState = blockBufferState{ringIndex: ringIndex{Size: size}, numBands: numBands,
		numChannels: numChannels}
	b.data = data
}

// At returns slot i as a Block that aliases the ring storage.
func (b *BlockBuffer) At(i int) Block {
	st := b.blockBufferState
	n := st.numBands * st.numChannels * BlockSize
	return Block{numBands: st.numBands, numChannels: st.numChannels, data: b.data[i*n : (i+1)*n : (i+1)*n]}
}

func (b *BlockBuffer) NumBands() int    { return b.numBands }
func (b *BlockBuffer) NumChannels() int { return b.numChannels }

// SpectrumBuffer is a ring of per-channel power spectra (spectrum_buffer.h),
// stored slot-major in one backing array.
type SpectrumBuffer struct {
	*spectrumBufferState
	data [][FftLengthBy2Plus1]float32
}

// spectrumBufferState is the state of a SpectrumBuffer but its slots.
type spectrumBufferState struct {
	ringIndex
	numChannels int
	dataSpan    span
}

func NewSpectrumBuffer(size, numChannels int) *SpectrumBuffer {
	b := &SpectrumBuffer{}
	b.bind(&spectrumBufferState{}, nil)
	b.init(size, numChannels)
	return b
}

// bind makes b the view of st.
func (b *SpectrumBuffer) bind(st *spectrumBufferState, bd *binder) {
	b.spectrumBufferState = st
	bd.spec(&b.data, &st.dataSpan)
}

// init sets up b in place.
func (b *SpectrumBuffer) init(size, numChannels int) {
	*b.spectrumBufferState = spectrumBufferState{ringIndex: ringIndex{Size: size}, numChannels: numChannels}
	b.data = make([][FftLengthBy2Plus1]float32, size*numChannels)
}

// At returns the per-channel spectra of slot i.
func (b *SpectrumBuffer) At(i int) [][FftLengthBy2Plus1]float32 {
	n := b.numChannels
	return b.data[i*n : (i+1)*n : (i+1)*n]
}

func (b *SpectrumBuffer) NumChannels() int { return b.numChannels }

// FftBuffer is a ring of per-channel FFTs (fft_buffer.h), stored slot-major
// in one backing array.
type FftBuffer struct {
	*fftBufferState
	data []FftData
}

// fftBufferState is the state of an FftBuffer but its slots.
type fftBufferState struct {
	ringIndex
	numChannels int
	dataSpan    span
}

func NewFftBuffer(size, numChannels int) *FftBuffer {
	b := &FftBuffer{}
	b.bind(&fftBufferState{}, nil)
	b.init(size, numChannels)
	return b
}

// bind makes b the view of st.
func (b *FftBuffer) bind(st *fftBufferState, bd *binder) {
	b.fftBufferState = st
	bd.fft(&b.data, &st.dataSpan)
}

// init sets up b in place.
func (b *FftBuffer) init(size, numChannels int) {
	*b.fftBufferState = fftBufferState{ringIndex: ringIndex{Size: size}, numChannels: numChannels}
	b.data = make([]FftData, size*numChannels)
}

// At returns the per-channel FFTs of slot i.
func (b *FftBuffer) At(i int) []FftData {
	n := b.numChannels
	return b.data[i*n : (i+1)*n : (i+1)*n]
}

func (b *FftBuffer) NumChannels() int { return b.numChannels }

// DownsampledRenderBuffer is a ring of decimated samples
// (downsampled_render_buffer.h).
type DownsampledRenderBuffer struct {
	*downsampledRenderBufferState
	Buffer []float32
}

// downsampledRenderBufferState is the state of a DownsampledRenderBuffer but
// its samples.
type downsampledRenderBufferState struct {
	ringIndex
	bufferSpan span
}

func NewDownsampledRenderBuffer(size int) *DownsampledRenderBuffer {
	b := &DownsampledRenderBuffer{}
	b.bind(&downsampledRenderBufferState{}, nil)
	b.init(make([]float32, size))
	return b
}

// bind makes b the view of st.
func (b *DownsampledRenderBuffer) bind(st *downsampledRenderBufferState, bd *binder) {
	b.downsampledRenderBufferState = st
	bd.f32(&b.Buffer, &st.bufferSpan)
}

// init sets up b in place over buffer.
func (b *DownsampledRenderBuffer) init(buffer []float32) {
	*b.downsampledRenderBufferState = downsampledRenderBufferState{ringIndex: ringIndex{Size: len(buffer)}}
	b.Buffer = buffer
}
