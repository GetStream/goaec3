package aec3

// RenderBuffer reads the block, spectrum and FFT rings of the render signal
// (render_buffer.h).
type RenderBuffer struct {
	blockBuffer    *BlockBuffer
	spectrumBuffer *SpectrumBuffer
	fftBuffer      *FftBuffer
	renderActivity bool
	// spectraMatchFfts is set by the owner of the rings. See
	// SpectraMatchFfts.
	spectraMatchFfts bool
}

// SpectraMatchFfts reports whether every slot of the spectrum ring holds
// FftData.Spectrum of the same slot of the FFT ring, and the two rings have
// one size and one read position. The spectra SpectralSum adds are then
// those of the FFTs the filters read. The rings of a RenderDelayBuffer are
// kept that way.
func (r *RenderBuffer) SpectraMatchFfts() bool {
	return r.spectraMatchFfts && r.spectrumBuffer.Size == r.fftBuffer.Size &&
		r.spectrumBuffer.Read == r.fftBuffer.Read
}

func NewRenderBuffer(blockBuffer *BlockBuffer, spectrumBuffer *SpectrumBuffer, fftBuffer *FftBuffer) *RenderBuffer {
	return &RenderBuffer{blockBuffer: blockBuffer, spectrumBuffer: spectrumBuffer, fftBuffer: fftBuffer}
}

// GetBlock returns the block bufferOffsetBlocks from the read position.
func (r *RenderBuffer) GetBlock(bufferOffsetBlocks int) Block {
	p := r.blockBuffer.OffsetIndex(r.blockBuffer.Read, bufferOffsetBlocks)
	return r.blockBuffer.At(p)
}

func (r *RenderBuffer) Spectrum(bufferOffsetFfts int) [][FftLengthBy2Plus1]float32 {
	p := r.spectrumBuffer.OffsetIndex(r.spectrumBuffer.Read, bufferOffsetFfts)
	return r.spectrumBuffer.At(p)
}

func (r *RenderBuffer) GetFftBuffer() *FftBuffer { return r.fftBuffer }

func (r *RenderBuffer) Position() int { return r.fftBuffer.Read }

// SpectralSum sets X2 to the sum of the spectra of all channels of
// numSpectra slots from the read position.
func (r *RenderBuffer) SpectralSum(numSpectra int, X2 *[FftLengthBy2Plus1]float32) {
	var rows [rbMaxRows]*[FftLengthBy2Plus1]float32
	if n, ok := r.spectralRows(numSpectra, &rows); ok {
		kernSumRows(X2, rows[:n])
		return
	}
	*X2 = [FftLengthBy2Plus1]float32{}
	position := r.spectrumBuffer.Read
	for j := 0; j < numSpectra; j++ {
		for c := range r.spectrumBuffer.At(position) {
			ch := &r.spectrumBuffer.At(position)[c]
			for k := range X2 {
				X2[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
}

// rbMaxRows is the largest number of spectra SpectralSum and SpectralSums
// hand to kernSumRows; longer sums run the scalar loops.
const rbMaxRows = 64

// spectralRows fills rows with the spectra of numSpectra slots from the read
// position, all channels, in the order of the sum. ok is false when they do
// not fit.
func (r *RenderBuffer) spectralRows(numSpectra int, rows *[rbMaxRows]*[FftLengthBy2Plus1]float32) (int, bool) {
	sb := r.spectrumBuffer
	if numSpectra*len(sb.At(0)) > rbMaxRows {
		return 0, false
	}
	n := 0
	position := sb.Read
	for j := 0; j < numSpectra; j++ {
		b := sb.At(position)
		for c := range b {
			rows[n] = &b[c]
			n++
		}
		position = sb.IncIndex(position)
	}
	return n, true
}

// SpectralSums computes the sums of the first numShorter and numLonger
// spectra. The longer sum continues from the shorter one, as in the C++.
func (r *RenderBuffer) SpectralSums(numShorter, numLonger int, X2Shorter, X2Longer *[FftLengthBy2Plus1]float32) {
	var rows [rbMaxRows]*[FftLengthBy2Plus1]float32
	if n, ok := r.spectralRows(numLonger, &rows); ok {
		kernSumRows(X2Shorter, rows[:numShorter*len(r.spectrumBuffer.At(0))])
		kernSumRows(X2Longer, rows[:n])
		return
	}
	*X2Shorter = [FftLengthBy2Plus1]float32{}
	position := r.spectrumBuffer.Read
	j := 0
	for ; j < numShorter; j++ {
		for c := range r.spectrumBuffer.At(position) {
			ch := &r.spectrumBuffer.At(position)[c]
			for k := range X2Shorter {
				X2Shorter[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
	*X2Longer = *X2Shorter
	for ; j < numLonger; j++ {
		for c := range r.spectrumBuffer.At(position) {
			ch := &r.spectrumBuffer.At(position)[c]
			for k := range X2Longer {
				X2Longer[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
}

func (r *RenderBuffer) GetRenderActivity() bool            { return r.renderActivity }
func (r *RenderBuffer) GetSpectrumBuffer() *SpectrumBuffer { return r.spectrumBuffer }
func (r *RenderBuffer) GetBlockBuffer() *BlockBuffer       { return r.blockBuffer }

func (r *RenderBuffer) Headroom() int {
	f := r.fftBuffer
	if f.Write < f.Read {
		return f.Read - f.Write
	}
	return f.Size - f.Write + f.Read
}
