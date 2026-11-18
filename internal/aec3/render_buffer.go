package aec3

// RenderBuffer is a read view over the render block, spectrum and FFT rings.
type RenderBuffer struct {
	blockBuffer    *BlockBuffer
	spectrumBuffer *SpectrumBuffer
	fftBuffer      *FftBuffer
	renderActivity bool
}

func NewRenderBuffer(blockBuffer *BlockBuffer, spectrumBuffer *SpectrumBuffer, fftBuffer *FftBuffer) *RenderBuffer {
	return &RenderBuffer{blockBuffer: blockBuffer, spectrumBuffer: spectrumBuffer, fftBuffer: fftBuffer}
}

func (r *RenderBuffer) GetBlock(bufferOffsetBlocks int) *Block {
	p := r.blockBuffer.OffsetIndex(r.blockBuffer.Read, bufferOffsetBlocks)
	return &r.blockBuffer.Buffer[p]
}

func (r *RenderBuffer) Spectrum(bufferOffsetFfts int) [][FftLengthBy2Plus1]float32 {
	p := r.spectrumBuffer.OffsetIndex(r.spectrumBuffer.Read, bufferOffsetFfts)
	return r.spectrumBuffer.Buffer[p]
}

func (r *RenderBuffer) GetFftBuffer() [][]FftData { return r.fftBuffer.Buffer }

func (r *RenderBuffer) Position() int { return r.fftBuffer.Read }

// SpectralSum sums numSpectra spectra (all channels) starting at the read
// position.
func (r *RenderBuffer) SpectralSum(numSpectra int, X2 *[FftLengthBy2Plus1]float32) {
	var rows [rbMaxRows]*[FftLengthBy2Plus1]float32
	if n, ok := r.spectralRows(numSpectra, &rows); ok {
		kernSumRows(X2, rows[:n])
		return
	}
	*X2 = [FftLengthBy2Plus1]float32{}
	position := r.spectrumBuffer.Read
	for j := 0; j < numSpectra; j++ {
		for c := range r.spectrumBuffer.Buffer[position] {
			ch := &r.spectrumBuffer.Buffer[position][c]
			for k := range X2 {
				X2[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
}

// rbMaxRows bounds the spectra that SpectralSum(s) hand to kernSumRows from
// a stack array; longer sums run the scalar loops.
const rbMaxRows = 64

// spectralRows lists the spectra of numSpectra ring positions from the read
// position (all channels, in summation order). ok is false when they do not
// fit in rows.
func (r *RenderBuffer) spectralRows(numSpectra int, rows *[rbMaxRows]*[FftLengthBy2Plus1]float32) (int, bool) {
	sb := r.spectrumBuffer
	if numSpectra*len(sb.Buffer[0]) > rbMaxRows {
		return 0, false
	}
	n := 0
	position := sb.Read
	for j := 0; j < numSpectra; j++ {
		b := sb.Buffer[position]
		for c := range b {
			rows[n] = &b[c]
			n++
		}
		position = sb.IncIndex(position)
	}
	return n, true
}

// SpectralSums computes the sums over the first numShorter and numLonger
// spectra. Each running sum is the same per-bin sequence of additions as
// the C++ (the longer sum continues from the shorter one).
func (r *RenderBuffer) SpectralSums(numShorter, numLonger int, X2Shorter, X2Longer *[FftLengthBy2Plus1]float32) {
	var rows [rbMaxRows]*[FftLengthBy2Plus1]float32
	if n, ok := r.spectralRows(numLonger, &rows); ok {
		kernSumRows(X2Shorter, rows[:numShorter*len(r.spectrumBuffer.Buffer[0])])
		kernSumRows(X2Longer, rows[:n])
		return
	}
	*X2Shorter = [FftLengthBy2Plus1]float32{}
	position := r.spectrumBuffer.Read
	j := 0
	for ; j < numShorter; j++ {
		for c := range r.spectrumBuffer.Buffer[position] {
			ch := &r.spectrumBuffer.Buffer[position][c]
			for k := range X2Shorter {
				X2Shorter[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
	*X2Longer = *X2Shorter
	for ; j < numLonger; j++ {
		for c := range r.spectrumBuffer.Buffer[position] {
			ch := &r.spectrumBuffer.Buffer[position][c]
			for k := range X2Longer {
				X2Longer[k] += ch[k]
			}
		}
		position = r.spectrumBuffer.IncIndex(position)
	}
}

func (r *RenderBuffer) GetRenderActivity() bool            { return r.renderActivity }
func (r *RenderBuffer) SetRenderActivity(a bool)           { r.renderActivity = a }
func (r *RenderBuffer) GetSpectrumBuffer() *SpectrumBuffer { return r.spectrumBuffer }
func (r *RenderBuffer) GetBlockBuffer() *BlockBuffer       { return r.blockBuffer }

func (r *RenderBuffer) Headroom() int {
	f := r.fftBuffer
	if f.Write < f.Read {
		return f.Read - f.Write
	}
	return f.Size - f.Write + f.Read
}
