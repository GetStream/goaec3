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

func (r *RenderBuffer) SpectralSums(numShorter, numLonger int, X2Shorter, X2Longer *[FftLengthBy2Plus1]float32) {
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
