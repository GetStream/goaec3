package aec3

// FftData holds the 65 non-redundant bins of a 128-point real FFT.
type FftData struct {
	Re [FftLengthBy2Plus1]float32
	Im [FftLengthBy2Plus1]float32
}

// Assign copies src and zeroes the imaginary parts of the first and the last
// bin.
func (f *FftData) Assign(src *FftData) {
	f.Re = src.Re
	f.Im = src.Im
	f.Im[0] = 0
	f.Im[FftLengthBy2] = 0
}

func (f *FftData) Clear() {
	f.Re = [FftLengthBy2Plus1]float32{}
	f.Im = [FftLengthBy2Plus1]float32{}
}

// Spectrum sets powerSpectrum to re*re + im*im per bin. The AVX2 lane fuses
// the sum (FftData::SpectrumAVX2), on amd64 only; the other lanes agree bit
// for bit.
func (f *FftData) Spectrum(opt Aec3Optimization, powerSpectrum []float32) {
	_ = powerSpectrum[FftLengthBy2]
	fftDataSpectrum(opt, &f.Re, &f.Im, (*[FftLengthBy2Plus1]float32)(powerSpectrum))
}

func fftDataSpectrumGeneric(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := range out {
		out[k] = float32(re[k]*re[k]) + float32(im[k]*im[k])
	}
}

// CopyFromPackedArray unpacks the output of the Ooura transform.
func (f *FftData) CopyFromPackedArray(v *[FftLength]float32) { kernUnpack(v, &f.Re, &f.Im) }

// CopyToPackedArray packs f into the input of the Ooura transform.
func (f *FftData) CopyToPackedArray(v *[FftLength]float32) { kernPack(&f.Re, &f.Im, v) }
