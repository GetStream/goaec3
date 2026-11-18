package aec3

// FftData holds the 65 non-redundant bins of a 128-point real FFT.
type FftData struct {
	Re [FftLengthBy2Plus1]float32
	Im [FftLengthBy2Plus1]float32
}

// Assign copies src and zeroes the DC/Nyquist imaginary parts.
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

// Spectrum computes re^2+im^2 per bin. The scalar, SSE2 and NEON lanes round
// each product separately and agree bitwise; the AVX2 lane
// (FftData::SpectrumAVX2) fuses im*im+re^2 with _mm256_fmadd_ps and differs.
// libwebrtc compiles the AVX2 case only on x86, so OptAvx2 is honored on amd64
// only.
func (f *FftData) Spectrum(opt Aec3Optimization, powerSpectrum []float32) {
	_ = powerSpectrum[FftLengthBy2]
	fftDataSpectrum(opt, &f.Re, &f.Im, (*[FftLengthBy2Plus1]float32)(powerSpectrum))
}

func fftDataSpectrumGeneric(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := range out {
		out[k] = float32(re[k]*re[k]) + float32(im[k]*im[k])
	}
}

// CopyFromPackedArray unpacks the Ooura output layout.
func (f *FftData) CopyFromPackedArray(v *[FftLength]float32) { kernUnpack(v, &f.Re, &f.Im) }

// CopyToPackedArray packs into the Ooura input layout.
func (f *FftData) CopyToPackedArray(v *[FftLength]float32) { kernPack(&f.Re, &f.Im, v) }
