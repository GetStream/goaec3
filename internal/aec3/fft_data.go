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

// Spectrum computes re^2+im^2 per bin. The SSE2/AVX2 variants compute the
// same per-lane products and sums, so every lane is bit-identical; the
// optimization argument only picks the kernel.
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
func (f *FftData) CopyFromPackedArray(v *[FftLength]float32) {
	f.Re[0] = v[0]
	f.Re[FftLengthBy2] = v[1]
	f.Im[0] = 0
	f.Im[FftLengthBy2] = 0
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		f.Re[k] = v[j]
		f.Im[k] = v[j+1]
	}
}

// CopyToPackedArray packs into the Ooura input layout.
func (f *FftData) CopyToPackedArray(v *[FftLength]float32) {
	v[0] = f.Re[0]
	v[1] = f.Re[FftLengthBy2]
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		v[j] = f.Re[k]
		v[j+1] = f.Im[k]
	}
}
