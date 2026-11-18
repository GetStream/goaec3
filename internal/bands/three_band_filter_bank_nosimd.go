//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package bands

// ThreeBandFilterBank splits a 48 kHz frame into three 16 kHz bands and
// merges them back. The prototype low-pass filter has half bandwidth, so a
// DCT shifts it to the center frequencies [1/12, 3/12, 5/12].
type ThreeBandFilterBank struct {
	ref threeBandFilterBankRef
}

// Analysis splits in (FullBandSize samples) into NumBands downsampled
// frequency bands in out, each of SplitBandSize samples.
func (f *ThreeBandFilterBank) Analysis(in *[FullBandSize]float32, out *[NumBands][]float32) {
	f.ref.analysis(in, out)
}

// Synthesis merges the NumBands downsampled bands in (SplitBandSize samples
// each) into out (FullBandSize samples).
func (f *ThreeBandFilterBank) Synthesis(in *[NumBands][]float32, out *[FullBandSize]float32) {
	f.ref.synthesis(in, out)
}
