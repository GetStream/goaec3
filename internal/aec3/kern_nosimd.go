//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// Kernel entry points without archsimd: the pure-Go lane references.

// fftDataSpectrum is FftData::Spectrum. Only x86 builds of libwebrtc have
// the kAvx2 case (fused); everywhere else every optimization runs the
// scalar expression.
func fftDataSpectrum(opt Aec3Optimization, re, im, out *[FftLengthBy2Plus1]float32) {
	if isAMD64 && opt == OptAvx2 {
		kernSpectrumAvx2Ref(re, im, out)
		return
	}
	fftDataSpectrumGeneric(re, im, out)
}

func vectorMultiply(x, y, z []float32) { vectorMultiplyGeneric(x, y, z) }

func vectorAccumulate(x, z []float32) { vectorAccumulateGeneric(x, z) }

// kernVectorSqrt is VectorMath::Sqrt.
func kernVectorSqrt(x []float32) { kernVectorSqrtGeneric(x) }
