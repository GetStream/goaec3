package aec3

import "math"

// Pure-Go lane references for the small vector kernels (FftData::Spectrum,
// VectorMath::Sqrt/Multiply/Accumulate).
//
// Multiply, Accumulate and Sqrt compute one correctly rounded IEEE operation
// per element in every lane (scalar, SSE2, AVX2, NEON), so they share the
// generic code in vector_math.go. FftData::Spectrum is the same in the
// scalar, SSE2 and NEON lanes (NEON uses the scalar code), but
// FftData::SpectrumAVX2 computes bins 0..63 as _mm256_fmadd_ps(r, r, i*i):
// a fused multiply-add, which kernSpectrumAvx2Ref reproduces.

// kernSpectrumAvx2Ref is FftData::SpectrumAVX2.
func kernSpectrumAvx2Ref(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k++ {
		r, i2 := re[k], float32(im[k]*im[k])
		q := fmafSum(r, r, i2)
		v := float32(q)
		if fmafHard(q) {
			v = fmafSlow(r, r, i2)
		}
		out[k] = v
	}
	out[FftLengthBy2] = float32(re[FftLengthBy2]*re[FftLengthBy2]) + float32(im[FftLengthBy2]*im[FftLengthBy2])
}

// kernVectorSqrtGeneric is VectorMath::Sqrt in every lane.
func kernVectorSqrtGeneric(x []float32) {
	for i, a := range x {
		x[i] = float32(math.Sqrt(float64(a)))
	}
}
