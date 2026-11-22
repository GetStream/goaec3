package aec3

import "math"

// Pure-Go lane references of FftData::Spectrum and VectorMath.
//
// Multiply, Accumulate and Sqrt are one IEEE operation per element in every
// lane. Spectrum is the same in the scalar, SSE2 and NEON lanes;
// FftData::SpectrumAVX2 computes bins 0..63 as _mm256_fmadd_ps(r, r, i*i).

// kernSpectrumAvx2Ref is FftData::SpectrumAVX2.
//
// The loop makes no call (see fmaf): the first group of four bins with a sum
// that needs fmafSlow hands the rest of the spectrum to
// kernSpectrumAvx2From.
func kernSpectrumAvx2Ref(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 4 {
		r0, r1, r2, r3 := re[k], re[k+1], re[k+2], re[k+3]
		i0, i1 := float32(im[k]*im[k]), float32(im[k+1]*im[k+1])
		i2, i3 := float32(im[k+2]*im[k+2]), float32(im[k+3]*im[k+3])
		q0, q1, q2, q3 := fmafSum(r0, r0, i0), fmafSum(r1, r1, i1), fmafSum(r2, r2, i2), fmafSum(r3, r3, i3)
		if fmafHard(q0) || fmafHard(q1) || fmafHard(q2) || fmafHard(q3) {
			kernSpectrumAvx2From(re, im, out, k)
			return
		}
		out[k], out[k+1], out[k+2], out[k+3] = float32(q0), float32(q1), float32(q2), float32(q3)
	}
	out[FftLengthBy2] = float32(re[FftLengthBy2]*re[FftLengthBy2]) + float32(im[FftLengthBy2]*im[FftLengthBy2])
}

// kernSpectrumAvx2From is FftData::SpectrumAVX2 for bins from..64, one fmaf
// per bin.
//
//go:noinline
func kernSpectrumAvx2From(re, im, out *[FftLengthBy2Plus1]float32, from int) {
	for k := from; k < FftLengthBy2; k++ {
		out[k] = fmaf(re[k], re[k], float32(im[k]*im[k]))
	}
	out[FftLengthBy2] = float32(re[FftLengthBy2]*re[FftLengthBy2]) + float32(im[FftLengthBy2]*im[FftLengthBy2])
}

// kernVectorSqrtGeneric is VectorMath::Sqrt of every lane.
func kernVectorSqrtGeneric(x []float32) {
	for i, a := range x {
		x[i] = float32(math.Sqrt(float64(a)))
	}
}
