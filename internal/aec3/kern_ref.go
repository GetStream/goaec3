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
		out[k] = kernFmaf(re[k], re[k], float32(im[k]*im[k]))
	}
	out[FftLengthBy2] = float32(re[FftLengthBy2]*re[FftLengthBy2]) + float32(im[FftLengthBy2]*im[FftLengthBy2])
}

// kernFmaf returns a*b+c rounded once to float32, like fmaf / vfmadd.
//
// a*b is exact in float64 (48-bit significand). The sum with c is computed
// with an exact error term (TwoSum) and rounded to odd, which makes the final
// rounding to float32 correct (53 >= 24+2 bits); float32(math.FMA(...)) would
// round twice.
func kernFmaf(a, b, c float32) float32 {
	p := float64(a) * float64(b)
	q := float64(c)
	s := p + q
	if math.IsInf(s, 0) || math.IsNaN(s) || s == 0 {
		// Overflow cannot happen for float32 operands; Inf/NaN propagate, and
		// an exact zero needs no rounding (its sign follows IEEE addition).
		return float32(s)
	}
	// TwoSum: s + e == p + q exactly.
	bv := s - p
	e := (p - (s - bv)) + (q - bv)
	if e != 0 {
		bits := math.Float64bits(s)
		if bits&1 == 0 {
			// Round to odd: move s one ulp toward the exact value.
			if (e > 0) == (s > 0) {
				bits++
			} else {
				bits--
			}
			s = math.Float64frombits(bits)
		}
	}
	return float32(s)
}

// kernVectorSqrtGeneric is VectorMath::Sqrt in every lane.
func kernVectorSqrtGeneric(x []float32) {
	for i, a := range x {
		x[i] = float32(math.Sqrt(float64(a)))
	}
}
