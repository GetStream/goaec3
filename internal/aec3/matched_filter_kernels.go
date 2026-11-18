package aec3

import "math"

// Pure-Go lane references for the MatchedFilterCore inner loops. Each one
// reproduces the arithmetic of one libwebrtc SIMD variant lane by lane; they
// are the specification the archsimd kernels are tested against and the
// fallback when no kernel exists for a lane on the build architecture.
//
// Accumulator layouts:
//   - NEON  (*[8]float32):  [0:4] s_128, [4:8] x2_sum_128.
//   - SSE2  (*[16]float32): [0:4] s_128, [4:8] s_128_4, [8:12] x2_sum_128,
//     [12:16] x2_sum_128_4.
//   - AVX2  (*[32]float32): [0:8] s_256, [8:16] s_256_8, [16:24] x2_sum_256,
//     [24:32] x2_sum_256_8.

// dlyDotNeonRef accumulates h*x and x*x over len(h)/4 groups of 4 into the
// NEON lane accumulators with unfused multiply-add (vmlaq_f32).
func dlyDotNeonRef(h, x []float32, acc *[8]float32) {
	x = x[:len(h)]
	for k := 0; k+4 <= len(h); k += 4 {
		for l := 0; l < 4; l++ {
			xk := x[k+l]
			acc[4+l] += float32(xk * xk)
			acc[l] += float32(h[k+l] * xk)
		}
	}
}

// dlyDotSse2Ref accumulates over len(h)/8 groups of 8 into the two pairs of
// SSE2 lane accumulators (_mm_mul_ps + _mm_add_ps).
func dlyDotSse2Ref(h, x []float32, acc *[16]float32) {
	x = x[:len(h)]
	for k := 0; k+8 <= len(h); k += 8 {
		for l := 0; l < 4; l++ {
			xk := x[k+l]
			xk4 := x[k+4+l]
			acc[8+l] += float32(xk * xk)
			acc[12+l] += float32(xk4 * xk4)
			acc[l] += float32(h[k+l] * xk)
			acc[4+l] += float32(h[k+4+l] * xk4)
		}
	}
}

// dlyDotAvx2Ref accumulates over len(h)/16 groups of 16 into the two pairs
// of AVX2 lane accumulators with fused multiply-add (_mm256_fmadd_ps).
func dlyDotAvx2Ref(h, x []float32, acc *[32]float32) {
	x = x[:len(h)]
	for k := 0; k+16 <= len(h); k += 16 {
		for l := 0; l < 8; l++ {
			xk := x[k+l]
			xk8 := x[k+8+l]
			acc[16+l] = dlyFmaf(xk, xk, acc[16+l])
			acc[24+l] = dlyFmaf(xk8, xk8, acc[24+l])
			acc[l] = dlyFmaf(h[k+l], xk, acc[l])
			acc[8+l] = dlyFmaf(h[k+8+l], xk8, acc[8+l])
		}
	}
}

// dlyAccErrNeonRef runs the filter loop of
// MatchedFilterCoreWithAccumulatedError_NEON for one capture sample over
// len(h)/4 groups. It accumulates x*x into the x2 lanes, adds the squared
// running error after every group into accErr and returns the filter output.
func dlyAccErrNeonRef(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	x = x[:len(h)]
	var s float32
	for g := 0; g < len(h)/4; g++ {
		k := 4 * g
		var p [4]float32
		for l := 0; l < 4; l++ {
			xk := x[k+l]
			x2[l] += float32(xk * xk)
			p[l] = float32(h[k+l] * xk)
		}
		s += (p[0] + p[1]) + (p[2] + p[3])
		e := s - y
		accErr[g] += float32(e * e)
	}
	return s
}

// dlyAccErrSse2Ref is the filter loop of
// MatchedFilterCore_AccumulatedError_SSE2 for one capture sample over
// len(h)/8 groups; x2 holds x2_sum_128 then x2_sum_128_4.
func dlyAccErrSse2Ref(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	x = x[:len(h)]
	var s float32
	for g := 0; g < len(h)/8; g++ {
		k := 8 * g
		var p, q [4]float32
		for l := 0; l < 4; l++ {
			xk := x[k+l]
			xk4 := x[k+4+l]
			x2[l] += float32(xk * xk)
			x2[4+l] += float32(xk4 * xk4)
			p[l] = float32(h[k+l] * xk)
			q[l] = float32(h[k+4+l] * xk4)
		}
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		accErr[2*g] += float32(e0 * e0)
		accErr[2*g+1] += float32(e1 * e1)
	}
	return s
}

// dlyAccErrAvx2Ref is the filter loop of
// MatchedFilterCore_AccumulatedError_AVX2 for one capture sample over
// len(h)/16 groups; x2 holds x2_sum_256 then x2_sum_256_8. The hadd lane
// order [0],[4],[1],[5] visits the four 4-sample sub-groups in sequence.
func dlyAccErrAvx2Ref(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	x = x[:len(h)]
	var s float32
	for g := 0; g < len(h)/16; g++ {
		k := 16 * g
		var p [16]float32
		for l := 0; l < 8; l++ {
			xk := x[k+l]
			xk8 := x[k+8+l]
			x2[l] = dlyFmaf(xk, xk, x2[l])
			x2[8+l] = dlyFmaf(xk8, xk8, x2[8+l])
			p[l] = float32(h[k+l] * xk)
			p[8+l] = float32(h[k+8+l] * xk8)
		}
		for sub := 0; sub < 4; sub++ {
			o := 4 * sub
			s += (p[o] + p[o+1]) + (p[o+2] + p[o+3])
			e := s - y
			accErr[4*g+sub] = dlyFmaf(e, e, accErr[4*g+sub])
		}
	}
	return s
}

// dlyAxpyRef computes h += alpha*x with the product rounded before the add.
// The scalar, NEON (vmlaq_f32) and SSE2 update loops all compute this per
// element, so it serves every unfused lane.
func dlyAxpyRef(h, x []float32, alpha float32) {
	x = x[:len(h)]
	for k := range h {
		h[k] += float32(alpha * x[k])
	}
}

// dlyAxpyDotNeonRef applies h += alpha*xa and then runs dlyDotNeonRef on
// the updated h against xb. len(h) is a multiple of 4.
func dlyAxpyDotNeonRef(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyRef(h, xa, alpha)
	dlyDotNeonRef(h, xb, acc)
}

// dlyAxpyDotSse2Ref applies h += alpha*xa and then runs dlyDotSse2Ref on
// the updated h against xb. len(h) is a multiple of 8.
func dlyAxpyDotSse2Ref(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyRef(h, xa, alpha)
	dlyDotSse2Ref(h, xb, acc)
}

// dlyAxpyFmaDotAvx2Ref applies h = fma(xa, alpha, h) and then runs
// dlyDotAvx2Ref on the updated h against xb. len(h) is a multiple of 16.
func dlyAxpyFmaDotAvx2Ref(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	dlyAxpyFmaRef(h, xa, alpha)
	dlyDotAvx2Ref(h, xb, acc)
}

// dlyAxpyFmaRef computes h = fma(x, alpha, h) (_mm256_fmadd_ps).
func dlyAxpyFmaRef(h, x []float32, alpha float32) {
	x = x[:len(h)]
	for k := range h {
		h[k] = dlyFmaf(x[k], alpha, h[k])
	}
}

// dlyFmaf returns a*b+c rounded once to float32. The float64 product of two
// float32 values is exact; the sum is computed with TwoSum and rounded to
// odd at float64 precision, which makes the final rounding to float32
// correct (float64 has more than 24+2 significand bits).
func dlyFmaf(a, b, c float32) float32 {
	p := float64(float64(a) * float64(b))
	cc := float64(c)
	s := float64(p + cc)
	bb := float64(s - p)
	err := float64(float64(p-float64(s-bb)) + float64(cc-bb))
	if err != 0 && !math.IsInf(s, 0) {
		bits := math.Float64bits(s)
		if bits&1 == 0 {
			if (err > 0) == (s > 0) {
				bits++
			} else {
				bits--
			}
			s = math.Float64frombits(bits)
		}
	}
	return float32(s)
}
