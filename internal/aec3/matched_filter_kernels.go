package aec3

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
	n := len(h) &^ 3
	h = h[:n:n]
	x = x[:n:n]
	// The eight lanes are independent chains, kept in locals.
	s0, s1, s2, s3 := acc[0], acc[1], acc[2], acc[3]
	q0, q1, q2, q3 := acc[4], acc[5], acc[6], acc[7]
	for k := 0; k < len(h)-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		x0, x1, x2, x3 := xx[0], xx[1], xx[2], xx[3]
		q0 += float32(x0 * x0)
		q1 += float32(x1 * x1)
		q2 += float32(x2 * x2)
		q3 += float32(x3 * x3)
		s0 += float32(hh[0] * x0)
		s1 += float32(hh[1] * x1)
		s2 += float32(hh[2] * x2)
		s3 += float32(hh[3] * x3)
	}
	acc[0], acc[1], acc[2], acc[3] = s0, s1, s2, s3
	acc[4], acc[5], acc[6], acc[7] = q0, q1, q2, q3
}

// dlyDotSse2Ref accumulates over len(h)/8 groups of 8 into the two pairs of
// SSE2 lane accumulators (_mm_mul_ps + _mm_add_ps).
func dlyDotSse2Ref(h, x []float32, acc *[16]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	// The sixteen lanes are independent chains, kept in locals.
	s0, s1, s2, s3 := acc[0], acc[1], acc[2], acc[3]
	s4, s5, s6, s7 := acc[4], acc[5], acc[6], acc[7]
	q0, q1, q2, q3 := acc[8], acc[9], acc[10], acc[11]
	q4, q5, q6, q7 := acc[12], acc[13], acc[14], acc[15]
	for k := 0; k < len(h)-7; k += 8 {
		hh := (*[8]float32)(h[k : k+8])
		xx := (*[8]float32)(x[k : k+8])
		x0, x1, x2, x3 := xx[0], xx[1], xx[2], xx[3]
		x4, x5, x6, x7 := xx[4], xx[5], xx[6], xx[7]
		q0 += float32(x0 * x0)
		q1 += float32(x1 * x1)
		q2 += float32(x2 * x2)
		q3 += float32(x3 * x3)
		q4 += float32(x4 * x4)
		q5 += float32(x5 * x5)
		q6 += float32(x6 * x6)
		q7 += float32(x7 * x7)
		s0 += float32(hh[0] * x0)
		s1 += float32(hh[1] * x1)
		s2 += float32(hh[2] * x2)
		s3 += float32(hh[3] * x3)
		s4 += float32(hh[4] * x4)
		s5 += float32(hh[5] * x5)
		s6 += float32(hh[6] * x6)
		s7 += float32(hh[7] * x7)
	}
	acc[0], acc[1], acc[2], acc[3] = s0, s1, s2, s3
	acc[4], acc[5], acc[6], acc[7] = s4, s5, s6, s7
	acc[8], acc[9], acc[10], acc[11] = q0, q1, q2, q3
	acc[12], acc[13], acc[14], acc[15] = q4, q5, q6, q7
}

// dlyDotAvx2Ref accumulates over len(h)/16 groups of 16 into the two pairs
// of AVX2 lane accumulators with fused multiply-add (_mm256_fmadd_ps).
//
// Sample j of a group feeds s_256 (j < 8) or s_256_8 (j >= 8) lane j%8,
// which is acc[j], and the matching x2 lane acc[16+j]; each lane is an
// independent chain of fmaf, expanded inline (see fmaf).
func dlyDotAvx2Ref(h, x []float32, acc *[32]float32) {
	x = x[:len(h)]
	s := (*[16]float32)(acc[0:16])
	x2 := (*[16]float32)(acc[16:32])
	for len(h) >= 16 && len(x) >= 16 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		for j := range 16 {
			xj := xx[j]
			q := fmafSum(xj, xj, x2[j])
			r := float32(q)
			if fmafHard(q) {
				r = fmafSlow(xj, xj, x2[j])
			}
			x2[j] = r
			hj := hh[j]
			q = fmafSum(hj, xj, s[j])
			r = float32(q)
			if fmafHard(q) {
				r = fmafSlow(hj, xj, s[j])
			}
			s[j] = r
		}
		h = h[16:]
		x = x[16:]
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
	accErr = accErr[:len(h)/16*4]
	var s float32
	for len(h) >= 16 && len(x) >= 16 && len(accErr) >= 4 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		ae := (*[4]float32)(accErr)
		var p [16]float32
		for j := range 16 {
			xj := xx[j]
			q := fmafSum(xj, xj, x2[j])
			r := float32(q)
			if fmafHard(q) {
				r = fmafSlow(xj, xj, x2[j])
			}
			x2[j] = r
			p[j] = float32(hh[j] * xj)
		}
		for sub := range 4 {
			o := 4 * sub
			s += (p[o] + p[o+1]) + (p[o+2] + p[o+3])
			e := s - y
			q := fmafSum(e, e, ae[sub])
			r := float32(q)
			if fmafHard(q) {
				r = fmafSlow(e, e, ae[sub])
			}
			ae[sub] = r
		}
		h = h[16:]
		x = x[16:]
		accErr = accErr[4:]
	}
	return s
}

// dlyAxpyRef computes h += alpha*x with the product rounded before the add.
// The scalar, NEON (vmlaq_f32) and SSE2 update loops all compute this per
// element, so it serves every unfused lane.
func dlyAxpyRef(h, x []float32, alpha float32) {
	n := len(h)
	x = x[:n:n]
	k := 0
	for ; k < n-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		hh[0] += float32(alpha * xx[0])
		hh[1] += float32(alpha * xx[1])
		hh[2] += float32(alpha * xx[2])
		hh[3] += float32(alpha * xx[3])
	}
	for ; k < n; k++ {
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
	for k, xk := range x {
		q := fmafSum(xk, alpha, h[k])
		r := float32(q)
		if fmafHard(q) {
			r = fmafSlow(xk, alpha, h[k])
		}
		h[k] = r
	}
}
