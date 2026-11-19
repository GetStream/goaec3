package aec3

import "math"

// Pure-Go lane references for the MatchedFilterCore inner loops. Each one
// reproduces the arithmetic of one libwebrtc SIMD variant lane by lane; they
// are the specification the archsimd kernels are tested against and the
// fallback when no kernel exists for a lane on the build architecture.
//
// The C++ loops accumulate h*x and x*x side by side. The two sums never mix
// before the final reduction, so the kernels here compute them separately:
// the Dot kernels accumulate h*x, the Sq kernels x*x, each into the lane
// accumulators of its sum:
//   - NEON  (*[4]float32):  s_128 (or x2_sum_128).
//   - SSE2  (*[8]float32):  [0:4] s_128, [4:8] s_128_4.
//   - AVX2  (*[16]float32): [0:8] s_256, [8:16] s_256_8.
//
// Sample j of a group feeds lane j of its accumulators, and every lane is an
// independent chain over the groups in order.

// dlyDotSNeonRef accumulates h*x over len(h)/4 groups of 4 with unfused
// multiply-add (vmlaq_f32).
func dlyDotSNeonRef(h, x []float32, acc *[4]float32) {
	n := len(h) &^ 3
	h = h[:n:n]
	x = x[:n:n]
	s0, s1, s2, s3 := acc[0], acc[1], acc[2], acc[3]
	for k := 0; k < len(h)-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		s0 += float32(hh[0] * xx[0])
		s1 += float32(hh[1] * xx[1])
		s2 += float32(hh[2] * xx[2])
		s3 += float32(hh[3] * xx[3])
	}
	acc[0], acc[1], acc[2], acc[3] = s0, s1, s2, s3
}

// dlyAxpyDotSNeonRef applies h += alpha*xa (the product rounded before the
// add) and accumulates the updated h against xb, element by element; it
// equals dlyAxpyRef(h, xa, alpha) followed by dlyDotSNeonRef(h, xb, acc).
// len(h) is a multiple of 4.
func dlyAxpyDotSNeonRef(h, xa, xb []float32, alpha float32, acc *[4]float32) {
	n := len(h) &^ 3
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	s0, s1, s2, s3 := acc[0], acc[1], acc[2], acc[3]
	for k := 0; k < len(h)-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		aa := (*[4]float32)(xa[k : k+4])
		bb := (*[4]float32)(xb[k : k+4])
		h0 := hh[0] + float32(alpha*aa[0])
		h1 := hh[1] + float32(alpha*aa[1])
		h2 := hh[2] + float32(alpha*aa[2])
		h3 := hh[3] + float32(alpha*aa[3])
		hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
		s0 += float32(h0 * bb[0])
		s1 += float32(h1 * bb[1])
		s2 += float32(h2 * bb[2])
		s3 += float32(h3 * bb[3])
	}
	acc[0], acc[1], acc[2], acc[3] = s0, s1, s2, s3
}

// dlySqNeonRef accumulates x*x over len(x)/4 groups of 4 (vmlaq_f32).
func dlySqNeonRef(x []float32, acc *[4]float32) {
	n := len(x) &^ 3
	x = x[:n:n]
	q0, q1, q2, q3 := acc[0], acc[1], acc[2], acc[3]
	for k := 0; k < len(x)-3; k += 4 {
		xx := (*[4]float32)(x[k : k+4])
		q0 += float32(xx[0] * xx[0])
		q1 += float32(xx[1] * xx[1])
		q2 += float32(xx[2] * xx[2])
		q3 += float32(xx[3] * xx[3])
	}
	acc[0], acc[1], acc[2], acc[3] = q0, q1, q2, q3
}

// dlyDotSSse2Ref accumulates h*x over len(h)/8 groups of 8 into the two SSE2
// accumulators (_mm_mul_ps + _mm_add_ps). The eight lanes run in two passes
// of four (the low and the high half of every group), so the accumulators
// stay in registers.
func dlyDotSSse2Ref(h, x []float32, acc *[8]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	for half := 0; half < 8; half += 4 {
		s0, s1, s2, s3 := acc[half], acc[half+1], acc[half+2], acc[half+3]
		for k := half; k < len(h)-3; k += 8 {
			hh := (*[4]float32)(h[k : k+4])
			xx := (*[4]float32)(x[k : k+4])
			s0 += float32(hh[0] * xx[0])
			s1 += float32(hh[1] * xx[1])
			s2 += float32(hh[2] * xx[2])
			s3 += float32(hh[3] * xx[3])
		}
		acc[half], acc[half+1], acc[half+2], acc[half+3] = s0, s1, s2, s3
	}
}

// dlyAxpyDotSSse2Ref equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSSse2Ref(h, xb, acc). len(h) is a multiple of 8.
func dlyAxpyDotSSse2Ref(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	for half := 0; half < 8; half += 4 {
		s0, s1, s2, s3 := acc[half], acc[half+1], acc[half+2], acc[half+3]
		for k := half; k < len(h)-3; k += 8 {
			hh := (*[4]float32)(h[k : k+4])
			aa := (*[4]float32)(xa[k : k+4])
			bb := (*[4]float32)(xb[k : k+4])
			h0 := hh[0] + float32(alpha*aa[0])
			h1 := hh[1] + float32(alpha*aa[1])
			h2 := hh[2] + float32(alpha*aa[2])
			h3 := hh[3] + float32(alpha*aa[3])
			hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
			s0 += float32(h0 * bb[0])
			s1 += float32(h1 * bb[1])
			s2 += float32(h2 * bb[2])
			s3 += float32(h3 * bb[3])
		}
		acc[half], acc[half+1], acc[half+2], acc[half+3] = s0, s1, s2, s3
	}
}

// dlySqSse2Ref accumulates x*x over len(x)/8 groups of 8.
func dlySqSse2Ref(x []float32, acc *[8]float32) {
	n := len(x) &^ 7
	x = x[:n:n]
	for half := 0; half < 8; half += 4 {
		q0, q1, q2, q3 := acc[half], acc[half+1], acc[half+2], acc[half+3]
		for k := half; k < len(x)-3; k += 8 {
			xx := (*[4]float32)(x[k : k+4])
			q0 += float32(xx[0] * xx[0])
			q1 += float32(xx[1] * xx[1])
			q2 += float32(xx[2] * xx[2])
			q3 += float32(xx[3] * xx[3])
		}
		acc[half], acc[half+1], acc[half+2], acc[half+3] = q0, q1, q2, q3
	}
}

// Software fused multiply-add of the AVX2 lane references.
//
// fmaf (fmaf.go) computes q = fl64(a*b + c) and returns float32(q) unless q
// is a float32 rounding boundary, the only case in which rounding q differs
// from rounding the exact a*b + c. Every boundary has at most 25 significant
// bits, so its 28 low significand bits are zero:
//   - the midpoints of adjacent normal float32 values and the overflow
//     threshold 2^128 - 2^103 have exactly 25, the last of them bit 28 of
//     the float64 significand;
//   - the midpoints of adjacent subnormal float32 values, the odd multiples
//     of 2^-150 below 2^-126, have at most 24: bit 28 is zero as well.
//
// dlyFmaTry therefore reports q as a possible boundary only when its 28 low
// bits are zero and either bit 28 is set or 0 < |q| < 2^-126. The loops
// below reach the second half of that test for sums that are exact in few
// bits (zeros, small integers), and fmaf itself for about one random sum in
// 2^29.

// dlyFmaTry returns float32(q) for q = fmafSum(a, b, c) and whether q may
// be a rounding boundary, in which case the result is fmaf(a, b, c)
// instead. The product of two float32 values is exact in float64, so q is
// the same whether or not the compiler contracts the expression.
//
// The loops below make no call: a call inside a loop makes the compiler
// spill every live accumulator in every iteration. They test the flags of a
// whole group and, on the rare flagged group, store their state and hand
// the rest of the work, from that group on, to a function that calls fmaf
// for every element (the From functions).
func dlyFmaTry(a, b, c float32) (float32, bool) {
	q := float64(a)*float64(b) + float64(c)
	bits := math.Float64bits(q)
	return float32(q), bits&0x0FFFFFFF == 0 && (bits&0x10000000 != 0 || bits<<1-1 < 897<<53-1)
}

// dlyDotSAvx2Ref accumulates h*x over len(h)/16 groups of 16 into the two
// AVX2 accumulators with fused multiply-add (_mm256_fmadd_ps). The sixteen
// lanes run in four passes of four, so the accumulators stay in registers.
func dlyDotSAvx2Ref(h, x []float32, acc *[16]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	for q := 0; q < 16; q += 4 {
		a := (*[4]float32)(acc[q : q+4])
		s0, s1, s2, s3 := a[0], a[1], a[2], a[3]
		for k := q; k < len(h)-3; k += 16 {
			hh := (*[4]float32)(h[k : k+4])
			xx := (*[4]float32)(x[k : k+4])
			r0, b0 := dlyFmaTry(hh[0], xx[0], s0)
			r1, b1 := dlyFmaTry(hh[1], xx[1], s1)
			r2, b2 := dlyFmaTry(hh[2], xx[2], s2)
			r3, b3 := dlyFmaTry(hh[3], xx[3], s3)
			if b0 || b1 || b2 || b3 {
				a[0], a[1], a[2], a[3] = s0, s1, s2, s3
				dlyDotSAvx2RefFrom(h, x, acc, q, k)
				return
			}
			s0, s1, s2, s3 = r0, r1, r2, r3
		}
		a[0], a[1], a[2], a[3] = s0, s1, s2, s3
	}
}

// dlyDotSAvx2RefFrom finishes dlyDotSAvx2Ref from element k of the pass of
// lanes q..q+3, whose accumulators are in acc.
//
//go:noinline
func dlyDotSAvx2RefFrom(h, x []float32, acc *[16]float32, q, k int) {
	for ; q < 16; q += 4 {
		for ; k < len(h)-3; k += 16 {
			for j := 0; j < 4; j++ {
				acc[q+j] = fmaf(h[k+j], x[k+j], acc[q+j])
			}
		}
		k = q + 4
	}
}

// dlyAxpyFmaDotSAvx2Ref applies h = fma(xa, alpha, h) and accumulates the
// updated h against xb, element by element; it equals
// dlyAxpyFmaRef(h, xa, alpha) followed by dlyDotSAvx2Ref(h, xb, acc).
// len(h) is a multiple of 16.
func dlyAxpyFmaDotSAvx2Ref(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	for q := 0; q < 16; q += 4 {
		a := (*[4]float32)(acc[q : q+4])
		s0, s1, s2, s3 := a[0], a[1], a[2], a[3]
		for k := q; k < len(h)-3; k += 16 {
			hh := (*[4]float32)(h[k : k+4])
			aa := (*[4]float32)(xa[k : k+4])
			bb := (*[4]float32)(xb[k : k+4])
			h0, b0 := dlyFmaTry(aa[0], alpha, hh[0])
			h1, b1 := dlyFmaTry(aa[1], alpha, hh[1])
			h2, b2 := dlyFmaTry(aa[2], alpha, hh[2])
			h3, b3 := dlyFmaTry(aa[3], alpha, hh[3])
			r0, c0 := dlyFmaTry(h0, bb[0], s0)
			r1, c1 := dlyFmaTry(h1, bb[1], s1)
			r2, c2 := dlyFmaTry(h2, bb[2], s2)
			r3, c3 := dlyFmaTry(h3, bb[3], s3)
			if b0 || b1 || b2 || b3 || c0 || c1 || c2 || c3 {
				a[0], a[1], a[2], a[3] = s0, s1, s2, s3
				dlyAxpyFmaDotSAvx2RefFrom(h, xa, xb, alpha, acc, q, k)
				return
			}
			hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
			s0, s1, s2, s3 = r0, r1, r2, r3
		}
		a[0], a[1], a[2], a[3] = s0, s1, s2, s3
	}
}

// dlyAxpyFmaDotSAvx2RefFrom finishes dlyAxpyFmaDotSAvx2Ref from element k of
// the pass of lanes q..q+3, whose accumulators are in acc.
//
//go:noinline
func dlyAxpyFmaDotSAvx2RefFrom(h, xa, xb []float32, alpha float32, acc *[16]float32, q, k int) {
	for ; q < 16; q += 4 {
		for ; k < len(h)-3; k += 16 {
			for j := 0; j < 4; j++ {
				h[k+j] = fmaf(xa[k+j], alpha, h[k+j])
				acc[q+j] = fmaf(h[k+j], xb[k+j], acc[q+j])
			}
		}
		k = q + 4
	}
}

// dlySqAvx2Ref accumulates x*x over len(x)/16 groups of 16 with fused
// multiply-add.
func dlySqAvx2Ref(x []float32, acc *[16]float32) {
	x = x[:len(x)&^15]
	dlyDotSAvx2Ref(x, x, acc)
}

// dlyAccErrNeonRef runs the filter loop of
// MatchedFilterCoreWithAccumulatedError_NEON for one capture sample over
// len(h)/4 groups, without its x*x accumulation (see dlySqNeonRef): it adds
// the squared running error after every group into accErr and returns the
// filter output.
func dlyAccErrNeonRef(h, x []float32, y float32, accErr []float32) float32 {
	n := len(h) &^ 3
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := range accErr {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		xx := (*[4]float32)(x[4*g : 4*g+4])
		p0, p1 := float32(hh[0]*xx[0]), float32(hh[1]*xx[1])
		p2, p3 := float32(hh[2]*xx[2]), float32(hh[3]*xx[3])
		s += (p0 + p1) + (p2 + p3)
		e := s - y
		accErr[g] += float32(e * e)
	}
	return s
}

// dlyAxpyAccErrNeonRef applies h += alpha*xa and runs dlyAccErrNeonRef on
// the updated h against xb, group by group. len(h) is a multiple of 4.
func dlyAxpyAccErrNeonRef(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	n := len(h) &^ 3
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := range accErr {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		aa := (*[4]float32)(xa[4*g : 4*g+4])
		xx := (*[4]float32)(xb[4*g : 4*g+4])
		h0 := hh[0] + float32(alpha*aa[0])
		h1 := hh[1] + float32(alpha*aa[1])
		h2 := hh[2] + float32(alpha*aa[2])
		h3 := hh[3] + float32(alpha*aa[3])
		hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
		p0, p1 := float32(h0*xx[0]), float32(h1*xx[1])
		p2, p3 := float32(h2*xx[2]), float32(h3*xx[3])
		s += (p0 + p1) + (p2 + p3)
		e := s - y
		accErr[g] += float32(e * e)
	}
	return s
}

// dlyAccErrSse2Ref is the filter loop of
// MatchedFilterCore_AccumulatedError_SSE2 for one capture sample over
// len(h)/8 groups, without its x*x accumulation (see dlySqSse2Ref).
func dlyAccErrSse2Ref(h, x []float32, y float32, accErr []float32) float32 {
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		hh := (*[8]float32)(h[4*g : 4*g+8])
		xx := (*[8]float32)(x[4*g : 4*g+8])
		ae := (*[2]float32)(accErr[g : g+2])
		p0, p1 := float32(hh[0]*xx[0]), float32(hh[1]*xx[1])
		p2, p3 := float32(hh[2]*xx[2]), float32(hh[3]*xx[3])
		q0, q1 := float32(hh[4]*xx[4]), float32(hh[5]*xx[5])
		q2, q3 := float32(hh[6]*xx[6]), float32(hh[7]*xx[7])
		s += ((p0 + p1) + p2) + p3
		e0 := s - y
		s += ((q0 + q1) + q2) + q3
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
	}
	return s
}

// dlyAxpyAccErrSse2Ref applies h += alpha*xa and runs dlyAccErrSse2Ref on
// the updated h against xb, group by group. len(h) is a multiple of 8.
func dlyAxpyAccErrSse2Ref(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		hh := (*[8]float32)(h[4*g : 4*g+8])
		aa := (*[8]float32)(xa[4*g : 4*g+8])
		xx := (*[8]float32)(xb[4*g : 4*g+8])
		ae := (*[2]float32)(accErr[g : g+2])
		var u [8]float32
		for j := range u {
			u[j] = hh[j] + float32(alpha*aa[j])
		}
		*hh = u
		p0, p1 := float32(u[0]*xx[0]), float32(u[1]*xx[1])
		p2, p3 := float32(u[2]*xx[2]), float32(u[3]*xx[3])
		q0, q1 := float32(u[4]*xx[4]), float32(u[5]*xx[5])
		q2, q3 := float32(u[6]*xx[6]), float32(u[7]*xx[7])
		s += ((p0 + p1) + p2) + p3
		e0 := s - y
		s += ((q0 + q1) + q2) + q3
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
	}
	return s
}

// dlyAccErrAvx2Ref is the filter loop of
// MatchedFilterCore_AccumulatedError_AVX2 for one capture sample over
// len(h)/16 groups, without its x*x accumulation (see dlySqAvx2Ref). The
// hadd lane order [0],[4],[1],[5] visits the four 4-sample sub-groups in
// sequence; the products are unfused and the accumulated error is fused.
func dlyAccErrAvx2Ref(h, x []float32, y float32, accErr []float32) float32 {
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := range accErr {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		xx := (*[4]float32)(x[4*g : 4*g+4])
		p0, p1 := float32(hh[0]*xx[0]), float32(hh[1]*xx[1])
		p2, p3 := float32(hh[2]*xx[2]), float32(hh[3]*xx[3])
		t := s + ((p0 + p1) + (p2 + p3))
		e := t - y
		r, b := dlyFmaTry(e, e, accErr[g])
		if b {
			return dlyAccErrAvx2RefFrom(h, x, y, accErr, g, s)
		}
		accErr[g] = r
		s = t
	}
	return s
}

// dlyAccErrAvx2RefFrom finishes dlyAccErrAvx2Ref from sub-group g with the
// running sum s.
//
//go:noinline
func dlyAccErrAvx2RefFrom(h, x []float32, y float32, accErr []float32, g int, s float32) float32 {
	for ; g < len(accErr); g++ {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		xx := (*[4]float32)(x[4*g : 4*g+4])
		p0, p1 := float32(hh[0]*xx[0]), float32(hh[1]*xx[1])
		p2, p3 := float32(hh[2]*xx[2]), float32(hh[3]*xx[3])
		s += (p0 + p1) + (p2 + p3)
		e := s - y
		accErr[g] = fmaf(e, e, accErr[g])
	}
	return s
}

// dlyAxpyFmaAccErrAvx2Ref applies h = fma(xa, alpha, h) and runs
// dlyAccErrAvx2Ref on the updated h against xb, sub-group by sub-group.
// len(h) is a multiple of 16.
func dlyAxpyFmaAccErrAvx2Ref(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := range accErr {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		aa := (*[4]float32)(xa[4*g : 4*g+4])
		xx := (*[4]float32)(xb[4*g : 4*g+4])
		h0, b0 := dlyFmaTry(aa[0], alpha, hh[0])
		h1, b1 := dlyFmaTry(aa[1], alpha, hh[1])
		h2, b2 := dlyFmaTry(aa[2], alpha, hh[2])
		h3, b3 := dlyFmaTry(aa[3], alpha, hh[3])
		p0, p1 := float32(h0*xx[0]), float32(h1*xx[1])
		p2, p3 := float32(h2*xx[2]), float32(h3*xx[3])
		t := s + ((p0 + p1) + (p2 + p3))
		e := t - y
		r, b4 := dlyFmaTry(e, e, accErr[g])
		if b0 || b1 || b2 || b3 || b4 {
			return dlyAxpyFmaAccErrAvx2RefFrom(h, xa, xb, alpha, y, accErr, g, s)
		}
		hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
		accErr[g] = r
		s = t
	}
	return s
}

// dlyAxpyFmaAccErrAvx2RefFrom finishes dlyAxpyFmaAccErrAvx2Ref from
// sub-group g with the running sum s.
//
//go:noinline
func dlyAxpyFmaAccErrAvx2RefFrom(h, xa, xb []float32, alpha, y float32, accErr []float32, g int, s float32) float32 {
	for ; g < len(accErr); g++ {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		aa := (*[4]float32)(xa[4*g : 4*g+4])
		xx := (*[4]float32)(xb[4*g : 4*g+4])
		for j := range hh {
			hh[j] = fmaf(aa[j], alpha, hh[j])
		}
		p0, p1 := float32(hh[0]*xx[0]), float32(hh[1]*xx[1])
		p2, p3 := float32(hh[2]*xx[2]), float32(hh[3]*xx[3])
		s += (p0 + p1) + (p2 + p3)
		e := s - y
		accErr[g] = fmaf(e, e, accErr[g])
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

// dlyAxpyFmaRef computes h = fma(x, alpha, h) (_mm256_fmadd_ps).
func dlyAxpyFmaRef(h, x []float32, alpha float32) {
	n := len(h)
	x = x[:n:n]
	k := 0
	for ; k < n-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		h0, b0 := dlyFmaTry(xx[0], alpha, hh[0])
		h1, b1 := dlyFmaTry(xx[1], alpha, hh[1])
		h2, b2 := dlyFmaTry(xx[2], alpha, hh[2])
		h3, b3 := dlyFmaTry(xx[3], alpha, hh[3])
		if b0 || b1 || b2 || b3 {
			break
		}
		hh[0], hh[1], hh[2], hh[3] = h0, h1, h2, h3
	}
	if k < n {
		dlyAxpyFmaRefFrom(h, x, alpha, k)
	}
}

// dlyAxpyFmaRefFrom finishes dlyAxpyFmaRef from element k.
//
//go:noinline
func dlyAxpyFmaRefFrom(h, x []float32, alpha float32, k int) {
	for ; k < len(h); k++ {
		h[k] = fmaf(x[k], alpha, h[k])
	}
}

// The Sum references run a dot product (or update and dot product) over
// all of h from zero accumulators and reduce them.

func dlyDotSSumSse2Ref(h, x []float32) float32 {
	var acc [16]float32
	dlyDotSSse2Ref(h, x, (*[8]float32)(acc[:8]))
	return dlyReduceS(8, &acc, 0)
}

func dlyAxpyDotSSumSse2Ref(h, xa, xb []float32, alpha float32) float32 {
	var acc [16]float32
	dlyAxpyDotSSse2Ref(h, xa, xb, alpha, (*[8]float32)(acc[:8]))
	return dlyReduceS(8, &acc, 0)
}

func dlyDotSSumAvx2Ref(h, x []float32) float32 {
	var acc [16]float32
	dlyDotSAvx2Ref(h, x, &acc)
	return dlyReduceS(16, &acc, 0)
}

func dlyAxpyFmaDotSSumAvx2Ref(h, xa, xb []float32, alpha float32) float32 {
	var acc [16]float32
	dlyAxpyFmaDotSAvx2Ref(h, xa, xb, alpha, &acc)
	return dlyReduceS(16, &acc, 0)
}

// dlyDotSSum is dlyDotS over all of h from zero accumulators, reduced.
func dlyDotSSum(g int, h, x []float32) float32 {
	switch g {
	case 4:
		var acc [16]float32
		dlyDotSNeon(h, x, (*[4]float32)(acc[:4]))
		return dlyReduceS(4, &acc, 0)
	case 8:
		return dlyDotSSumSse2(h, x)
	}
	return dlyDotSSumAvx2(h, x)
}

// dlyAxpyDotSSum is dlyAxpyDotS over all of h from zero accumulators,
// reduced.
func dlyAxpyDotSSum(g int, h, xa, xb []float32, alpha float32) float32 {
	switch g {
	case 4:
		var acc [16]float32
		dlyAxpyDotSNeon(h, xa, xb, alpha, (*[4]float32)(acc[:4]))
		return dlyReduceS(4, &acc, 0)
	case 8:
		return dlyAxpyDotSSumSse2(h, xa, xb, alpha)
	}
	return dlyAxpyFmaDotSSumAvx2(h, xa, xb, alpha)
}

// The two-filter references run the one-filter references on each filter.

func dlyDotS2NeonRef(hA, xA []float32, accA *[4]float32, hB, xB []float32, accB *[4]float32) {
	dlyDotSNeonRef(hA, xA, accA)
	dlyDotSNeonRef(hB, xB, accB)
}

func dlyAxpyDotS2NeonRef(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[4]float32) {
	dlyAxpyDotSNeonRef(hA, xaA, xbA, alphaA, accA)
	dlyAxpyDotSNeonRef(hB, xaB, xbB, alphaB, accB)
}

func dlyDotS2Sse2Ref(hA, xA []float32, accA *[8]float32, hB, xB []float32, accB *[8]float32) {
	dlyDotSSse2Ref(hA, xA, accA)
	dlyDotSSse2Ref(hB, xB, accB)
}

func dlyAxpyDotS2Sse2Ref(hA, xaA, xbA []float32, alphaA float32, accA *[8]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[8]float32) {
	dlyAxpyDotSSse2Ref(hA, xaA, xbA, alphaA, accA)
	dlyAxpyDotSSse2Ref(hB, xaB, xbB, alphaB, accB)
}

func dlyDotS2Avx2Ref(hA, xA []float32, accA *[16]float32, hB, xB []float32, accB *[16]float32) {
	dlyDotSAvx2Ref(hA, xA, accA)
	dlyDotSAvx2Ref(hB, xB, accB)
}

func dlyAxpyFmaDotS2Avx2Ref(hA, xaA, xbA []float32, alphaA float32, accA *[16]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[16]float32) {
	dlyAxpyFmaDotSAvx2Ref(hA, xaA, xbA, alphaA, accA)
	dlyAxpyFmaDotSAvx2Ref(hB, xaB, xbB, alphaB, accB)
}
