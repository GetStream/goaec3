package aec3

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

// The SSE2 references keep the eight lanes of the two accumulators in
// registers and advance by half groups: the low four lanes take the first
// half of a group, the high four the second. The halves are separate basic
// blocks, which bounds the values live at once (the eight chains and the
// four products of a half) below the fifteen float registers of amd64; in
// one block the compiler computes the eight products first and spills a
// chain. The length is a multiple of 8, so the test between the halves
// never fails.

// dlyDotSSse2Ref accumulates h*x over len(h)/8 groups of 8 into the two SSE2
// accumulators (_mm_mul_ps + _mm_add_ps).
func dlyDotSSse2Ref(h, x []float32, acc *[8]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	s0, s1, s2, s3, s4, s5, s6, s7 := acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
	for k := 0; k < len(h)-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		s0 += float32(hh[0] * xx[0])
		s1 += float32(hh[1] * xx[1])
		s2 += float32(hh[2] * xx[2])
		s3 += float32(hh[3] * xx[3])
		k += 4
		if k >= len(h)-3 {
			break
		}
		hh = (*[4]float32)(h[k : k+4])
		xx = (*[4]float32)(x[k : k+4])
		s4 += float32(hh[0] * xx[0])
		s5 += float32(hh[1] * xx[1])
		s6 += float32(hh[2] * xx[2])
		s7 += float32(hh[3] * xx[3])
	}
	acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7] = s0, s1, s2, s3, s4, s5, s6, s7
}

// dlyAxpyDotSSse2Ref equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSSse2Ref(h, xb, acc). len(h) is a multiple of 8.
func dlyAxpyDotSSse2Ref(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	s0, s1, s2, s3, s4, s5, s6, s7 := acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
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
		k += 4
		if k >= len(h)-3 {
			break
		}
		hh = (*[4]float32)(h[k : k+4])
		aa = (*[4]float32)(xa[k : k+4])
		bb = (*[4]float32)(xb[k : k+4])
		h4 := hh[0] + float32(alpha*aa[0])
		h5 := hh[1] + float32(alpha*aa[1])
		h6 := hh[2] + float32(alpha*aa[2])
		h7 := hh[3] + float32(alpha*aa[3])
		hh[0], hh[1], hh[2], hh[3] = h4, h5, h6, h7
		s4 += float32(h4 * bb[0])
		s5 += float32(h5 * bb[1])
		s6 += float32(h6 * bb[2])
		s7 += float32(h7 * bb[3])
	}
	acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7] = s0, s1, s2, s3, s4, s5, s6, s7
}

// dlySqSse2Ref accumulates x*x over len(x)/8 groups of 8.
func dlySqSse2Ref(x []float32, acc *[8]float32) {
	n := len(x) &^ 7
	x = x[:n:n]
	q0, q1, q2, q3, q4, q5, q6, q7 := acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
	for k := 0; k < len(x)-3; k += 4 {
		xx := (*[4]float32)(x[k : k+4])
		q0 += float32(xx[0] * xx[0])
		q1 += float32(xx[1] * xx[1])
		q2 += float32(xx[2] * xx[2])
		q3 += float32(xx[3] * xx[3])
		k += 4
		if k >= len(x)-3 {
			break
		}
		xx = (*[4]float32)(x[k : k+4])
		q4 += float32(xx[0] * xx[0])
		q5 += float32(xx[1] * xx[1])
		q6 += float32(xx[2] * xx[2])
		q7 += float32(xx[3] * xx[3])
	}
	acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7] = q0, q1, q2, q3, q4, q5, q6, q7
}

// Software fused multiply-add of the AVX2 lane references.
//
// A fused a*b + c is s = fl64(a*b + c) rounded to float32, unless s is a
// float32 rounding boundary (fmaf.go). The loops below work on float64
// values: a lane accumulator is the float64 widening of its float32 value,
// so one step of its chain is a multiply and an add, the narrowing that
// rounds it and the widening that the next step and the boundary test
// (fmafTie) both use. The float64 products are exact, so the sums are the
// same whether or not the compiler contracts them.
//
// The loops make no call: a call inside a loop makes the compiler spill
// every live accumulator in every iteration. They test every sum and, on the
// rare flagged one, leave the loop and hand the work to a function that
// calls fmaf for every element (the From functions). The accumulators of a
// pass are stored when it ends, and a pass stores nothing else but the
// elements of h whose sums passed the test; a From function therefore
// restarts the accumulation of the pass from the stored accumulators, and
// the loop keeps no value alive for it.

// dlyDotSAvx2Ref accumulates h*x over len(h)/16 groups of 16 into the two
// AVX2 accumulators with fused multiply-add (_mm256_fmadd_ps). The sixteen
// lanes run in two passes of eight, so the accumulators stay in registers.
func dlyDotSAvx2Ref(h, x []float32, acc *[16]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	if n == 0 {
		return
	}
	q := 0
pass:
	for ; q < 16; q += 8 {
		a := (*[8]float32)(acc[q : q+8])
		s0, s1, s2, s3, s4, s5, s6, s7 := float64(a[0]), float64(a[1]), float64(a[2]), float64(a[3]), float64(a[4]), float64(a[5]), float64(a[6]), float64(a[7])
		for k := q; k < len(h)-7; k += 16 {
			hh := (*[8]float32)(h[k : k+8])
			xx := (*[8]float32)(x[k : k+8])
			u0 := float64(hh[0])*float64(xx[0]) + s0
			s0 = float64(float32(u0))
			if fmafTie(u0, s0) {
				break pass
			}
			u1 := float64(hh[1])*float64(xx[1]) + s1
			s1 = float64(float32(u1))
			if fmafTie(u1, s1) {
				break pass
			}
			u2 := float64(hh[2])*float64(xx[2]) + s2
			s2 = float64(float32(u2))
			if fmafTie(u2, s2) {
				break pass
			}
			u3 := float64(hh[3])*float64(xx[3]) + s3
			s3 = float64(float32(u3))
			if fmafTie(u3, s3) {
				break pass
			}
			u4 := float64(hh[4])*float64(xx[4]) + s4
			s4 = float64(float32(u4))
			if fmafTie(u4, s4) {
				break pass
			}
			u5 := float64(hh[5])*float64(xx[5]) + s5
			s5 = float64(float32(u5))
			if fmafTie(u5, s5) {
				break pass
			}
			u6 := float64(hh[6])*float64(xx[6]) + s6
			s6 = float64(float32(u6))
			if fmafTie(u6, s6) {
				break pass
			}
			u7 := float64(hh[7])*float64(xx[7]) + s7
			s7 = float64(float32(u7))
			if fmafTie(u7, s7) {
				break pass
			}
		}
		a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] = float32(s0), float32(s1), float32(s2), float32(s3), float32(s4), float32(s5), float32(s6), float32(s7)
	}
	if q < 16 {
		dlyDotSAvx2RefFrom(h, x, acc, q)
	}
}

// dlyDotSAvx2RefFrom finishes dlyDotSAvx2Ref from the pass that starts at
// lane q, whose accumulators acc holds as they were before the pass.
//
//go:noinline
func dlyDotSAvx2RefFrom(h, x []float32, acc *[16]float32, q int) {
	for ; q < 16; q += 8 {
		for k := q; k < len(h)-7; k += 16 {
			for j := 0; j < 8; j++ {
				acc[q+j] = fmaf(h[k+j], x[k+j], acc[q+j])
			}
		}
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
	if n == 0 {
		return
	}
	al := float64(alpha)
	// On a flagged sum, the pass has updated its elements of h before
	// element e and none after.
	q, e := 0, 0
pass:
	for ; q < 16; q += 8 {
		a := (*[8]float32)(acc[q : q+8])
		s0, s1, s2, s3, s4, s5, s6, s7 := float64(a[0]), float64(a[1]), float64(a[2]), float64(a[3]), float64(a[4]), float64(a[5]), float64(a[6]), float64(a[7])
		for k := q; k < len(h)-7; k += 16 {
			hh := (*[8]float32)(h[k : k+8])
			aa := (*[8]float32)(xa[k : k+8])
			bb := (*[8]float32)(xb[k : k+8])
			v0 := float64(aa[0])*al + float64(hh[0])
			h0 := float32(v0)
			g0 := float64(h0)
			u0 := g0*float64(bb[0]) + s0
			s0 = float64(float32(u0))
			if fmafTie(v0, g0) {
				e = k + 0
				break pass
			}
			if fmafTie(u0, s0) {
				e = k + 0
				break pass
			}
			hh[0] = h0
			v1 := float64(aa[1])*al + float64(hh[1])
			h1 := float32(v1)
			g1 := float64(h1)
			u1 := g1*float64(bb[1]) + s1
			s1 = float64(float32(u1))
			if fmafTie(v1, g1) {
				e = k + 1
				break pass
			}
			if fmafTie(u1, s1) {
				e = k + 1
				break pass
			}
			hh[1] = h1
			v2 := float64(aa[2])*al + float64(hh[2])
			h2 := float32(v2)
			g2 := float64(h2)
			u2 := g2*float64(bb[2]) + s2
			s2 = float64(float32(u2))
			if fmafTie(v2, g2) {
				e = k + 2
				break pass
			}
			if fmafTie(u2, s2) {
				e = k + 2
				break pass
			}
			hh[2] = h2
			v3 := float64(aa[3])*al + float64(hh[3])
			h3 := float32(v3)
			g3 := float64(h3)
			u3 := g3*float64(bb[3]) + s3
			s3 = float64(float32(u3))
			if fmafTie(v3, g3) {
				e = k + 3
				break pass
			}
			if fmafTie(u3, s3) {
				e = k + 3
				break pass
			}
			hh[3] = h3
			v4 := float64(aa[4])*al + float64(hh[4])
			h4 := float32(v4)
			g4 := float64(h4)
			u4 := g4*float64(bb[4]) + s4
			s4 = float64(float32(u4))
			if fmafTie(v4, g4) {
				e = k + 4
				break pass
			}
			if fmafTie(u4, s4) {
				e = k + 4
				break pass
			}
			hh[4] = h4
			v5 := float64(aa[5])*al + float64(hh[5])
			h5 := float32(v5)
			g5 := float64(h5)
			u5 := g5*float64(bb[5]) + s5
			s5 = float64(float32(u5))
			if fmafTie(v5, g5) {
				e = k + 5
				break pass
			}
			if fmafTie(u5, s5) {
				e = k + 5
				break pass
			}
			hh[5] = h5
			v6 := float64(aa[6])*al + float64(hh[6])
			h6 := float32(v6)
			g6 := float64(h6)
			u6 := g6*float64(bb[6]) + s6
			s6 = float64(float32(u6))
			if fmafTie(v6, g6) {
				e = k + 6
				break pass
			}
			if fmafTie(u6, s6) {
				e = k + 6
				break pass
			}
			hh[6] = h6
			v7 := float64(aa[7])*al + float64(hh[7])
			h7 := float32(v7)
			g7 := float64(h7)
			u7 := g7*float64(bb[7]) + s7
			s7 = float64(float32(u7))
			if fmafTie(v7, g7) {
				e = k + 7
				break pass
			}
			if fmafTie(u7, s7) {
				e = k + 7
				break pass
			}
			hh[7] = h7
		}
		a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] = float32(s0), float32(s1), float32(s2), float32(s3), float32(s4), float32(s5), float32(s6), float32(s7)
	}
	if q < 16 {
		dlyAxpyFmaDotSAvx2RefFrom(h, xa, xb, alpha, acc, q, e)
	}
}

// dlyAxpyFmaDotSAvx2RefFrom finishes dlyAxpyFmaDotSAvx2Ref from the pass
// that starts at lane q: acc holds the accumulators as they were before the
// pass, and the pass has updated its elements of h before element e (k+j
// for lane j of the group at k) and none after.
//
//go:noinline
func dlyAxpyFmaDotSAvx2RefFrom(h, xa, xb []float32, alpha float32, acc *[16]float32, q, e int) {
	for ; q < 16; q += 8 {
		for k := q; k < len(h)-7; k += 16 {
			for j := 0; j < 8; j++ {
				if k+j >= e {
					h[k+j] = fmaf(xa[k+j], alpha, h[k+j])
				}
				acc[q+j] = fmaf(h[k+j], xb[k+j], acc[q+j])
			}
		}
		e = 0
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
		u0 := hh[0] + float32(alpha*aa[0])
		u1 := hh[1] + float32(alpha*aa[1])
		u2 := hh[2] + float32(alpha*aa[2])
		u3 := hh[3] + float32(alpha*aa[3])
		hh[0], hh[1], hh[2], hh[3] = u0, u1, u2, u3
		p0, p1 := float32(u0*xx[0]), float32(u1*xx[1])
		p2, p3 := float32(u2*xx[2]), float32(u3*xx[3])
		s += ((p0 + p1) + p2) + p3
		e0 := s - y
		u4 := hh[4] + float32(alpha*aa[4])
		u5 := hh[5] + float32(alpha*aa[5])
		u6 := hh[6] + float32(alpha*aa[6])
		u7 := hh[7] + float32(alpha*aa[7])
		hh[4], hh[5], hh[6], hh[7] = u4, u5, u6, u7
		q0, q1 := float32(u4*xx[4]), float32(u5*xx[5])
		q2, q3 := float32(u6*xx[6]), float32(u7*xx[7])
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
		e := float64(t - y)
		u := e*e + float64(accErr[g])
		r := float32(u)
		if fmafTie(u, float64(r)) {
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
	al := float64(alpha)
	var s float32
	for g := range accErr {
		hh := (*[4]float32)(h[4*g : 4*g+4])
		aa := (*[4]float32)(xa[4*g : 4*g+4])
		xx := (*[4]float32)(xb[4*g : 4*g+4])
		v0 := float64(aa[0])*al + float64(hh[0])
		v1 := float64(aa[1])*al + float64(hh[1])
		v2 := float64(aa[2])*al + float64(hh[2])
		v3 := float64(aa[3])*al + float64(hh[3])
		h0, h1, h2, h3 := float32(v0), float32(v1), float32(v2), float32(v3)
		p0, p1 := float32(h0*xx[0]), float32(h1*xx[1])
		p2, p3 := float32(h2*xx[2]), float32(h3*xx[3])
		t := s + ((p0 + p1) + (p2 + p3))
		e := float64(t - y)
		u := e*e + float64(accErr[g])
		r := float32(u)
		if fmafTie(v0, float64(h0)) || fmafTie(v1, float64(h1)) || fmafTie(v2, float64(h2)) ||
			fmafTie(v3, float64(h3)) || fmafTie(u, float64(r)) {
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
	al := float64(alpha)
	k := 0
	for ; k < n-3; k += 4 {
		hh := (*[4]float32)(h[k : k+4])
		xx := (*[4]float32)(x[k : k+4])
		v0 := float64(xx[0])*al + float64(hh[0])
		v1 := float64(xx[1])*al + float64(hh[1])
		v2 := float64(xx[2])*al + float64(hh[2])
		v3 := float64(xx[3])*al + float64(hh[3])
		h0, h1, h2, h3 := float32(v0), float32(v1), float32(v2), float32(v3)
		if fmafTie(v0, float64(h0)) || fmafTie(v1, float64(h1)) || fmafTie(v2, float64(h2)) ||
			fmafTie(v3, float64(h3)) {
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

func dlyAxpyDotDotS2NeonRef(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xB []float32, accB *[4]float32) {
	dlyAxpyDotSNeonRef(hA, xaA, xbA, alphaA, accA)
	dlyDotSNeonRef(hB, xB, accB)
}
