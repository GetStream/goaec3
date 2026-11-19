//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// NEON kernels for MatchedFilterCore_NEON. Mul followed by Add compiles to
// separate FMUL/FADD, matching the unfused vmlaq_f32 of the C++. The SSE2 and
// AVX2 lanes run their pure-Go references on arm64.
//
// The loops index fixed-size array views of slices resliced to a common
// length, so the compiler proves every load and store in bounds and the loop
// carries a single induction variable. Unrolled iterations feed the same
// accumulator registers in the same group order as the C++ loop.

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyDotSNeon(h, x []float32, acc *[4]float32) {
	n := len(h) &^ 3
	h = h[:n:n]
	x = x[:n:n]
	s := dlyLd4(acc)
	i := 0
	for ; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		p0 := dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4])))
		p1 := dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8])))
		p2 := dlyLd4((*[4]float32)(hh[8:12])).Mul(dlyLd4((*[4]float32)(xx[8:12])))
		p3 := dlyLd4((*[4]float32)(hh[12:16])).Mul(dlyLd4((*[4]float32)(xx[12:16])))
		s = s.Add(p0).Add(p1).Add(p2).Add(p3)
	}
	for ; i < n-3; i += 4 {
		s = s.Add(dlyLd4((*[4]float32)(h[i : i+4])).Mul(dlyLd4((*[4]float32)(x[i : i+4]))))
	}
	s.StoreArray(acc)
}

// dlyAxpyDotSNeon applies h += alpha*xa and then accumulates the updated h
// against xb, group by group; it equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSNeonRef(h, xb, acc) for len(h) a multiple of 4.
func dlyAxpyDotSNeon(h, xa, xb []float32, alpha float32, acc *[4]float32) {
	n := len(h) &^ 3
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := dlyLd4(acc)
	i := 0
	for ; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		h2 := dlyLd4((*[4]float32)(hh[8:12])).Add(a.Mul(dlyLd4((*[4]float32)(aa[8:12]))))
		h3 := dlyLd4((*[4]float32)(hh[12:16])).Add(a.Mul(dlyLd4((*[4]float32)(aa[12:16]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h2.StoreArray((*[4]float32)(hh[8:12]))
		h3.StoreArray((*[4]float32)(hh[12:16]))
		p0 := h0.Mul(dlyLd4((*[4]float32)(bb[0:4])))
		p1 := h1.Mul(dlyLd4((*[4]float32)(bb[4:8])))
		p2 := h2.Mul(dlyLd4((*[4]float32)(bb[8:12])))
		p3 := h3.Mul(dlyLd4((*[4]float32)(bb[12:16])))
		s = s.Add(p0).Add(p1).Add(p2).Add(p3)
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		hk := dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(xa[i : i+4]))))
		hk.StoreArray(hp)
		s = s.Add(hk.Mul(dlyLd4((*[4]float32)(xb[i : i+4]))))
	}
	s.StoreArray(acc)
}

// dlyDotS2Neon is dlyDotSNeon for two filters of the same length at once:
// each filter's groups reach its own accumulator in order, and the two
// accumulator chains run side by side.
func dlyDotS2Neon(hA, xA []float32, accA *[4]float32, hB, xB []float32, accB *[4]float32) {
	n := len(hA) &^ 3
	hA, xA, hB, xB = hA[:n:n], xA[:n:n], hB[:n:n], xB[:n:n]
	sA := dlyLd4(accA)
	sB := dlyLd4(accB)
	i := 0
	for ; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		bb := (*[8]float32)(xA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		dd := (*[8]float32)(xB[i : i+8])
		p0 := dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(bb[0:4])))
		p1 := dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(bb[4:8])))
		q0 := dlyLd4((*[4]float32)(gg[0:4])).Mul(dlyLd4((*[4]float32)(dd[0:4])))
		q1 := dlyLd4((*[4]float32)(gg[4:8])).Mul(dlyLd4((*[4]float32)(dd[4:8])))
		sA = sA.Add(p0).Add(p1)
		sB = sB.Add(q0).Add(q1)
	}
	for ; i < n-3; i += 4 {
		sA = sA.Add(dlyLd4((*[4]float32)(hA[i : i+4])).Mul(dlyLd4((*[4]float32)(xA[i : i+4]))))
		sB = sB.Add(dlyLd4((*[4]float32)(hB[i : i+4])).Mul(dlyLd4((*[4]float32)(xB[i : i+4]))))
	}
	sA.StoreArray(accA)
	sB.StoreArray(accB)
}

// dlyAxpyDotS2Neon is dlyAxpyDotSNeon for two filters of the same length at
// once.
func dlyAxpyDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[4]float32) {
	n := len(hA) &^ 3
	hA, xaA, xbA, hB, xaB, xbB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xaB[:n:n], xbB[:n:n]
	aA := archsimd.BroadcastFloat32x4(alphaA)
	aB := archsimd.BroadcastFloat32x4(alphaB)
	sA := dlyLd4(accA)
	sB := dlyLd4(accB)
	i := 0
	for ; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		aa := (*[8]float32)(xaA[i : i+8])
		bb := (*[8]float32)(xbA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		cc := (*[8]float32)(xaB[i : i+8])
		dd := (*[8]float32)(xbB[i : i+8])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		g0 := dlyLd4((*[4]float32)(gg[0:4])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[0:4]))))
		g1 := dlyLd4((*[4]float32)(gg[4:8])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[4:8]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		g0.StoreArray((*[4]float32)(gg[0:4]))
		g1.StoreArray((*[4]float32)(gg[4:8]))
		p0 := h0.Mul(dlyLd4((*[4]float32)(bb[0:4])))
		p1 := h1.Mul(dlyLd4((*[4]float32)(bb[4:8])))
		q0 := g0.Mul(dlyLd4((*[4]float32)(dd[0:4])))
		q1 := g1.Mul(dlyLd4((*[4]float32)(dd[4:8])))
		sA = sA.Add(p0).Add(p1)
		sB = sB.Add(q0).Add(q1)
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(hA[i : i+4])
		hk := dlyLd4(hp).Add(aA.Mul(dlyLd4((*[4]float32)(xaA[i : i+4]))))
		hk.StoreArray(hp)
		sA = sA.Add(hk.Mul(dlyLd4((*[4]float32)(xbA[i : i+4]))))
		gp := (*[4]float32)(hB[i : i+4])
		gk := dlyLd4(gp).Add(aB.Mul(dlyLd4((*[4]float32)(xaB[i : i+4]))))
		gk.StoreArray(gp)
		sB = sB.Add(gk.Mul(dlyLd4((*[4]float32)(xbB[i : i+4]))))
	}
	sA.StoreArray(accA)
	sB.StoreArray(accB)
}

// dlyAxpyDotDotS2Neon is dlyAxpyDotSNeon on the filter A and dlyDotSNeon
// on the filter B, of the same length, at once.
func dlyAxpyDotDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xB []float32, accB *[4]float32) {
	n := len(hA) &^ 3
	hA, xaA, xbA, hB, xB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xB[:n:n]
	aA := archsimd.BroadcastFloat32x4(alphaA)
	sA := dlyLd4(accA)
	sB := dlyLd4(accB)
	i := 0
	for ; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		aa := (*[8]float32)(xaA[i : i+8])
		bb := (*[8]float32)(xbA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		dd := (*[8]float32)(xB[i : i+8])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		p0 := h0.Mul(dlyLd4((*[4]float32)(bb[0:4])))
		p1 := h1.Mul(dlyLd4((*[4]float32)(bb[4:8])))
		q0 := dlyLd4((*[4]float32)(gg[0:4])).Mul(dlyLd4((*[4]float32)(dd[0:4])))
		q1 := dlyLd4((*[4]float32)(gg[4:8])).Mul(dlyLd4((*[4]float32)(dd[4:8])))
		sA = sA.Add(p0).Add(p1)
		sB = sB.Add(q0).Add(q1)
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(hA[i : i+4])
		hk := dlyLd4(hp).Add(aA.Mul(dlyLd4((*[4]float32)(xaA[i : i+4]))))
		hk.StoreArray(hp)
		sA = sA.Add(hk.Mul(dlyLd4((*[4]float32)(xbA[i : i+4]))))
		sB = sB.Add(dlyLd4((*[4]float32)(hB[i : i+4])).Mul(dlyLd4((*[4]float32)(xB[i : i+4]))))
	}
	sA.StoreArray(accA)
	sB.StoreArray(accB)
}

// dlyPairNeon selects the two-filter kernels for the NEON lane.
var dlyPairNeon = true

func dlySqNeon(x []float32, acc *[4]float32) {
	n := len(x) &^ 3
	x = x[:n:n]
	q := dlyLd4(acc)
	for i := 0; i < n-3; i += 4 {
		xk := dlyLd4((*[4]float32)(x[i : i+4]))
		q = q.Add(xk.Mul(xk))
	}
	q.StoreArray(acc)
}

// dlySq5Neon runs dlySqNeonRef on the five stretches x[offs[t]:offs[t]+n]
// in one loop, so that their chains overlap in time; acc[t] holds the
// accumulators of stretch t.
func dlySq5Neon(x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	n &^= 3
	x0 := x[offs[0]:][:n:n]
	x1 := x[offs[1]:][:n:n]
	x2 := x[offs[2]:][:n:n]
	x3 := x[offs[3]:][:n:n]
	x4 := x[offs[4]:][:n:n]
	q0 := dlyLd4((*[4]float32)(acc[0][0:4]))
	q1 := dlyLd4((*[4]float32)(acc[1][0:4]))
	q2 := dlyLd4((*[4]float32)(acc[2][0:4]))
	q3 := dlyLd4((*[4]float32)(acc[3][0:4]))
	q4 := dlyLd4((*[4]float32)(acc[4][0:4]))
	for i := 0; i < n-3; i += 4 {
		v0 := dlyLd4((*[4]float32)(x0[i : i+4]))
		v1 := dlyLd4((*[4]float32)(x1[i : i+4]))
		v2 := dlyLd4((*[4]float32)(x2[i : i+4]))
		v3 := dlyLd4((*[4]float32)(x3[i : i+4]))
		v4 := dlyLd4((*[4]float32)(x4[i : i+4]))
		q0 = q0.Add(v0.Mul(v0))
		q1 = q1.Add(v1.Mul(v1))
		q2 = q2.Add(v2.Mul(v2))
		q3 = q3.Add(v3.Mul(v3))
		q4 = q4.Add(v4.Mul(v4))
	}
	q0.StoreArray((*[4]float32)(acc[0][0:4]))
	q1.StoreArray((*[4]float32)(acc[1][0:4]))
	q2.StoreArray((*[4]float32)(acc[2][0:4]))
	q3.StoreArray((*[4]float32)(acc[3][0:4]))
	q4.StoreArray((*[4]float32)(acc[4][0:4]))
}

// dlySqStreams is the number of stretches dlySqMulti accumulates at once:
// five for the NEON lane, whose kernel it is; the other lanes run their
// references.
func dlySqStreams(g int) int {
	if g == 4 {
		return 5
	}
	return 2
}

func dlySqMulti(g int, x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	if g == 4 {
		dlySq5Neon(x, n, offs, acc)
		return
	}
	dlySqMultiRef(g, x, n, offs, acc, 2)
}

// dlySqSlideNeon stores in out[a], for a < 16, the lane chain that starts
// at x[a] (see dlyX2LanesRef), and in out[16:20] the four chains that start
// at x[last:]. The chains of lanes 4t..4t+3 add the squares of the vectors
// x[4m:4m+4] for m = t, t+1, ...: each vector is loaded and squared once and
// then feeds the four accumulators in consecutive iterations. len(x) is at
// least n+12 and last+n.
func dlySqSlideNeon(x []float32, n, last int, out *[20]float32) {
	n &^= 3
	xs := x[: n+12 : n+12]
	xa := x[last:][:n:n]
	var q0, q1, q2, q3, qa archsimd.Float32x4
	v := dlyLd4((*[4]float32)(xs[0:4]))
	s0 := v.Mul(v)
	v = dlyLd4((*[4]float32)(xs[4:8]))
	s1 := v.Mul(v)
	v = dlyLd4((*[4]float32)(xs[8:12]))
	s2 := v.Mul(v)
	for i := 0; i < n-3; i += 4 {
		v3 := dlyLd4((*[4]float32)(xs[i+12 : i+16]))
		va := dlyLd4((*[4]float32)(xa[i : i+4]))
		s3 := v3.Mul(v3)
		q0 = q0.Add(s0)
		q1 = q1.Add(s1)
		q2 = q2.Add(s2)
		q3 = q3.Add(s3)
		qa = qa.Add(va.Mul(va))
		s0, s1, s2 = s1, s2, s3
	}
	q0.StoreArray((*[4]float32)(out[0:4]))
	q1.StoreArray((*[4]float32)(out[4:8]))
	q2.StoreArray((*[4]float32)(out[8:12]))
	q3.StoreArray((*[4]float32)(out[12:16]))
	qa.StoreArray((*[4]float32)(out[16:20]))
}

// dlyX2Lanes is dlyX2LanesRef; the NEON lane computes five groups of four
// chains at a time.
func dlyX2Lanes(g int, x []float32, n int, lanes []float32) {
	if g != 4 {
		dlyX2LanesRef(g, x, n, lanes)
		return
	}
	last := len(lanes) - 4
	if n >= 4 && last >= 12 && last <= 16 && len(x) >= n+12 {
		var out [20]float32
		dlySqSlideNeon(x, n, last, &out)
		copy(lanes[:16], out[:16])
		copy(lanes[last:], out[16:20])
		return
	}
	dlyX2LanesMulti(g, x, n, lanes)
}

// dlySqSlide2Neon is dlySqSlideNeon on two stretches at once.
func dlySqSlide2Neon(xA, xB []float32, n, last int, outA, outB *[20]float32) {
	n &^= 3
	xs := xA[: n+12 : n+12]
	xa := xA[last:][:n:n]
	ys := xB[: n+12 : n+12]
	ya := xB[last:][:n:n]
	var q0, q1, q2, q3, qa, r0, r1, r2, r3, ra archsimd.Float32x4
	v := dlyLd4((*[4]float32)(xs[0:4]))
	s0 := v.Mul(v)
	v = dlyLd4((*[4]float32)(xs[4:8]))
	s1 := v.Mul(v)
	v = dlyLd4((*[4]float32)(xs[8:12]))
	s2 := v.Mul(v)
	v = dlyLd4((*[4]float32)(ys[0:4]))
	t0 := v.Mul(v)
	v = dlyLd4((*[4]float32)(ys[4:8]))
	t1 := v.Mul(v)
	v = dlyLd4((*[4]float32)(ys[8:12]))
	t2 := v.Mul(v)
	for i := 0; i < n-3; i += 4 {
		v3 := dlyLd4((*[4]float32)(xs[i+12 : i+16]))
		va := dlyLd4((*[4]float32)(xa[i : i+4]))
		w3 := dlyLd4((*[4]float32)(ys[i+12 : i+16]))
		wa := dlyLd4((*[4]float32)(ya[i : i+4]))
		s3 := v3.Mul(v3)
		t3 := w3.Mul(w3)
		q0 = q0.Add(s0)
		q1 = q1.Add(s1)
		q2 = q2.Add(s2)
		q3 = q3.Add(s3)
		qa = qa.Add(va.Mul(va))
		r0 = r0.Add(t0)
		r1 = r1.Add(t1)
		r2 = r2.Add(t2)
		r3 = r3.Add(t3)
		ra = ra.Add(wa.Mul(wa))
		s0, s1, s2 = s1, s2, s3
		t0, t1, t2 = t1, t2, t3
	}
	q0.StoreArray((*[4]float32)(outA[0:4]))
	q1.StoreArray((*[4]float32)(outA[4:8]))
	q2.StoreArray((*[4]float32)(outA[8:12]))
	q3.StoreArray((*[4]float32)(outA[12:16]))
	qa.StoreArray((*[4]float32)(outA[16:20]))
	r0.StoreArray((*[4]float32)(outB[0:4]))
	r1.StoreArray((*[4]float32)(outB[4:8]))
	r2.StoreArray((*[4]float32)(outB[8:12]))
	r3.StoreArray((*[4]float32)(outB[12:16]))
	ra.StoreArray((*[4]float32)(outB[16:20]))
}

// dlyX2Lanes2 is dlyX2Lanes2Ref; the NEON lane interleaves the chains of the
// two stretches.
func dlyX2Lanes2(g int, xA, xB []float32, n int, lanesA, lanesB []float32) {
	last := len(lanesA) - 4
	if g != 4 || len(lanesB) != len(lanesA) || n < 4 || last < 12 || last > 16 || len(xA) < n+12 || len(xB) < n+12 {
		dlyX2Lanes2Ref(g, xA, xB, n, lanesA, lanesB)
		return
	}
	var outA, outB [20]float32
	dlySqSlide2Neon(xA, xB, n, last, &outA, &outB)
	copy(lanesA[:16], outA[:16])
	copy(lanesA[last:], outA[16:20])
	copy(lanesB[:16], outB[:16])
	copy(lanesB[last:], outB[16:20])
}

// dlyX2Wrapped is dlyX2WrappedRef; the NEON lane accumulates five windows
// at a time.
func dlyX2Wrapped(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	if g != 4 {
		dlyX2WrappedRef(g, x, hSize, lo, r, out)
		return
	}
	dlyX2WrappedMulti(g, x, hSize, lo, r, out)
}

// dlyAccErrNeon keeps four groups per iteration: the SumAllElements
// pairwise additions of the four products run in one tree of vpadds (each
// group keeps its (p0+p1)+(p2+p3)), the running sum stays sequential, and
// the four squared errors are added to accErr lane by lane.
func dlyAccErrNeon(h, x []float32, y float32, accErr []float32) float32 {
	n := len(h) &^ 3
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	i := 0
	for ; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		ap := (*[4]float32)(accErr[i/4 : i/4+4])
		p0 := dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4])))
		p1 := dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8])))
		p2 := dlyLd4((*[4]float32)(hh[8:12])).Mul(dlyLd4((*[4]float32)(xx[8:12])))
		p3 := dlyLd4((*[4]float32)(hh[12:16])).Mul(dlyLd4((*[4]float32)(xx[12:16])))
		r := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		var e [4]float32
		s += r.GetElem(0)
		e[0] = s - y
		s += r.GetElem(1)
		e[1] = s - y
		s += r.GetElem(2)
		e[2] = s - y
		s += r.GetElem(3)
		e[3] = s - y
		ev := archsimd.LoadFloat32x4Array(&e)
		dlyLd4(ap).Add(ev.Mul(ev)).StoreArray(ap)
	}
	for ; i < n-3; i += 4 {
		p := dlyLd4((*[4]float32)(h[i : i+4])).Mul(dlyLd4((*[4]float32)(x[i : i+4])))
		// SumAllElements: vpadd(lo, hi) then vpadd(sum, sum).
		r := p.ConcatAddPairs(p)
		r = r.ConcatAddPairs(r)
		s += r.GetElem(0)
		e := s - y
		accErr[i/4] += float32(e * e)
	}
	return s
}

// dlyAxpyAccErrNeon is dlyAccErrNeon with h += alpha*xa applied to every
// group before its products.
func dlyAxpyAccErrNeon(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	n := len(h) &^ 3
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: n/4 : n/4]
	a := archsimd.BroadcastFloat32x4(alpha)
	var s float32
	i := 0
	for ; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		xx := (*[16]float32)(xb[i : i+16])
		ap := (*[4]float32)(accErr[i/4 : i/4+4])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		h2 := dlyLd4((*[4]float32)(hh[8:12])).Add(a.Mul(dlyLd4((*[4]float32)(aa[8:12]))))
		h3 := dlyLd4((*[4]float32)(hh[12:16])).Add(a.Mul(dlyLd4((*[4]float32)(aa[12:16]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h2.StoreArray((*[4]float32)(hh[8:12]))
		h3.StoreArray((*[4]float32)(hh[12:16]))
		p0 := h0.Mul(dlyLd4((*[4]float32)(xx[0:4])))
		p1 := h1.Mul(dlyLd4((*[4]float32)(xx[4:8])))
		p2 := h2.Mul(dlyLd4((*[4]float32)(xx[8:12])))
		p3 := h3.Mul(dlyLd4((*[4]float32)(xx[12:16])))
		r := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		var e [4]float32
		s += r.GetElem(0)
		e[0] = s - y
		s += r.GetElem(1)
		e[1] = s - y
		s += r.GetElem(2)
		e[2] = s - y
		s += r.GetElem(3)
		e[3] = s - y
		ev := archsimd.LoadFloat32x4Array(&e)
		dlyLd4(ap).Add(ev.Mul(ev)).StoreArray(ap)
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		hk := dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(xa[i : i+4]))))
		hk.StoreArray(hp)
		p := hk.Mul(dlyLd4((*[4]float32)(xb[i : i+4])))
		r := p.ConcatAddPairs(p)
		r = r.ConcatAddPairs(r)
		s += r.GetElem(0)
		e := s - y
		accErr[i/4] += float32(e * e)
	}
	return s
}

func dlyX2Reduce(g int, accErr bool, lanes *[dlyX2Run + 15]float32, r int, out *[dlyX2Run]float32) {
	dlyX2ReduceRef(g, accErr, lanes, r, out)
}

func dlyAxpy(h, x []float32, alpha float32) {
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	i := 0
	for ; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		h2 := dlyLd4((*[4]float32)(hh[8:12])).Add(a.Mul(dlyLd4((*[4]float32)(xx[8:12]))))
		h3 := dlyLd4((*[4]float32)(hh[12:16])).Add(a.Mul(dlyLd4((*[4]float32)(xx[12:16]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h2.StoreArray((*[4]float32)(hh[8:12]))
		h3.StoreArray((*[4]float32)(hh[12:16]))
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(x[i : i+4])))).StoreArray(hp)
	}
	for ; i < n; i++ {
		h[i] += float32(alpha * x[i])
	}
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	for len(h) >= 4 && len(x) >= 4 {
		hp := (*[4]float32)(h)
		dlyLd4((*[4]float32)(x)).MulAdd(a, dlyLd4(hp)).StoreArray(hp)
		h = h[4:]
		x = x[4:]
	}
	x = x[:len(h)]
	for k := range h {
		h[k] = fmaf(x[k], alpha, h[k])
	}
}

// The SSE2 and AVX2 lanes run their pure-Go references on arm64.

func dlyDotSSse2(h, x []float32, acc *[8]float32) { dlyDotSSse2Ref(h, x, acc) }

func dlyAxpyDotSSse2(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotSSse2Ref(h, xa, xb, alpha, acc)
}

func dlySqSse2(x []float32, acc *[8]float32) { dlySqSse2Ref(x, acc) }

func dlyDotSAvx2(h, x []float32, acc *[16]float32) { dlyDotSAvx2Ref(h, x, acc) }

func dlyAxpyFmaDotSAvx2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyFmaDotSAvx2Ref(h, xa, xb, alpha, acc)
}

func dlySqAvx2(x []float32, acc *[16]float32) { dlySqAvx2Ref(x, acc) }

func dlyDotSSumSse2(h, x []float32) float32 { return dlyDotSSumSse2Ref(h, x) }

func dlyAxpyDotSSumSse2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyDotSSumSse2Ref(h, xa, xb, alpha)
}

func dlyDotSSumAvx2(h, x []float32) float32 { return dlyDotSSumAvx2Ref(h, x) }

func dlyAxpyFmaDotSSumAvx2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyFmaDotSSumAvx2Ref(h, xa, xb, alpha)
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrSse2Ref(h, x, y, accErr)
}

func dlyAxpyAccErrSse2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyAccErrSse2Ref(h, xa, xb, alpha, y, accErr)
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrAvx2Ref(h, x, y, accErr)
}

func dlyAxpyFmaAccErrAvx2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyFmaAccErrAvx2Ref(h, xa, xb, alpha, y, accErr)
}

const dlyPeakSimd = true

// dlyZeroAcc clears the lane accumulators.
func dlyZeroAcc(acc *[16]float32) { *acc = [16]float32{} }

// dlyPeakLanes256 is the 256-bit dlyPeakLanes128 (NEON has no 256-bit
// registers).
func dlyPeakLanes256(p []float32, mv *[16]float32, iv *[16]int32) bool { return false }

// dlyLanesPureGo selects what serves the pure-Go references and the
// archsimd kernels have no use for (the tables of squares of the x*x sums).
const dlyLanesPureGo = false

// dlyCoreLaneKernel runs the MatchedFilterCore loop of the lane with group
// size g on one filter where the lane has its own core, and reports whether
// it did; no lane of this build has one.
func dlyCoreLaneKernel(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) bool {
	return false
}
