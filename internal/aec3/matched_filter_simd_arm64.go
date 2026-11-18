//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// NEON kernels for MatchedFilterCore_NEON. Mul followed by Add compiles to
// separate FMUL/FADD, matching the unfused vmlaq_f32 of the C++. The SSE2 and
// AVX2 lanes run their pure-Go references on arm64.
//
// The loops walk shrinking slices through fixed-size array views, so the
// compiler proves every load and store in bounds. Unrolled iterations feed
// the same accumulator registers in the same group order as the C++ loop.

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyDotNeon(h, x []float32, acc *[8]float32) {
	x = x[:len(h)]
	s := archsimd.LoadFloat32x4Array((*[4]float32)(acc[0:4]))
	x2 := archsimd.LoadFloat32x4Array((*[4]float32)(acc[4:8]))
	for len(h) >= 16 && len(x) >= 16 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		x0, h0 := dlyLd4((*[4]float32)(xx[0:4])), dlyLd4((*[4]float32)(hh[0:4]))
		x1, h1 := dlyLd4((*[4]float32)(xx[4:8])), dlyLd4((*[4]float32)(hh[4:8]))
		x2v, h2 := dlyLd4((*[4]float32)(xx[8:12])), dlyLd4((*[4]float32)(hh[8:12]))
		x3, h3 := dlyLd4((*[4]float32)(xx[12:16])), dlyLd4((*[4]float32)(hh[12:16]))
		x2 = x2.Add(x0.Mul(x0))
		s = s.Add(h0.Mul(x0))
		x2 = x2.Add(x1.Mul(x1))
		s = s.Add(h1.Mul(x1))
		x2 = x2.Add(x2v.Mul(x2v))
		s = s.Add(h2.Mul(x2v))
		x2 = x2.Add(x3.Mul(x3))
		s = s.Add(h3.Mul(x3))
		h = h[16:]
		x = x[16:]
	}
	for len(h) >= 4 && len(x) >= 4 {
		xk := dlyLd4((*[4]float32)(x))
		hk := dlyLd4((*[4]float32)(h))
		x2 = x2.Add(xk.Mul(xk))
		s = s.Add(hk.Mul(xk))
		h = h[4:]
		x = x[4:]
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	x2.StoreArray((*[4]float32)(acc[4:8]))
}

// dlyAxpyDotNeon applies h += alpha*xa and then accumulates the updated h
// against xb, group by group; it equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotNeonRef(h, xb, acc) for len(h) a multiple of 4.
func dlyAxpyDotNeon(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	xa = xa[:len(h)]
	xb = xb[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := archsimd.LoadFloat32x4Array((*[4]float32)(acc[0:4]))
	x2 := archsimd.LoadFloat32x4Array((*[4]float32)(acc[4:8]))
	for len(h) >= 16 && len(xa) >= 16 && len(xb) >= 16 {
		hh := (*[16]float32)(h)
		aa := (*[16]float32)(xa)
		bb := (*[16]float32)(xb)
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		h2 := dlyLd4((*[4]float32)(hh[8:12])).Add(a.Mul(dlyLd4((*[4]float32)(aa[8:12]))))
		h3 := dlyLd4((*[4]float32)(hh[12:16])).Add(a.Mul(dlyLd4((*[4]float32)(aa[12:16]))))
		b0 := dlyLd4((*[4]float32)(bb[0:4]))
		b1 := dlyLd4((*[4]float32)(bb[4:8]))
		b2 := dlyLd4((*[4]float32)(bb[8:12]))
		b3 := dlyLd4((*[4]float32)(bb[12:16]))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h2.StoreArray((*[4]float32)(hh[8:12]))
		h3.StoreArray((*[4]float32)(hh[12:16]))
		x2 = x2.Add(b0.Mul(b0))
		s = s.Add(h0.Mul(b0))
		x2 = x2.Add(b1.Mul(b1))
		s = s.Add(h1.Mul(b1))
		x2 = x2.Add(b2.Mul(b2))
		s = s.Add(h2.Mul(b2))
		x2 = x2.Add(b3.Mul(b3))
		s = s.Add(h3.Mul(b3))
		h = h[16:]
		xa = xa[16:]
		xb = xb[16:]
	}
	for len(h) >= 4 && len(xa) >= 4 && len(xb) >= 4 {
		hp := (*[4]float32)(h)
		hk := dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(xa))))
		hk.StoreArray(hp)
		bk := dlyLd4((*[4]float32)(xb))
		x2 = x2.Add(bk.Mul(bk))
		s = s.Add(hk.Mul(bk))
		h = h[4:]
		xa = xa[4:]
		xb = xb[4:]
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	x2.StoreArray((*[4]float32)(acc[4:8]))
}

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	x = x[:len(h)]
	accErr = accErr[:len(h)/4]
	x2v := archsimd.LoadFloat32x4Array(x2)
	var s float32
	// Four groups per iteration: the SumAllElements pairwise additions of the
	// four products run in one tree of vpadds (each group keeps its
	// (p0+p1)+(p2+p3)), the running sum stays sequential, and the four
	// squared errors are added to accErr lane by lane.
	for len(h) >= 16 && len(x) >= 16 && len(accErr) >= 4 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		ap := (*[4]float32)(accErr)
		x0, h0 := dlyLd4((*[4]float32)(xx[0:4])), dlyLd4((*[4]float32)(hh[0:4]))
		x1, h1 := dlyLd4((*[4]float32)(xx[4:8])), dlyLd4((*[4]float32)(hh[4:8]))
		x2k, h2 := dlyLd4((*[4]float32)(xx[8:12])), dlyLd4((*[4]float32)(hh[8:12]))
		x3, h3 := dlyLd4((*[4]float32)(xx[12:16])), dlyLd4((*[4]float32)(hh[12:16]))
		x2v = x2v.Add(x0.Mul(x0))
		x2v = x2v.Add(x1.Mul(x1))
		x2v = x2v.Add(x2k.Mul(x2k))
		x2v = x2v.Add(x3.Mul(x3))
		r01 := h0.Mul(x0).ConcatAddPairs(h1.Mul(x1))
		r23 := h2.Mul(x2k).ConcatAddPairs(h3.Mul(x3))
		r := r01.ConcatAddPairs(r23)
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
		h = h[16:]
		x = x[16:]
		accErr = accErr[4:]
	}
	for len(h) >= 4 && len(x) >= 4 && len(accErr) >= 1 {
		xk := dlyLd4((*[4]float32)(x))
		hk := dlyLd4((*[4]float32)(h))
		x2v = x2v.Add(xk.Mul(xk))
		p := hk.Mul(xk)
		// SumAllElements: vpadd(lo, hi) then vpadd(sum, sum).
		r := p.ConcatAddPairs(p)
		r = r.ConcatAddPairs(r)
		s += r.GetElem(0)
		e := s - y
		accErr[0] += float32(e * e)
		h = h[4:]
		x = x[4:]
		accErr = accErr[1:]
	}
	x2v.StoreArray(x2)
	return s
}

func dlyAxpy(h, x []float32, alpha float32) {
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	for len(h) >= 16 && len(x) >= 16 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		h2 := dlyLd4((*[4]float32)(hh[8:12])).Add(a.Mul(dlyLd4((*[4]float32)(xx[8:12]))))
		h3 := dlyLd4((*[4]float32)(hh[12:16])).Add(a.Mul(dlyLd4((*[4]float32)(xx[12:16]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h2.StoreArray((*[4]float32)(hh[8:12]))
		h3.StoreArray((*[4]float32)(hh[12:16]))
		h = h[16:]
		x = x[16:]
	}
	for len(h) >= 4 && len(x) >= 4 {
		hp := (*[4]float32)(h)
		dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(x)))).StoreArray(hp)
		h = h[4:]
		x = x[4:]
	}
	x = x[:len(h)]
	for k := range h {
		h[k] += float32(alpha * x[k])
	}
}

func dlyDotSse2(h, x []float32, acc *[16]float32) { dlyDotSse2Ref(h, x, acc) }

func dlyDotAvx2(h, x []float32, acc *[32]float32) { dlyDotAvx2Ref(h, x, acc) }

func dlyAxpyDotSse2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyDotSse2Ref(h, xa, xb, alpha, acc)
}

func dlyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, acc)
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	return dlyAccErrSse2Ref(h, x, y, accErr, x2)
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
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

const dlyPeakSimd = true

// dlyCoreLane runs the MatchedFilterCore loop of the SIMD lane with group
// size g, with every update fused into the next dot product.
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	dlyCoreFused(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

func dlyReduceSse2(acc *[16]float32, s, x2Sum float32) (float32, float32) {
	return dlyReduceSse2Ref(acc, s, x2Sum)
}

func dlyReduceAvx2(acc *[32]float32, s, x2Sum float32) (float32, float32) {
	return dlyReduceAvx2Ref(acc, s, x2Sum)
}

// dlyZeroAcc clears the lane accumulators.
func dlyZeroAcc(acc *[32]float32) { *acc = [32]float32{} }

func dlyDotSse2Sum(h, x []float32) (s, x2Sum float32) { return dlyDotSse2SumRef(h, x) }

func dlyAxpyDotSse2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	return dlyAxpyDotSse2SumRef(h, xa, xb, alpha)
}

func dlyDotAvx2Sum(h, x []float32) (s, x2Sum float32) { return dlyDotAvx2SumRef(h, x) }

func dlyAxpyFmaDotAvx2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	return dlyAxpyFmaDotAvx2SumRef(h, xa, xb, alpha)
}
