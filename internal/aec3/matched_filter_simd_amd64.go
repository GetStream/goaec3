//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// x86 kernels for MatchedFilterCore_SSE2 and MatchedFilterCore_AVX2. The
// 128-bit archsimd operations are VEX encoded (AVX); VMULPS/VADDPS on XMM
// registers round exactly like MULPS/ADDPS, so they reproduce the SSE2 lane.
// The NEON lane runs its pure-Go reference on amd64.
//
// The loops index fixed-size array views of slices resliced to a common
// length and capacity (x[:n:n]) with an index bounded by len-15 (or len-7),
// so the compiler proves every access in bounds and the loop carries a
// single induction variable. fma(a, b, c) is symmetric in a and b, and
// VFMADD213PS overwrites the receiver, so the receiver of each MulAdd is the
// multiplicand that dies there.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

// dlyCoreLane runs the MatchedFilterCore loop of the SIMD lane with group
// size g. On amd64 the SSE2 and AVX2 lanes fuse the update of a sample into
// the next sample's dot product only when both cover h in one stretch (see
// dlyCoreSimple); the NEON lane, a pure-Go reference here, keeps
// dlyCoreFused.
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	if g == 4 {
		dlyCoreFused(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
		return
	}
	dlyCoreSimple(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyLd8(p *[8]float32) archsimd.Float32x8 { return archsimd.LoadFloat32x8Array(p) }

func dlyDotNeon(h, x []float32, acc *[8]float32) { dlyDotNeonRef(h, x, acc) }

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr, x2)
}

func dlyAxpyDotNeon(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotNeonRef(h, xa, xb, alpha, acc)
}

func dlyDotSse2(h, x []float32, acc *[16]float32) {
	if !dlyUseAvx {
		dlyDotSse2Ref(h, x, acc)
		return
	}
	n := len(h)
	x = x[:n:n]
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(xk))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4))
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
	x2.StoreArray((*[4]float32)(acc[8:12]))
	x24.StoreArray((*[4]float32)(acc[12:16]))
}

// dlyAxpyDotSse2 equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSse2Ref(h, xb, acc) for len(h) a multiple of 8.
func dlyAxpyDotSse2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	if !dlyUseAvx {
		dlyAxpyDotSse2Ref(h, xa, xb, alpha, acc)
		return
	}
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
	x2.StoreArray((*[4]float32)(acc[8:12]))
	x24.StoreArray((*[4]float32)(acc[12:16]))
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	if !dlyUseAvx {
		return dlyAccErrSse2Ref(h, x, y, accErr, x2)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	x2v := dlyLd4((*[4]float32)(x2[0:4]))
	x2v4 := dlyLd4((*[4]float32)(x2[4:8]))
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		i := 4 * g
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		ae := (*[2]float32)(accErr[g : g+2])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2v = x2v.Add(xk.Mul(xk))
		x2v4 = x2v4.Add(xk4.Mul(xk4))
		var p, q [4]float32
		dlyLd4((*[4]float32)(hh[0:4])).Mul(xk).StoreArray(&p)
		dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4).StoreArray(&q)
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
	}
	x2v.StoreArray((*[4]float32)(x2[0:4]))
	x2v4.StoreArray((*[4]float32)(x2[4:8]))
	return s
}

func dlyAxpy(h, x []float32, alpha float32) {
	if !dlyUseAvx {
		dlyAxpyRef(h, x, alpha)
		return
	}
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	i := 0
	for ; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(x[i : i+4])))).StoreArray(hp)
	}
	for ; i < n; i++ {
		h[i] += float32(alpha * x[i])
	}
}

func dlyDotAvx2(h, x []float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyDotAvx2Ref(h, x, acc)
		return
	}
	n := len(h)
	x = x[:n:n]
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		s = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(xk, s)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
	archsimd.ClearAVXUpperBits()
}

// dlyAxpyFmaDotAvx2 equals dlyAxpyFmaRef(h, xa, alpha) followed by
// dlyDotAvx2Ref(h, xb, acc) for len(h) a multiple of 16.
func dlyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, acc)
		return
	}
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		s = hk.MulAdd(xk, s)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
	archsimd.ClearAVXUpperBits()
}

// dlyAccErrAvx2 keeps the running sum s in lane 0 of a 128-bit register,
// so the whole loop is VEX encoded and needs no ClearAVXUpperBits until it
// ends. Each group's four sub-group sums (hadd lanes [0], [4], [1], [5])
// enter lane 0 in turn; lane 0 of each step is the scalar running sum of
// the C++ and the other lanes are unused. The four errors are then gathered
// into one vector for the fused update of accErr.
func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
	}
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	yv := archsimd.BroadcastFloat32x4(y)
	var sv archsimd.Float32x4
	x2v := dlyLd8((*[8]float32)(x2[0:8]))
	x2v8 := dlyLd8((*[8]float32)(x2[8:16]))
	for g, i := 0, 0; g < len(accErr)-3 && i < len(h)-15; g, i = g+4, i+16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		p := dlyLd8((*[8]float32)(hh[0:8])).Mul(xk)
		q := dlyLd8((*[8]float32)(hh[8:16])).Mul(xk8)
		x2v = xk.MulAdd(xk, x2v)
		x2v8 = xk8.MulAdd(xk8, x2v8)
		r := p.ConcatAddPairsGrouped(q)
		r = r.ConcatAddPairsGrouped(r)
		lo, hi := r.GetLo(), r.GetHi()
		t0 := sv.Add(lo)
		t1 := t0.Add(hi)
		t2 := t1.Add(lo.ConcatPermuteScalars(1, 1, 5, 5, lo))
		t3 := t2.Add(hi.ConcatPermuteScalars(1, 1, 5, 5, hi))
		sv = t3
		ev := t0.ConcatPermuteScalars(0, 0, 4, 4, t1).ConcatPermuteScalars(0, 2, 4, 6,
			t2.ConcatPermuteScalars(0, 0, 4, 4, t3)).Sub(yv)
		ap := (*[4]float32)(accErr[g : g+4])
		ev.MulAdd(ev, dlyLd4(ap)).StoreArray(ap)
	}
	x2v.StoreArray((*[8]float32)(x2[0:8]))
	x2v8.StoreArray((*[8]float32)(x2[8:16]))
	var out [4]float32
	sv.StoreArray(&out)
	archsimd.ClearAVXUpperBits()
	// out[0] is read with a 16-byte load from the 16-byte store at the
	// same address, which forwards.
	return out[0]
}

// dlyAccErrAvx2Sample is dlyAccErrAvx2SampleRef: dlyAccErrAvx2 from x*x
// accumulators that start in registers at zero (not loaded from memory
// stored just before) and reduced there.
func dlyAccErrAvx2Sample(h, x []float32, y float32, accErr []float32) (s, x2Sum float32) {
	if !dlyUseFma {
		return dlyAccErrAvx2SampleRef(h, x, y, accErr)
	}
	var r [4]float32
	dlyAccErrAvx2SampleLanes(h, x, y, accErr, &r)
	return r[0], 0 + r[2]
}

// dlyAccErrAvx2SampleLanes stores s in r[0] and the reduced x*x sum (before
// the C++ adds it to 0) in r[2] (see dlyReduceAvx2Lanes).
//
//go:noinline
func dlyAccErrAvx2SampleLanes(h, x []float32, y float32, accErr []float32, r *[4]float32) {
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	yv := archsimd.BroadcastFloat32x4(y)
	var sv archsimd.Float32x4
	var x2v, x2v8 archsimd.Float32x8
	for g, i := 0, 0; g < len(accErr)-3 && i < len(h)-15; g, i = g+4, i+16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		p := dlyLd8((*[8]float32)(hh[0:8])).Mul(xk)
		q := dlyLd8((*[8]float32)(hh[8:16])).Mul(xk8)
		x2v = xk.MulAdd(xk, x2v)
		x2v8 = xk8.MulAdd(xk8, x2v8)
		rr := p.ConcatAddPairsGrouped(q)
		rr = rr.ConcatAddPairsGrouped(rr)
		lo, hi := rr.GetLo(), rr.GetHi()
		t0 := sv.Add(lo)
		t1 := t0.Add(hi)
		t2 := t1.Add(lo.ConcatPermuteScalars(1, 1, 5, 5, lo))
		t3 := t2.Add(hi.ConcatPermuteScalars(1, 1, 5, 5, hi))
		sv = t3
		ev := t0.ConcatPermuteScalars(0, 0, 4, 4, t1).ConcatPermuteScalars(0, 2, 4, 6,
			t2.ConcatPermuteScalars(0, 0, 4, 4, t3)).Sub(yv)
		ap := (*[4]float32)(accErr[g : g+4])
		ev.MulAdd(ev, dlyLd4(ap)).StoreArray(ap)
	}
	// x2: a = x2_sum_256 + x2_sum_256_8, v = low half + high half, then
	// ((v0 + v1) + v2) + v3 in lane 0.
	a := x2v.Add(x2v8)
	v := a.GetLo().Add(a.GetHi())
	t := v.Add(v.ConcatPermuteScalars(1, 1, 5, 5, v))
	t = t.Add(v.ConcatPermuteScalars(2, 2, 6, 6, v))
	t = t.Add(v.ConcatPermuteScalars(3, 3, 7, 7, v))
	sv.ConcatPermuteScalars(0, 0, 4, 4, t).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	i := 0
	for ; i < n-7; i += 8 {
		hp := (*[8]float32)(h[i : i+8])
		dlyLd8((*[8]float32)(x[i:i+8])).MulAdd(a, dlyLd8(hp)).StoreArray(hp)
	}
	archsimd.ClearAVXUpperBits()
	for ; i < n; i++ {
		h[i] = fmaf(x[i], alpha, h[i])
	}
}

var dlyPeakSimd = dlyUseAvx

// dlyZeroAcc clears the lane accumulators with 256-bit stores. The AVX2
// kernels load them as 256-bit vectors, and a load cannot be forwarded from
// the two narrower stores a Go zero value is written with: it would wait
// for them to reach the cache at the start of every capture sample.
//
// It does not inline, so its ClearAVXUpperBits does not make the caller's
// float32 values cross a VZEROUPPER (see dlyReduceAvx2Lanes).
//
//go:noinline
func dlyZeroAcc(acc *[32]float32) {
	if !dlyUseAvx {
		*acc = [32]float32{}
		return
	}
	var z archsimd.Float32x8
	z.StoreArray((*[8]float32)(acc[0:8]))
	z.StoreArray((*[8]float32)(acc[8:16]))
	z.StoreArray((*[8]float32)(acc[16:24]))
	z.StoreArray((*[8]float32)(acc[24:32]))
	archsimd.ClearAVXUpperBits()
}

// dlyHsumSse2 reduces the SSE2 lane accumulators as dlyReduceSse2Ref does,
// with the x2 and s sums side by side: lanes 0 and 1 of the result run
// ((v0 + v1) + v2) + v3 for x2 and for s.
func dlyHsumSse2(s, s4, x2, x24 archsimd.Float32x4) archsimd.Float32x4 {
	a := x2.Add(x24).ToBits()
	b := s.Add(s4).ToBits()
	lo := a.InterleaveLo(b).BitsToFloat32() // a0 b0 a1 b1
	hi := a.InterleaveHi(b).BitsToFloat32() // a2 b2 a3 b3
	t := lo.Add(lo.ConcatPermuteScalars(2, 3, 6, 7, lo))
	t = t.Add(hi)
	return t.Add(hi.ConcatPermuteScalars(2, 3, 6, 7, hi))
}

// dlyHsumAvx2 reduces the AVX2 lane accumulators as hsum_ab does
// (dlyReduceAvx2Ref): pairwise sums of a = x2_sum_256 + x2_sum_256_8 and
// b = s_256 + s_256_8, the x2 sum in lane 0 and the s sum in lane 1.
func dlyHsumAvx2(s, s8, x2, x28 archsimd.Float32x8) archsimd.Float32x4 {
	p := x2.Add(x28).ConcatAddPairsGrouped(s.Add(s8)) // a01 a23 b01 b23 | a45 a67 b45 b67
	lo, hi := p.GetLo(), p.GetHi()
	// t lanes: (a01 + a23), (b01 + b23), (a45 + a67), (b45 + b67).
	t := lo.ConcatPermuteScalars(0, 2, 4, 6, hi).Add(lo.ConcatPermuteScalars(1, 3, 5, 7, hi))
	return t.Add(t.ConcatPermuteScalars(2, 3, 6, 7, t))
}

func dlyReduceSse2(acc *[16]float32, s, x2Sum float32) (float32, float32) {
	if !dlyUseAvx {
		return dlyReduceSse2Ref(acc, s, x2Sum)
	}
	t := dlyHsumSse2(dlyLd4((*[4]float32)(acc[0:4])), dlyLd4((*[4]float32)(acc[4:8])),
		dlyLd4((*[4]float32)(acc[8:12])), dlyLd4((*[4]float32)(acc[12:16])))
	r0, r1 := t.GetElem(0), t.GetElem(1)
	return s + r1, x2Sum + r0
}

func dlyReduceAvx2(acc *[32]float32, s, x2Sum float32) (float32, float32) {
	if !dlyUseAvx {
		return dlyReduceAvx2Ref(acc, s, x2Sum)
	}
	var r [4]float32
	dlyReduceAvx2Lanes(acc, &r)
	return s + r[1], x2Sum + r[0]
}

// dlyReduceAvx2Lanes stores dlyHsumAvx2 of the accumulators in r.
//
// The 256-bit kernels that produce a float32 store it to the caller's
// memory and return nothing: a float32 live across ClearAVXUpperBits is
// spilled with a 4-byte store and restored with a 16-byte load, which the
// CPU cannot forward from that store.
//
//go:noinline
func dlyReduceAvx2Lanes(acc *[32]float32, r *[4]float32) {
	dlyHsumAvx2(dlyLd8((*[8]float32)(acc[0:8])), dlyLd8((*[8]float32)(acc[8:16])),
		dlyLd8((*[8]float32)(acc[16:24])), dlyLd8((*[8]float32)(acc[24:32]))).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

// The Sum kernels run a dot product (or update and dot product) over all
// of h from zero accumulators and return dlyReduce*Ref(acc, 0, 0) without
// storing the accumulators.

func dlyDotSse2Sum(h, x []float32) (s, x2Sum float32) {
	if !dlyUseAvx {
		return dlyDotSse2SumRef(h, x)
	}
	n := len(h)
	x = x[:n:n]
	var sv, s4, x2, x24 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(xk))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4))
	}
	t := dlyHsumSse2(sv, s4, x2, x24)
	r0, r1 := t.GetElem(0), t.GetElem(1)
	return 0 + r1, 0 + r0
}

func dlyAxpyDotSse2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	if !dlyUseAvx {
		return dlyAxpyDotSse2SumRef(h, xa, xb, alpha)
	}
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	var sv, s4, x2, x24 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
	}
	t := dlyHsumSse2(sv, s4, x2, x24)
	r0, r1 := t.GetElem(0), t.GetElem(1)
	return 0 + r1, 0 + r0
}

func dlyDotAvx2Sum(h, x []float32) (s, x2Sum float32) {
	if !dlyUseFma {
		return dlyDotAvx2SumRef(h, x)
	}
	var r [4]float32
	dlyDotAvx2SumLanes(h, x, &r)
	return 0 + r[1], 0 + r[0]
}

// dlyDotAvx2SumLanes stores the reduced lane sums of dlyDotAvx2Sum in r
// (see dlyReduceAvx2Lanes).
//
//go:noinline
func dlyDotAvx2SumLanes(h, x []float32, r *[4]float32) {
	n := len(h)
	x = x[:n:n]
	var sv, s8, x2, x28 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		sv = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(xk, sv)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	dlyHsumAvx2(sv, s8, x2, x28).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFmaDotAvx2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	if !dlyUseFma {
		return dlyAxpyFmaDotAvx2SumRef(h, xa, xb, alpha)
	}
	var r [4]float32
	dlyAxpyFmaDotAvx2SumLanes(h, xa, xb, alpha, &r)
	return 0 + r[1], 0 + r[0]
}

// dlyAxpyFmaDotAvx2SumLanes stores the reduced lane sums of
// dlyAxpyFmaDotAvx2Sum in r (see dlyReduceAvx2Lanes).
//
//go:noinline
func dlyAxpyFmaDotAvx2SumLanes(h, xa, xb []float32, alpha float32, r *[4]float32) {
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	var sv, s8, x2, x28 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		sv = hk.MulAdd(xk, sv)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	dlyHsumAvx2(sv, s8, x2, x28).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

// dlyPeakLanes256 is dlyPeakLanes128 with two 256-bit vectors per 16
// samples (lanes 0-7 and 8-15, the same index residues). It reports false
// without AVX2.
func dlyPeakLanes256(p []float32, mv *[16]float32, iv *[16]int32) bool {
	if !kernHasAVX2 {
		return false
	}
	m0 := archsimd.BroadcastFloat32x8(-1)
	m1 := m0
	i0 := archsimd.BroadcastInt32x8(-1)
	i1 := i0
	cur := archsimd.LoadInt32x8Array(&[8]int32{0, 1, 2, 3, 4, 5, 6, 7})
	eight := archsimd.BroadcastInt32x8(8)
	step := archsimd.BroadcastInt32x8(16)
	n := len(p)
	for i := 0; i < n-15; i += 16 {
		pp := (*[16]float32)(p[i : i+16])
		v0 := dlyLd8((*[8]float32)(pp[0:8]))
		v1 := dlyLd8((*[8]float32)(pp[8:16]))
		v0, v1 = v0.Mul(v0), v1.Mul(v1)
		c1 := cur.Add(eight)
		g0, g1 := v0.Greater(m0), v1.Greater(m1)
		m0, i0 = v0.IfElse(g0, m0), cur.IfElse(g0, i0)
		m1, i1 = v1.IfElse(g1, m1), c1.IfElse(g1, i1)
		cur = cur.Add(step)
	}
	m0.StoreArray((*[8]float32)(mv[0:8]))
	m1.StoreArray((*[8]float32)(mv[8:16]))
	i0.StoreArray((*[8]int32)(iv[0:8]))
	i1.StoreArray((*[8]int32)(iv[8:16]))
	archsimd.ClearAVXUpperBits()
	return true
}
