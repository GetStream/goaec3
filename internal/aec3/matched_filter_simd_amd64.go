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
// size g. On amd64 the update of a sample is fused into the next sample's
// dot product only when both cover h in one stretch (see dlyCoreSimple).
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
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
	return out[0]
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

// dlyReduceSse2 is dlyReduceSse2Ref with the lane sums of x2 and s side by
// side: lanes 0 and 1 of t run ((v0 + v1) + v2) + v3 for each.
func dlyReduceSse2(acc *[16]float32, s, x2Sum float32) (float32, float32) {
	if !dlyUseAvx {
		return dlyReduceSse2Ref(acc, s, x2Sum)
	}
	a := dlyLd4((*[4]float32)(acc[8:12])).Add(dlyLd4((*[4]float32)(acc[12:16]))).ToBits()
	b := dlyLd4((*[4]float32)(acc[0:4])).Add(dlyLd4((*[4]float32)(acc[4:8]))).ToBits()
	lo := a.InterleaveLo(b).BitsToFloat32() // a0 b0 a1 b1
	hi := a.InterleaveHi(b).BitsToFloat32() // a2 b2 a3 b3
	t := lo.Add(lo.ConcatPermuteScalars(2, 3, 6, 7, lo))
	t = t.Add(hi)
	t = t.Add(hi.ConcatPermuteScalars(2, 3, 6, 7, hi))
	var r [4]float32
	t.StoreArray(&r)
	return s + r[1], x2Sum + r[0]
}

// dlyReduceAvx2 is dlyReduceAvx2Ref as hsum_ab: pairwise sums of a =
// x2_sum_256 + x2_sum_256_8 and b = s_256 + s_256_8 in the reference order.
func dlyReduceAvx2(acc *[32]float32, s, x2Sum float32) (float32, float32) {
	if !dlyUseAvx {
		return dlyReduceAvx2Ref(acc, s, x2Sum)
	}
	a := dlyLd8((*[8]float32)(acc[16:24])).Add(dlyLd8((*[8]float32)(acc[24:32])))
	b := dlyLd8((*[8]float32)(acc[0:8])).Add(dlyLd8((*[8]float32)(acc[8:16])))
	p := a.ConcatAddPairsGrouped(b) // a01 a23 b01 b23 | a45 a67 b45 b67
	// t2 lanes: (a01 + a23), (b01 + b23), (a45 + a67), (b45 + b67).
	lo, hi := p.GetLo(), p.GetHi()
	t2 := lo.ConcatPermuteScalars(0, 2, 4, 6, hi).Add(lo.ConcatPermuteScalars(1, 3, 5, 7, hi))
	u := t2.Add(t2.ConcatPermuteScalars(2, 3, 6, 7, t2))
	var r [4]float32
	u.StoreArray(&r)
	archsimd.ClearAVXUpperBits()
	return s + r[1], x2Sum + r[0]
}
