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
//
// The 256-bit kernels that produce a float32 store it to the caller's
// memory from a function that does not inline, and return nothing: a
// float32 live across ClearAVXUpperBits is spilled with a 4-byte store and
// restored with a 16-byte load, which the CPU cannot forward from that
// store.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyLd8(p *[8]float32) archsimd.Float32x8 { return archsimd.LoadFloat32x8Array(p) }

// The NEON lane runs its pure-Go references on amd64.

func dlyDotSNeon(h, x []float32, acc *[4]float32) { dlyDotSNeonRef(h, x, acc) }

func dlyAxpyDotSNeon(h, xa, xb []float32, alpha float32, acc *[4]float32) {
	dlyAxpyDotSNeonRef(h, xa, xb, alpha, acc)
}

func dlySqNeon(x []float32, acc *[4]float32) { dlySqNeonRef(x, acc) }

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr)
}

func dlyAxpyAccErrNeon(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyAccErrNeonRef(h, xa, xb, alpha, y, accErr)
}

// dlyPairNeon selects the two-filter kernels for the NEON lane; they exist
// as archsimd kernels on arm64 only.
var dlyPairNeon = false

func dlyDotS2Neon(hA, xA []float32, accA *[4]float32, hB, xB []float32, accB *[4]float32) {
	dlyDotS2NeonRef(hA, xA, accA, hB, xB, accB)
}

func dlyAxpyDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[4]float32) {
	dlyAxpyDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

// dlyX2Wrapped is dlyX2WrappedRef; the SSE2 and AVX2 lanes accumulate
// several windows at a time.
func dlyX2Wrapped(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	if (g == 8 && dlyUseAvx) || (g == 16 && dlyUseFma) {
		dlyX2WrappedMulti(g, x, hSize, lo, r, out)
		return
	}
	dlyX2WrappedRef(g, x, hSize, lo, r, out)
}

// dlySqStreams is the number of stretches dlySqMulti accumulates at once:
// three for the SSE2 lane and two for the AVX2 lane.
func dlySqStreams(g int) int {
	if g == 8 {
		return 3
	}
	return 2
}

func dlySqMulti(g int, x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	switch {
	case g == 8 && dlyUseAvx:
		dlySq3Sse2(x, n, offs, acc)
	case g == 16 && dlyUseFma:
		dlySq2Avx2(x, n, offs, acc)
	default:
		dlySqMultiRef(g, x, n, offs, acc, dlySqStreams(g))
	}
}

// SSE2 lane.

func dlyDotSSse2(h, x []float32, acc *[8]float32) {
	if !dlyUseAvx {
		dlyDotSSse2Ref(h, x, acc)
		return
	}
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		s = s.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8]))))
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
}

// dlyAxpyDotSSse2 equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSSse2Ref(h, xb, acc) for len(h) a multiple of 8.
func dlyAxpyDotSSse2(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	if !dlyUseAvx {
		dlyAxpyDotSSse2Ref(h, xa, xb, alpha, acc)
		return
	}
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		s = s.Add(hk.Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		s4 = s4.Add(hk4.Mul(dlyLd4((*[4]float32)(bb[4:8]))))
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
}

func dlySqSse2(x []float32, acc *[8]float32) {
	if !dlyUseAvx {
		dlySqSse2Ref(x, acc)
		return
	}
	n := len(x) &^ 7
	x = x[:n:n]
	q := dlyLd4((*[4]float32)(acc[0:4]))
	q4 := dlyLd4((*[4]float32)(acc[4:8]))
	for i := 0; i < n-7; i += 8 {
		xx := (*[8]float32)(x[i : i+8])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		q = q.Add(xk.Mul(xk))
		q4 = q4.Add(xk4.Mul(xk4))
	}
	q.StoreArray((*[4]float32)(acc[0:4]))
	q4.StoreArray((*[4]float32)(acc[4:8]))
}

// dlySq3Sse2 runs dlySqSse2Ref on the three stretches x[offs[t]:offs[t]+n]
// in one loop, so that their chains overlap in time; acc[t] holds the
// accumulators of stretch t.
func dlySq3Sse2(x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	n &^= 7
	x0 := x[offs[0]:][:n:n]
	x1 := x[offs[1]:][:n:n]
	x2 := x[offs[2]:][:n:n]
	q0, q04 := dlyLd4((*[4]float32)(acc[0][0:4])), dlyLd4((*[4]float32)(acc[0][4:8]))
	q1, q14 := dlyLd4((*[4]float32)(acc[1][0:4])), dlyLd4((*[4]float32)(acc[1][4:8]))
	q2, q24 := dlyLd4((*[4]float32)(acc[2][0:4])), dlyLd4((*[4]float32)(acc[2][4:8]))
	for i := 0; i < n-7; i += 8 {
		a := (*[8]float32)(x0[i : i+8])
		b := (*[8]float32)(x1[i : i+8])
		c := (*[8]float32)(x2[i : i+8])
		a0, a4 := dlyLd4((*[4]float32)(a[0:4])), dlyLd4((*[4]float32)(a[4:8]))
		b0, b4 := dlyLd4((*[4]float32)(b[0:4])), dlyLd4((*[4]float32)(b[4:8]))
		c0, c4 := dlyLd4((*[4]float32)(c[0:4])), dlyLd4((*[4]float32)(c[4:8]))
		q0 = q0.Add(a0.Mul(a0))
		q04 = q04.Add(a4.Mul(a4))
		q1 = q1.Add(b0.Mul(b0))
		q14 = q14.Add(b4.Mul(b4))
		q2 = q2.Add(c0.Mul(c0))
		q24 = q24.Add(c4.Mul(c4))
	}
	q0.StoreArray((*[4]float32)(acc[0][0:4]))
	q04.StoreArray((*[4]float32)(acc[0][4:8]))
	q1.StoreArray((*[4]float32)(acc[1][0:4]))
	q14.StoreArray((*[4]float32)(acc[1][4:8]))
	q2.StoreArray((*[4]float32)(acc[2][0:4]))
	q24.StoreArray((*[4]float32)(acc[2][4:8]))
}

// dlySq2Avx2 runs dlySqAvx2Ref on the two stretches x[offs[t]:offs[t]+n] in
// one loop; acc[t] holds the accumulators of stretch t.
//
//go:noinline
func dlySq2Avx2(x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	n &^= 15
	x0 := x[offs[0]:][:n:n]
	x1 := x[offs[1]:][:n:n]
	q0, q08 := dlyLd8((*[8]float32)(acc[0][0:8])), dlyLd8((*[8]float32)(acc[0][8:16]))
	q1, q18 := dlyLd8((*[8]float32)(acc[1][0:8])), dlyLd8((*[8]float32)(acc[1][8:16]))
	for i := 0; i < n-15; i += 16 {
		a := (*[16]float32)(x0[i : i+16])
		b := (*[16]float32)(x1[i : i+16])
		a0, a8 := dlyLd8((*[8]float32)(a[0:8])), dlyLd8((*[8]float32)(a[8:16]))
		b0, b8 := dlyLd8((*[8]float32)(b[0:8])), dlyLd8((*[8]float32)(b[8:16]))
		q0 = a0.MulAdd(a0, q0)
		q08 = a8.MulAdd(a8, q08)
		q1 = b0.MulAdd(b0, q1)
		q18 = b8.MulAdd(b8, q18)
	}
	q0.StoreArray((*[8]float32)(acc[0][0:8]))
	q08.StoreArray((*[8]float32)(acc[0][8:16]))
	q1.StoreArray((*[8]float32)(acc[1][0:8]))
	q18.StoreArray((*[8]float32)(acc[1][8:16]))
	archsimd.ClearAVXUpperBits()
}

// dlyX2Lanes is dlyX2LanesRef with the chains of several groups of lanes
// accumulated in one loop: three groups of eight for the SSE2 lane, two
// groups of sixteen for the AVX2 lane.
func dlyX2Lanes(g int, x []float32, n int, lanes []float32) {
	if (g == 8 && dlyUseAvx) || (g == 16 && dlyUseFma) {
		dlyX2LanesMulti(g, x, n, lanes)
		return
	}
	dlyX2LanesRef(g, x, n, lanes)
}

// dlyHsumSSse2 reduces the two SSE2 accumulators of one sum as dlyReduceS
// does: lane 0 of the result runs ((v0 + v1) + v2) + v3 over v = s + s4.
func dlyHsumSSse2(s, s4 archsimd.Float32x4) archsimd.Float32x4 {
	v := s.Add(s4)
	t := v.Add(v.ConcatPermuteScalars(1, 1, 5, 5, v))
	t = t.Add(v.ConcatPermuteScalars(2, 2, 6, 6, v))
	return t.Add(v.ConcatPermuteScalars(3, 3, 7, 7, v))
}

// dlyHsumSAvx2 reduces the two AVX2 accumulators of one sum as hsum_ab
// does: lane 0 of the result is ((b0+b1) + (b2+b3)) + ((b4+b5) + (b6+b7))
// over b = s + s8.
func dlyHsumSAvx2(s, s8 archsimd.Float32x8) archsimd.Float32x4 {
	b := s.Add(s8)
	p := b.ConcatAddPairsGrouped(b) // b01 b23 b01 b23 | b45 b67 b45 b67
	lo, hi := p.GetLo(), p.GetHi()
	u := lo.Add(lo.ConcatPermuteScalars(1, 1, 5, 5, lo))
	w := hi.Add(hi.ConcatPermuteScalars(1, 1, 5, 5, hi))
	return u.Add(w)
}

// dlyX2Reduce is dlyX2ReduceRef with the reductions of the SSE2 and AVX2
// lanes in registers.
func dlyX2Reduce(g int, accErr bool, lanes *[dlyX2Run + 15]float32, r int, out *[dlyX2Run]float32) {
	switch {
	case g == 8 && dlyUseAvx:
		// The accumulated-error core reduces x*x like h*x in this lane.
		for j := 0; j < r && j < len(lanes)-7; j++ {
			l := (*[8]float32)(lanes[j : j+8])
			out[j] = 0 + dlyHsumSSse2(dlyLd4((*[4]float32)(l[0:4])), dlyLd4((*[4]float32)(l[4:8]))).GetElem(0)
		}
	case g == 16 && dlyUseFma:
		dlyX2ReduceAvx2(accErr, lanes, r, out)
	default:
		dlyX2ReduceRef(g, accErr, lanes, r, out)
	}
}

// dlyX2ReduceAvx2 is dlyX2Reduce for the AVX2 lane. The reduced lanes of
// sample j land in lane 0 of a 128-bit store at tmp[j] (the stores of later
// samples overwrite the other lanes), and the scalar addition to the zero
// x2_sum of the C++ runs after the VZEROUPPER.
//
//go:noinline
func dlyX2ReduceAvx2(accErr bool, lanes *[dlyX2Run + 15]float32, r int, out *[dlyX2Run]float32) {
	var tmp [dlyX2Run + 3]float32
	r = min(r, dlyX2Run)
	for j := 0; j < r; j++ {
		l := (*[16]float32)(lanes[j : j+16])
		lo, hi := dlyLd8((*[8]float32)(l[0:8])), dlyLd8((*[8]float32)(l[8:16]))
		var v archsimd.Float32x4
		if accErr {
			// v = low half + high half of lo+hi, then ((v0 + v1) + v2) + v3.
			a := lo.Add(hi)
			w := a.GetLo().Add(a.GetHi())
			v = w.Add(w.ConcatPermuteScalars(1, 1, 5, 5, w))
			v = v.Add(w.ConcatPermuteScalars(2, 2, 6, 6, w))
			v = v.Add(w.ConcatPermuteScalars(3, 3, 7, 7, w))
		} else {
			v = dlyHsumSAvx2(lo, hi)
		}
		v.StoreArray((*[4]float32)(tmp[j : j+4]))
	}
	archsimd.ClearAVXUpperBits()
	for j := 0; j < r; j++ {
		out[j] = 0 + tmp[j]
	}
}

// The Sum kernels run a dot product (or update and dot product) over all
// of h from zero accumulators and return dlyReduceS(acc, 0) without storing
// the accumulators.

func dlyDotSSumSse2(h, x []float32) float32 {
	if !dlyUseAvx {
		return dlyDotSSumSse2Ref(h, x)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	var s, s4 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		s = s.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8]))))
	}
	return 0 + dlyHsumSSse2(s, s4).GetElem(0)
}

func dlyAxpyDotSSumSse2(h, xa, xb []float32, alpha float32) float32 {
	if !dlyUseAvx {
		return dlyAxpyDotSSumSse2Ref(h, xa, xb, alpha)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	var s, s4 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		s = s.Add(hk.Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		s4 = s4.Add(hk4.Mul(dlyLd4((*[4]float32)(bb[4:8]))))
	}
	return 0 + dlyHsumSSse2(s, s4).GetElem(0)
}

// dlyAccErrSse2 gathers the two sub-group sums of a group with haddps-style
// permutes; the running sum and the errors stay scalar.
func dlyAccErrSse2(h, x []float32, y float32, accErr []float32) float32 {
	if !dlyUseAvx {
		return dlyAccErrSse2Ref(h, x, y, accErr)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		i := 4 * g
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		ae := (*[2]float32)(accErr[g : g+2])
		var p, q [4]float32
		dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4]))).StoreArray(&p)
		dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8]))).StoreArray(&q)
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
	}
	return s
}

// dlyAxpyAccErrSse2 is dlyAccErrSse2 with h += alpha*xa applied to every
// group before its products.
func dlyAxpyAccErrSse2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	if !dlyUseAvx {
		return dlyAxpyAccErrSse2Ref(h, xa, xb, alpha, y, accErr)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: n/4 : n/4]
	a := archsimd.BroadcastFloat32x4(alpha)
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		i := 4 * g
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		xx := (*[8]float32)(xb[i : i+8])
		ae := (*[2]float32)(accErr[g : g+2])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		var p, q [4]float32
		hk.Mul(dlyLd4((*[4]float32)(xx[0:4]))).StoreArray(&p)
		hk4.Mul(dlyLd4((*[4]float32)(xx[4:8]))).StoreArray(&q)
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
	}
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

// AVX2 lane.

func dlyDotSAvx2(h, x []float32, acc *[16]float32) {
	if !dlyUseFma {
		dlyDotSAvx2Ref(h, x, acc)
		return
	}
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		s = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(dlyLd8((*[8]float32)(xx[0:8])), s)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(dlyLd8((*[8]float32)(xx[8:16])), s8)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	archsimd.ClearAVXUpperBits()
}

// dlyAxpyFmaDotSAvx2 equals dlyAxpyFmaRef(h, xa, alpha) followed by
// dlyDotSAvx2Ref(h, xb, acc) for len(h) a multiple of 16.
func dlyAxpyFmaDotSAvx2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	if !dlyUseFma {
		dlyAxpyFmaDotSAvx2Ref(h, xa, xb, alpha, acc)
		return
	}
	dlyAxpyFmaDotSAvx2Lanes(h, xa, xb, alpha, acc)
}

// dlyAxpyFmaDotSAvx2Lanes is dlyAxpyFmaDotSAvx2; it does not inline, so
// alpha does not cross the VZEROUPPER in its caller as a float32.
//
//go:noinline
func dlyAxpyFmaDotSAvx2Lanes(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		s = hk.MulAdd(dlyLd8((*[8]float32)(bb[0:8])), s)
		s8 = hk8.MulAdd(dlyLd8((*[8]float32)(bb[8:16])), s8)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	archsimd.ClearAVXUpperBits()
}

func dlySqAvx2(x []float32, acc *[16]float32) {
	if !dlyUseFma {
		dlySqAvx2Ref(x, acc)
		return
	}
	n := len(x) &^ 15
	x = x[:n:n]
	q := dlyLd8((*[8]float32)(acc[0:8]))
	q8 := dlyLd8((*[8]float32)(acc[8:16]))
	for i := 0; i < n-15; i += 16 {
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		q = xk.MulAdd(xk, q)
		q8 = xk8.MulAdd(xk8, q8)
	}
	q.StoreArray((*[8]float32)(acc[0:8]))
	q8.StoreArray((*[8]float32)(acc[8:16]))
	archsimd.ClearAVXUpperBits()
}

func dlyDotSSumAvx2(h, x []float32) float32 {
	if !dlyUseFma {
		return dlyDotSSumAvx2Ref(h, x)
	}
	var r [4]float32
	dlyDotSSumAvx2Lanes(h, x, &r)
	return 0 + r[0]
}

// dlyDotSSumAvx2Lanes stores the reduced sum of dlyDotSSumAvx2 in r[0].
//
//go:noinline
func dlyDotSSumAvx2Lanes(h, x []float32, r *[4]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	var s, s8 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		s = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(dlyLd8((*[8]float32)(xx[0:8])), s)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(dlyLd8((*[8]float32)(xx[8:16])), s8)
	}
	dlyHsumSAvx2(s, s8).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFmaDotSSumAvx2(h, xa, xb []float32, alpha float32) float32 {
	if !dlyUseFma {
		return dlyAxpyFmaDotSSumAvx2Ref(h, xa, xb, alpha)
	}
	var r [4]float32
	dlyAxpyFmaDotSSumAvx2Lanes(h, xa, xb, alpha, &r)
	return 0 + r[0]
}

// dlyAxpyFmaDotSSumAvx2Lanes stores the reduced sum of
// dlyAxpyFmaDotSSumAvx2 in r[0].
//
//go:noinline
func dlyAxpyFmaDotSSumAvx2Lanes(h, xa, xb []float32, alpha float32, r *[4]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	var s, s8 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		s = hk.MulAdd(dlyLd8((*[8]float32)(bb[0:8])), s)
		s8 = hk8.MulAdd(dlyLd8((*[8]float32)(bb[8:16])), s8)
	}
	dlyHsumSAvx2(s, s8).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr)
	}
	var r [4]float32
	dlyAccErrAvx2Lanes(h, x, y, accErr, &r)
	return r[0]
}

// dlyAccErrAvx2Lanes stores the filter output of dlyAccErrAvx2 in r[0]. It
// keeps the running sum s in lane 0 of a 128-bit register. Each group's
// four sub-group sums (hadd lanes [0], [4], [1], [5]) enter lane 0 in turn;
// lane 0 of each step is the scalar running sum of the C++ and the other
// lanes are unused. The four errors are then gathered into one vector for
// the fused update of accErr.
//
//go:noinline
func dlyAccErrAvx2Lanes(h, x []float32, y float32, accErr []float32, r *[4]float32) {
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	yv := archsimd.BroadcastFloat32x4(y)
	var sv archsimd.Float32x4
	for g, i := 0, 0; g < len(accErr)-3 && i < len(h)-15; g, i = g+4, i+16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		p := dlyLd8((*[8]float32)(hh[0:8])).Mul(dlyLd8((*[8]float32)(xx[0:8])))
		q := dlyLd8((*[8]float32)(hh[8:16])).Mul(dlyLd8((*[8]float32)(xx[8:16])))
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
	sv.StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFmaAccErrAvx2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	if !dlyUseFma {
		return dlyAxpyFmaAccErrAvx2Ref(h, xa, xb, alpha, y, accErr)
	}
	var r [4]float32
	dlyAxpyFmaAccErrAvx2Lanes(h, xa, xb, alpha, y, accErr, &r)
	return r[0]
}

// dlyAxpyFmaAccErrAvx2Lanes is dlyAccErrAvx2Lanes with h = fma(xa, alpha, h)
// applied to every group before its products.
//
//go:noinline
func dlyAxpyFmaAccErrAvx2Lanes(h, xa, xb []float32, alpha, y float32, accErr []float32, r *[4]float32) {
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	a := archsimd.BroadcastFloat32x8(alpha)
	yv := archsimd.BroadcastFloat32x4(y)
	var sv archsimd.Float32x4
	for g, i := 0, 0; g < len(accErr)-3 && i < len(h)-15; g, i = g+4, i+16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		xx := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		p := hk.Mul(dlyLd8((*[8]float32)(xx[0:8])))
		q := hk8.Mul(dlyLd8((*[8]float32)(xx[8:16])))
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
	sv.StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	x = x[:len(h)]
	i := dlyAxpyFma256(h, x, alpha)
	for ; i < len(h); i++ {
		h[i] = fmaf(x[i], alpha, h[i])
	}
}

// dlyAxpyFma256 runs dlyAxpyFma over the whole groups of 8 and returns
// their length. It does not inline, so alpha does not cross the
// VZEROUPPER as a float32.
//
//go:noinline
func dlyAxpyFma256(h, x []float32, alpha float32) int {
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	i := 0
	for ; i < n-31; i += 32 {
		hh := (*[32]float32)(h[i : i+32])
		xx := (*[32]float32)(x[i : i+32])
		h0 := dlyLd8((*[8]float32)(xx[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		h1 := dlyLd8((*[8]float32)(xx[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		h2 := dlyLd8((*[8]float32)(xx[16:24])).MulAdd(a, dlyLd8((*[8]float32)(hh[16:24])))
		h3 := dlyLd8((*[8]float32)(xx[24:32])).MulAdd(a, dlyLd8((*[8]float32)(hh[24:32])))
		h0.StoreArray((*[8]float32)(hh[0:8]))
		h1.StoreArray((*[8]float32)(hh[8:16]))
		h2.StoreArray((*[8]float32)(hh[16:24]))
		h3.StoreArray((*[8]float32)(hh[24:32]))
	}
	for ; i < n-7; i += 8 {
		hp := (*[8]float32)(h[i : i+8])
		dlyLd8((*[8]float32)(x[i:i+8])).MulAdd(a, dlyLd8(hp)).StoreArray(hp)
	}
	archsimd.ClearAVXUpperBits()
	return i
}

var dlyPeakSimd = dlyUseAvx

// dlyZeroAcc clears the lane accumulators with 256-bit stores. The AVX2
// kernels load them as 256-bit vectors, and a load cannot be forwarded from
// the narrower stores a Go zero value is written with: it would wait for
// them to reach the cache at the start of every capture sample.
//
// It does not inline, so its ClearAVXUpperBits does not make the caller's
// float32 values cross a VZEROUPPER.
//
//go:noinline
func dlyZeroAcc(acc *[16]float32) {
	if !dlyUseAvx {
		*acc = [16]float32{}
		return
	}
	var z archsimd.Float32x8
	z.StoreArray((*[8]float32)(acc[0:8]))
	z.StoreArray((*[8]float32)(acc[8:16]))
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

// dlyPairSse2 and dlyPairAvx2 select the two-filter kernels for the SSE2
// and AVX2 lanes.
var dlyPairSse2, dlyPairAvx2 = false, false

// dlyDotS2Sse2 is dlyDotSSse2 for two filters of the same length at once.
func dlyDotS2Sse2(hA, xA []float32, accA *[8]float32, hB, xB []float32, accB *[8]float32) {
	if !dlyUseAvx {
		dlyDotS2Sse2Ref(hA, xA, accA, hB, xB, accB)
		return
	}
	n := len(hA) &^ 7
	hA, xA, hB, xB = hA[:n:n], xA[:n:n], hB[:n:n], xB[:n:n]
	sA := dlyLd4((*[4]float32)(accA[0:4]))
	sA4 := dlyLd4((*[4]float32)(accA[4:8]))
	sB := dlyLd4((*[4]float32)(accB[0:4]))
	sB4 := dlyLd4((*[4]float32)(accB[4:8]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		bb := (*[8]float32)(xA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		dd := (*[8]float32)(xB[i : i+8])
		sA = sA.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		sA4 = sA4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(bb[4:8]))))
		sB = sB.Add(dlyLd4((*[4]float32)(gg[0:4])).Mul(dlyLd4((*[4]float32)(dd[0:4]))))
		sB4 = sB4.Add(dlyLd4((*[4]float32)(gg[4:8])).Mul(dlyLd4((*[4]float32)(dd[4:8]))))
	}
	sA.StoreArray((*[4]float32)(accA[0:4]))
	sA4.StoreArray((*[4]float32)(accA[4:8]))
	sB.StoreArray((*[4]float32)(accB[0:4]))
	sB4.StoreArray((*[4]float32)(accB[4:8]))
}

// dlyAxpyDotS2Sse2 is dlyAxpyDotSSse2 for two filters of the same length at
// once.
func dlyAxpyDotS2Sse2(hA, xaA, xbA []float32, alphaA float32, accA *[8]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[8]float32) {
	if !dlyUseAvx {
		dlyAxpyDotS2Sse2Ref(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
		return
	}
	n := len(hA) &^ 7
	hA, xaA, xbA, hB, xaB, xbB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xaB[:n:n], xbB[:n:n]
	aA := archsimd.BroadcastFloat32x4(alphaA)
	aB := archsimd.BroadcastFloat32x4(alphaB)
	sA := dlyLd4((*[4]float32)(accA[0:4]))
	sA4 := dlyLd4((*[4]float32)(accA[4:8]))
	sB := dlyLd4((*[4]float32)(accB[0:4]))
	sB4 := dlyLd4((*[4]float32)(accB[4:8]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		aa := (*[8]float32)(xaA[i : i+8])
		bb := (*[8]float32)(xbA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		cc := (*[8]float32)(xaB[i : i+8])
		dd := (*[8]float32)(xbB[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		gk := dlyLd4((*[4]float32)(gg[0:4])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[0:4]))))
		gk4 := dlyLd4((*[4]float32)(gg[4:8])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		gk.StoreArray((*[4]float32)(gg[0:4]))
		gk4.StoreArray((*[4]float32)(gg[4:8]))
		sA = sA.Add(hk.Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		sA4 = sA4.Add(hk4.Mul(dlyLd4((*[4]float32)(bb[4:8]))))
		sB = sB.Add(gk.Mul(dlyLd4((*[4]float32)(dd[0:4]))))
		sB4 = sB4.Add(gk4.Mul(dlyLd4((*[4]float32)(dd[4:8]))))
	}
	sA.StoreArray((*[4]float32)(accA[0:4]))
	sA4.StoreArray((*[4]float32)(accA[4:8]))
	sB.StoreArray((*[4]float32)(accB[0:4]))
	sB4.StoreArray((*[4]float32)(accB[4:8]))
}

// dlyDotS2Avx2 is dlyDotSAvx2 for two filters of the same length at once.
func dlyDotS2Avx2(hA, xA []float32, accA *[16]float32, hB, xB []float32, accB *[16]float32) {
	if !dlyUseFma {
		dlyDotS2Avx2Ref(hA, xA, accA, hB, xB, accB)
		return
	}
	n := len(hA) &^ 15
	hA, xA, hB, xB = hA[:n:n], xA[:n:n], hB[:n:n], xB[:n:n]
	sA := dlyLd8((*[8]float32)(accA[0:8]))
	sA8 := dlyLd8((*[8]float32)(accA[8:16]))
	sB := dlyLd8((*[8]float32)(accB[0:8]))
	sB8 := dlyLd8((*[8]float32)(accB[8:16]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(hA[i : i+16])
		bb := (*[16]float32)(xA[i : i+16])
		gg := (*[16]float32)(hB[i : i+16])
		dd := (*[16]float32)(xB[i : i+16])
		sA = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(dlyLd8((*[8]float32)(bb[0:8])), sA)
		sA8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(dlyLd8((*[8]float32)(bb[8:16])), sA8)
		sB = dlyLd8((*[8]float32)(gg[0:8])).MulAdd(dlyLd8((*[8]float32)(dd[0:8])), sB)
		sB8 = dlyLd8((*[8]float32)(gg[8:16])).MulAdd(dlyLd8((*[8]float32)(dd[8:16])), sB8)
	}
	sA.StoreArray((*[8]float32)(accA[0:8]))
	sA8.StoreArray((*[8]float32)(accA[8:16]))
	sB.StoreArray((*[8]float32)(accB[0:8]))
	sB8.StoreArray((*[8]float32)(accB[8:16]))
	archsimd.ClearAVXUpperBits()
}

// dlyAxpyFmaDotS2Avx2 is dlyAxpyFmaDotSAvx2 for two filters of the same
// length at once.
func dlyAxpyFmaDotS2Avx2(hA, xaA, xbA []float32, alphaA float32, accA *[16]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[16]float32) {
	if !dlyUseFma {
		dlyAxpyFmaDotS2Avx2Ref(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
		return
	}
	dlyAxpyFmaDotS2Avx2Lanes(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

// dlyAxpyFmaDotS2Avx2Lanes is dlyAxpyFmaDotS2Avx2; it does not inline, so
// the alphas do not cross the VZEROUPPER in its caller as float32 values.
//
//go:noinline
func dlyAxpyFmaDotS2Avx2Lanes(hA, xaA, xbA []float32, alphaA float32, accA *[16]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[16]float32) {
	n := len(hA) &^ 15
	hA, xaA, xbA, hB, xaB, xbB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xaB[:n:n], xbB[:n:n]
	aA := archsimd.BroadcastFloat32x8(alphaA)
	aB := archsimd.BroadcastFloat32x8(alphaB)
	sA := dlyLd8((*[8]float32)(accA[0:8]))
	sA8 := dlyLd8((*[8]float32)(accA[8:16]))
	sB := dlyLd8((*[8]float32)(accB[0:8]))
	sB8 := dlyLd8((*[8]float32)(accB[8:16]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(hA[i : i+16])
		aa := (*[16]float32)(xaA[i : i+16])
		bb := (*[16]float32)(xbA[i : i+16])
		gg := (*[16]float32)(hB[i : i+16])
		cc := (*[16]float32)(xaB[i : i+16])
		dd := (*[16]float32)(xbB[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(aA, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(aA, dlyLd8((*[8]float32)(hh[8:16])))
		gk := dlyLd8((*[8]float32)(cc[0:8])).MulAdd(aB, dlyLd8((*[8]float32)(gg[0:8])))
		gk8 := dlyLd8((*[8]float32)(cc[8:16])).MulAdd(aB, dlyLd8((*[8]float32)(gg[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		gk.StoreArray((*[8]float32)(gg[0:8]))
		gk8.StoreArray((*[8]float32)(gg[8:16]))
		sA = hk.MulAdd(dlyLd8((*[8]float32)(bb[0:8])), sA)
		sA8 = hk8.MulAdd(dlyLd8((*[8]float32)(bb[8:16])), sA8)
		sB = gk.MulAdd(dlyLd8((*[8]float32)(dd[0:8])), sB)
		sB8 = gk8.MulAdd(dlyLd8((*[8]float32)(dd[8:16])), sB8)
	}
	sA.StoreArray((*[8]float32)(accA[0:8]))
	sA8.StoreArray((*[8]float32)(accA[8:16]))
	sB.StoreArray((*[8]float32)(accB[0:8]))
	sB8.StoreArray((*[8]float32)(accB[8:16]))
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyDotDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xB []float32, accB *[4]float32) {
	dlyAxpyDotDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xB, accB)
}

func dlyX2Lanes2(g int, xA, xB []float32, n int, lanesA, lanesB []float32) {
	dlyX2Lanes2Ref(g, xA, xB, n, lanesA, lanesB)
}

// dlyDotS2Sum and dlyAxpyDotS2Sum are the two-filter Sum kernels of the
// lane with group size g: dlyDotSSum (dlyAxpyDotSSum) for two filters at
// once, with the reduced sums before their addition to the zero tail in
// out[0] and out[2]. They report false where the lane has none.

func dlyDotS2Sum(g int, hA, xA, hB, xB []float32, out *[4]float32) bool {
	switch {
	case g == 8 && dlyUseAvx:
		dlyDotS2SumSse2(hA, xA, hB, xB, out)
	case g == 16 && dlyUseFma:
		dlyDotS2SumAvx2(hA, xA, hB, xB, out)
	default:
		return false
	}
	return true
}

func dlyAxpyDotS2Sum(g int, hA, xaA, xbA []float32, alphaA float32,
	hB, xaB, xbB []float32, alphaB float32, out *[4]float32) bool {
	switch {
	case g == 8 && dlyUseAvx:
		dlyAxpyDotS2SumSse2(hA, xaA, xbA, alphaA, hB, xaB, xbB, alphaB, out)
	case g == 16 && dlyUseFma:
		dlyAxpyFmaDotS2SumAvx2(hA, xaA, xbA, alphaA, hB, xaB, xbB, alphaB, out)
	default:
		return false
	}
	return true
}

func dlyDotS2SumSse2(hA, xA, hB, xB []float32, out *[4]float32) {
	n := len(hA) &^ 7
	hA, xA, hB, xB = hA[:n:n], xA[:n:n], hB[:n:n], xB[:n:n]
	var sA, sA4, sB, sB4 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		bb := (*[8]float32)(xA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		dd := (*[8]float32)(xB[i : i+8])
		sA = sA.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		sA4 = sA4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(bb[4:8]))))
		sB = sB.Add(dlyLd4((*[4]float32)(gg[0:4])).Mul(dlyLd4((*[4]float32)(dd[0:4]))))
		sB4 = sB4.Add(dlyLd4((*[4]float32)(gg[4:8])).Mul(dlyLd4((*[4]float32)(dd[4:8]))))
	}
	tA, tB := dlyHsumSSse2(sA, sA4), dlyHsumSSse2(sB, sB4)
	tA.ConcatPermuteScalars(0, 0, 4, 4, tB).StoreArray(out)
}

func dlyAxpyDotS2SumSse2(hA, xaA, xbA []float32, alphaA float32,
	hB, xaB, xbB []float32, alphaB float32, out *[4]float32) {
	n := len(hA) &^ 7
	hA, xaA, xbA, hB, xaB, xbB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xaB[:n:n], xbB[:n:n]
	aA := archsimd.BroadcastFloat32x4(alphaA)
	aB := archsimd.BroadcastFloat32x4(alphaB)
	var sA, sA4, sB, sB4 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(hA[i : i+8])
		aa := (*[8]float32)(xaA[i : i+8])
		bb := (*[8]float32)(xbA[i : i+8])
		gg := (*[8]float32)(hB[i : i+8])
		cc := (*[8]float32)(xaB[i : i+8])
		dd := (*[8]float32)(xbB[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(aA.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		gk := dlyLd4((*[4]float32)(gg[0:4])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[0:4]))))
		gk4 := dlyLd4((*[4]float32)(gg[4:8])).Add(aB.Mul(dlyLd4((*[4]float32)(cc[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		gk.StoreArray((*[4]float32)(gg[0:4]))
		gk4.StoreArray((*[4]float32)(gg[4:8]))
		sA = sA.Add(hk.Mul(dlyLd4((*[4]float32)(bb[0:4]))))
		sA4 = sA4.Add(hk4.Mul(dlyLd4((*[4]float32)(bb[4:8]))))
		sB = sB.Add(gk.Mul(dlyLd4((*[4]float32)(dd[0:4]))))
		sB4 = sB4.Add(gk4.Mul(dlyLd4((*[4]float32)(dd[4:8]))))
	}
	tA, tB := dlyHsumSSse2(sA, sA4), dlyHsumSSse2(sB, sB4)
	tA.ConcatPermuteScalars(0, 0, 4, 4, tB).StoreArray(out)
}

//go:noinline
func dlyDotS2SumAvx2(hA, xA, hB, xB []float32, out *[4]float32) {
	n := len(hA) &^ 15
	hA, xA, hB, xB = hA[:n:n], xA[:n:n], hB[:n:n], xB[:n:n]
	var sA, sA8, sB, sB8 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(hA[i : i+16])
		bb := (*[16]float32)(xA[i : i+16])
		gg := (*[16]float32)(hB[i : i+16])
		dd := (*[16]float32)(xB[i : i+16])
		sA = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(dlyLd8((*[8]float32)(bb[0:8])), sA)
		sA8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(dlyLd8((*[8]float32)(bb[8:16])), sA8)
		sB = dlyLd8((*[8]float32)(gg[0:8])).MulAdd(dlyLd8((*[8]float32)(dd[0:8])), sB)
		sB8 = dlyLd8((*[8]float32)(gg[8:16])).MulAdd(dlyLd8((*[8]float32)(dd[8:16])), sB8)
	}
	tA, tB := dlyHsumSAvx2(sA, sA8), dlyHsumSAvx2(sB, sB8)
	tA.ConcatPermuteScalars(0, 0, 4, 4, tB).StoreArray(out)
	archsimd.ClearAVXUpperBits()
}

//go:noinline
func dlyAxpyFmaDotS2SumAvx2(hA, xaA, xbA []float32, alphaA float32,
	hB, xaB, xbB []float32, alphaB float32, out *[4]float32) {
	n := len(hA) &^ 15
	hA, xaA, xbA, hB, xaB, xbB = hA[:n:n], xaA[:n:n], xbA[:n:n], hB[:n:n], xaB[:n:n], xbB[:n:n]
	aA := archsimd.BroadcastFloat32x8(alphaA)
	aB := archsimd.BroadcastFloat32x8(alphaB)
	var sA, sA8, sB, sB8 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(hA[i : i+16])
		aa := (*[16]float32)(xaA[i : i+16])
		bb := (*[16]float32)(xbA[i : i+16])
		gg := (*[16]float32)(hB[i : i+16])
		cc := (*[16]float32)(xaB[i : i+16])
		dd := (*[16]float32)(xbB[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(aA, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(aA, dlyLd8((*[8]float32)(hh[8:16])))
		gk := dlyLd8((*[8]float32)(cc[0:8])).MulAdd(aB, dlyLd8((*[8]float32)(gg[0:8])))
		gk8 := dlyLd8((*[8]float32)(cc[8:16])).MulAdd(aB, dlyLd8((*[8]float32)(gg[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		gk.StoreArray((*[8]float32)(gg[0:8]))
		gk8.StoreArray((*[8]float32)(gg[8:16]))
		sA = hk.MulAdd(dlyLd8((*[8]float32)(bb[0:8])), sA)
		sA8 = hk8.MulAdd(dlyLd8((*[8]float32)(bb[8:16])), sA8)
		sB = gk.MulAdd(dlyLd8((*[8]float32)(dd[0:8])), sB)
		sB8 = gk8.MulAdd(dlyLd8((*[8]float32)(dd[8:16])), sB8)
	}
	tA, tB := dlyHsumSAvx2(sA, sA8), dlyHsumSAvx2(sB, sB8)
	tA.ConcatPermuteScalars(0, 0, 4, 4, tB).StoreArray(out)
	archsimd.ClearAVXUpperBits()
}
