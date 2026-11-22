//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// x86 kernels of MatchedFilterCore_SSE2 and MatchedFilterCore_AVX2. The
// 128-bit archsimd operations are VEX encoded and round like the SSE2
// instructions. The NEON lane runs its pure-Go references.
//
// The loops index array views of slices cut to one length and capacity, so
// the compiler proves every access in bounds. fma(a, b, c) is symmetric in a
// and b and VFMADD213PS overwrites its receiver, so the receiver of a MulAdd
// is the multiplicand that dies there.
//
// A 256-bit kernel that produces a float32 stores it to memory of the caller
// from a function that does not inline. A float32 live across
// ClearAVXUpperBits is spilled with a 4-byte store and restored with a
// 16-byte load, which the store cannot forward to.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyLd8(p *[8]float32) archsimd.Float32x8 { return archsimd.LoadFloat32x8Array(p) }

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

// dlyPairNeon reports whether the NEON lane has two-filter kernels.
var dlyPairNeon = false

func dlyDotS2Neon(hA, xA []float32, accA *[4]float32, hB, xB []float32, accB *[4]float32) {
	dlyDotS2NeonRef(hA, xA, accA, hB, xB, accB)
}

func dlyAxpyDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[4]float32) {
	dlyAxpyDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

// dlySqStreams returns the number of stretches dlySqMulti accumulates at
// once: three in the SSE2 lane, two in the AVX2 lane.
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
// in one loop, so that their chains run side by side. acc[t] holds the
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
// one loop. acc[t] holds the accumulators of stretch t.
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
// accumulated in one loop: three groups of eight in the SSE2 lane, two
// groups of sixteen in the AVX2 lane.
func dlyX2Lanes(g int, x []float32, n int, lanes []float32) {
	if (g == 8 && dlyUseAvx) || (g == 16 && dlyUseFma) {
		dlyX2LanesMulti(g, x, n, lanes)
		return
	}
	dlyX2LanesRef(g, x, n, lanes)
}

// dlyHsumSSse2 reduces the two SSE2 accumulators of one sum as dlyReduceS
// does: lane 0 of the result is ((v0 + v1) + v2) + v3 for v = s + s4.
func dlyHsumSSse2(s, s4 archsimd.Float32x4) archsimd.Float32x4 {
	v := s.Add(s4)
	t := v.Add(v.ConcatPermuteScalars(1, 1, 5, 5, v))
	t = t.Add(v.ConcatPermuteScalars(2, 2, 6, 6, v))
	return t.Add(v.ConcatPermuteScalars(3, 3, 7, 7, v))
}

// dlyHsumSAvx2 reduces the two AVX2 accumulators of one sum as hsum_ab does:
// lane 0 of the result is ((b0+b1) + (b2+b3)) + ((b4+b5) + (b6+b7)) for b =
// s + s8.
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
// sample j go to lane 0 of a 128-bit store at tmp[j], which the stores of
// later samples overwrite in the other lanes. The scalar addition to the
// zero x2_sum of the C++ runs after the VZEROUPPER.
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

// dlyAccErrSums returns in lanes 0 and 1 the sums of the four products of
// the sub-groups p and q of a group, each added in sequence as in the C++:
// ((v0 + v1) + v2) + v3.
func dlyAccErrSums(p, q archsimd.Float32x4) archsimd.Float32x4 {
	a, b := p.ToBits(), q.ToBits()
	lo := a.InterleaveLo(b).BitsToFloat32() // p0 q0 p1 q1
	hi := a.InterleaveHi(b).BitsToFloat32() // p2 q2 p3 q3
	t := lo.Add(lo.ConcatPermuteScalars(2, 3, 6, 7, lo))
	t = t.Add(hi)
	return t.Add(hi.ConcatPermuteScalars(2, 3, 6, 7, hi))
}

// dlyAccErrSse2 keeps the running sum s in lane 0 of a register. The two
// sub-group sums of a group enter it in turn, and the errors of two groups
// are gathered into one vector for the update of accErr. Only the errors of
// a group wait for the chain of s.
func dlyAccErrSse2(h, x []float32, y float32, accErr []float32) float32 {
	if !dlyUseAvx {
		return dlyAccErrSse2Ref(h, x, y, accErr)
	}
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	yv := kernSplat4(y)
	var sv archsimd.Float32x4
	for len(h) >= 16 && len(x) >= 16 && len(accErr) >= 4 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		ae := (*[4]float32)(accErr)
		t := dlyAccErrSums(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4]))),
			dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		u := dlyAccErrSums(dlyLd4((*[4]float32)(hh[8:12])).Mul(dlyLd4((*[4]float32)(xx[8:12]))),
			dlyLd4((*[4]float32)(hh[12:16])).Mul(dlyLd4((*[4]float32)(xx[12:16]))))
		s0 := sv.Add(t)
		s1 := s0.Add(t.ConcatPermuteScalars(1, 1, 5, 5, t))
		s2 := s1.Add(u)
		s3 := s2.Add(u.ConcatPermuteScalars(1, 1, 5, 5, u))
		sv = s3
		ev := s0.ConcatPermuteScalars(0, 0, 4, 4, s1).ConcatPermuteScalars(0, 2, 4, 6,
			s2.ConcatPermuteScalars(0, 0, 4, 4, s3)).Sub(yv)
		dlyLd4(ae).Add(ev.Mul(ev)).StoreArray(ae)
		h = h[16:]
		x = x[16:]
		accErr = accErr[4:]
	}
	if len(h) >= 8 && len(x) >= 8 && len(accErr) >= 2 {
		hh := (*[8]float32)(h)
		xx := (*[8]float32)(x)
		t := dlyAccErrSums(dlyLd4((*[4]float32)(hh[0:4])).Mul(dlyLd4((*[4]float32)(xx[0:4]))),
			dlyLd4((*[4]float32)(hh[4:8])).Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		s0 := sv.Add(t)
		sv = s0.Add(t.ConcatPermuteScalars(1, 1, 5, 5, t))
		e0 := s0.GetElem(0) - y
		e1 := sv.GetElem(0) - y
		accErr[0] += float32(e0 * e0)
		accErr[1] += float32(e1 * e1)
	}
	return sv.GetElem(0)
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
	a := kernSplat4(alpha)
	yv := kernSplat4(y)
	var sv archsimd.Float32x4
	for len(h) >= 16 && len(xa) >= 16 && len(xb) >= 16 && len(accErr) >= 4 {
		hh := (*[16]float32)(h)
		aa := (*[16]float32)(xa)
		xx := (*[16]float32)(xb)
		ae := (*[4]float32)(accErr)
		h0, h1 := (*[4]float32)(hh[0:4]), (*[4]float32)(hh[4:8])
		h2, h3 := (*[4]float32)(hh[8:12]), (*[4]float32)(hh[12:16])
		u0 := dlyLd4(h0).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		u1 := dlyLd4(h1).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		u2 := dlyLd4(h2).Add(a.Mul(dlyLd4((*[4]float32)(aa[8:12]))))
		u3 := dlyLd4(h3).Add(a.Mul(dlyLd4((*[4]float32)(aa[12:16]))))
		u0.StoreArray(h0)
		u1.StoreArray(h1)
		u2.StoreArray(h2)
		u3.StoreArray(h3)
		t := dlyAccErrSums(u0.Mul(dlyLd4((*[4]float32)(xx[0:4]))), u1.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		u := dlyAccErrSums(u2.Mul(dlyLd4((*[4]float32)(xx[8:12]))), u3.Mul(dlyLd4((*[4]float32)(xx[12:16]))))
		s0 := sv.Add(t)
		s1 := s0.Add(t.ConcatPermuteScalars(1, 1, 5, 5, t))
		s2 := s1.Add(u)
		s3 := s2.Add(u.ConcatPermuteScalars(1, 1, 5, 5, u))
		sv = s3
		ev := s0.ConcatPermuteScalars(0, 0, 4, 4, s1).ConcatPermuteScalars(0, 2, 4, 6,
			s2.ConcatPermuteScalars(0, 0, 4, 4, s3)).Sub(yv)
		dlyLd4(ae).Add(ev.Mul(ev)).StoreArray(ae)
		h = h[16:]
		xa = xa[16:]
		xb = xb[16:]
		accErr = accErr[4:]
	}
	if len(h) >= 8 && len(xa) >= 8 && len(xb) >= 8 && len(accErr) >= 2 {
		hh := (*[8]float32)(h)
		aa := (*[8]float32)(xa)
		xx := (*[8]float32)(xb)
		h0, h1 := (*[4]float32)(hh[0:4]), (*[4]float32)(hh[4:8])
		u0 := dlyLd4(h0).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		u1 := dlyLd4(h1).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		u0.StoreArray(h0)
		u1.StoreArray(h1)
		t := dlyAccErrSums(u0.Mul(dlyLd4((*[4]float32)(xx[0:4]))), u1.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		s0 := sv.Add(t)
		sv = s0.Add(t.ConcatPermuteScalars(1, 1, 5, 5, t))
		e0 := s0.GetElem(0) - y
		e1 := sv.GetElem(0) - y
		accErr[0] += float32(e0 * e0)
		accErr[1] += float32(e1 * e1)
	}
	return sv.GetElem(0)
}

func dlyAxpy(h, x []float32, alpha float32) {
	if !dlyUseAvx {
		dlyAxpyRef(h, x, alpha)
		return
	}
	n := len(h)
	x = x[:n:n]
	a := kernSplat4(alpha)
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

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr)
	}
	var r [4]float32
	dlyAccErrAvx2Lanes(h, x, y, accErr, &r)
	return r[0]
}

// dlyAccErrAvx2Lanes stores the filter output of dlyAccErrAvx2 in r[0]. It
// keeps the running sum s in lane 0 of a 128-bit register, which the four
// sub-group sums of a group (hadd lanes [0], [4], [1], [5]) enter in turn.
// The four errors are gathered into one vector for the fused update of
// accErr.
//
//go:noinline
func dlyAccErrAvx2Lanes(h, x []float32, y float32, accErr []float32, r *[4]float32) {
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	yv := archsimd.BroadcastFloat32x4(y)
	// The running sum starts from r, which the caller zeroes: a zero made
	// here would be a legacy-SSE instruction in the 256-bit section.
	sv := dlyLd4(r)
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
	// The running sum starts from r, which the caller zeroes (see
	// dlyAccErrAvx2Lanes).
	sv := dlyLd4(r)
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

// dlyAxpyFma256 runs dlyAxpyFma over the whole groups of 8 and returns their
// length. It does not inline, so that alpha does not cross the VZEROUPPER as
// a float32.
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
// kernels load them as 256-bit vectors, and narrower stores cannot forward
// to such a load.
//
// It does not inline, so that no float32 of the caller crosses its
// VZEROUPPER.
//
//go:noinline
func dlyZeroAcc(acc *[16]float32) {
	switch {
	case kernHasAVX2:
		var z archsimd.Float32x8
		z.StoreArray((*[8]float32)(acc[0:8]))
		z.StoreArray((*[8]float32)(acc[8:16]))
		archsimd.ClearAVXUpperBits()
	case dlyUseAvx:
		// AVX without AVX2, for the kernels of the SSE2 lane: 128-bit
		// stores.
		var z archsimd.Float32x4
		z.StoreArray((*[4]float32)(acc[0:4]))
		z.StoreArray((*[4]float32)(acc[4:8]))
		z.StoreArray((*[4]float32)(acc[8:12]))
		z.StoreArray((*[4]float32)(acc[12:16]))
	default:
		*acc = [16]float32{}
	}
}

// dlyPeakLanes256 is dlyPeakLanes128 with two 256-bit vectors per 16
// samples: lanes 0-7 and 8-15, the same index residues. It reports false
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

func dlyAxpyDotDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xB []float32, accB *[4]float32) {
	dlyAxpyDotDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xB, accB)
}

func dlyX2Lanes2(g int, xA, xB []float32, n int, lanesA, lanesB []float32) {
	dlyX2Lanes2Ref(g, xA, xB, n, lanesA, lanesB)
}

// The kernels of dlyCoreLanes run their references in the SSE2 and AVX2
// lanes, whose cores run dlyCoreSimple.

func dlyDotSSse2(h, x []float32, acc *[8]float32) { dlyDotSSse2Ref(h, x, acc) }

func dlyAxpyDotSSse2(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotSSse2Ref(h, xa, xb, alpha, acc)
}

func dlyDotSAvx2(h, x []float32, acc *[16]float32) { dlyDotSAvx2Ref(h, x, acc) }

func dlyAxpyFmaDotSAvx2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyFmaDotSAvx2Ref(h, xa, xb, alpha, acc)
}

func dlyDotSSumSse2(h, x []float32) float32 { return dlyDotSSumSse2Ref(h, x) }

func dlyAxpyDotSSumSse2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyDotSSumSse2Ref(h, xa, xb, alpha)
}

func dlyDotSSumAvx2(h, x []float32) float32 { return dlyDotSSumAvx2Ref(h, x) }

func dlyAxpyFmaDotSSumAvx2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyFmaDotSSumAvx2Ref(h, xa, xb, alpha)
}

func dlyX2Wrapped(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	dlyX2WrappedRef(g, x, hSize, lo, r, out)
}

// Cores of the SSE2 and AVX2 lanes.
//
// On x86 the MatchedFilterCore loop is bound by its loads and stores, not by
// its accumulator chains: a separate x*x pass costs, and two filters in step
// gain nothing. With archsimd kernels the two lanes run dlyCoreSimple, whose
// kernels accumulate h*x and x*x side by side as the C++ loop does. Their
// references, and the NEON lane, run dlyCoreLanes.
//
// The accumulators of the kernels below:
//
//	SSE2  *[16]float32  [0:4] s_128, [4:8] s_128_4,
//	                    [8:12] x2_sum_128, [12:16] x2_sum_128_4
//	AVX2  *[32]float32  [0:8] s_256, [8:16] s_256_8,
//	                    [16:24] x2_sum_256, [24:32] x2_sum_256_8

// dlyLanesPureGo reports whether the lanes run their pure-Go references,
// which use the tables of squares for the x*x sums.
const dlyLanesPureGo = false

// dlyCoreLaneKernel runs the MatchedFilterCore loop of the lane with group
// size g on one filter if the lane has a core of its own, and reports
// whether it did.
func dlyCoreLaneKernel(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) bool {
	if (g == 8 && dlyUseAvx) || (g == 16 && dlyUseFma) {
		dlyCoreSimple(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
		return true
	}
	return false
}

// dlyCoreSimple runs the MatchedFilterCore loop of the SSE2 (g = 8) or the
// AVX2 (g = 16) lane on one filter. The update of a sample is deferred and
// fused into the dot product of the next one: one kernel call covers all of
// h when neither sample wraps and h is a whole number of groups, and
// dlyWrapSample runs the other samples. An element sees the same update
// either way.
//
// dlyCoreLaneKernel admits g = 16 only with dlyUseFma. The AVX2 kernels are
// called behind a test of their own (TestSimdInstructionsGated).
func dlyCoreSimple(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	hSize := len(h)
	xSize := len(x)
	var pend dlyPendingAxpy
	pendWhole := false // the pending update covers h with one x stretch at pendStart
	pendStart, pendChunk1 := 0, 0
	for i := range y {
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		v1 := chunk1 &^ (g - 1)
		if v1 == hSize {
			// One stretch, no tails: the Sum kernels reduce in registers.
			xb := x[xStartIndex : xStartIndex+hSize]
			if pend.active && pendWhole {
				xa := x[pendStart : pendStart+hSize]
				if g == 8 {
					s, x2Sum = dlyAxpyDotSse2Sum(h, xa, xb, pend.alpha)
				} else if dlyUseFma {
					s, x2Sum = dlyAxpyFmaDotAvx2Sum(h, xa, xb, pend.alpha)
				}
			} else {
				if pend.active {
					pend.set(pendStart, pendChunk1, hSize, pend.alpha, g == 16)
					pend.apply(h, x, 0, hSize)
				}
				if g == 8 {
					s, x2Sum = dlyDotSse2Sum(h, xb)
				} else if dlyUseFma {
					s, x2Sum = dlyDotAvx2Sum(h, xb)
				}
			}
		} else {
			s, x2Sum = dlyWrapSample(g, h, x, xStartIndex, chunk1, v1, chunk2&^(g-1), &pend, pendStart, pendChunk1)
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		pend.active = false
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			pend.active = true
			pend.alpha = alpha
			pendWhole = chunk1 == hSize
			pendStart, pendChunk1 = xStartIndex, chunk1
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
	if pend.active {
		pend.set(pendStart, pendChunk1, hSize, pend.alpha, g == 16)
		pend.apply(h, x, 0, hSize)
	}
}

// dlyHsumSse2 reduces the SSE2 lane accumulators of the x2 and the s sum
// side by side: lanes 0 and 1 of the result are ((v0 + v1) + v2) + v3 for x2
// and for s.
func dlyHsumSse2(s, s4, x2, x24 archsimd.Float32x4) archsimd.Float32x4 {
	a := x2.Add(x24).ToBits()
	b := s.Add(s4).ToBits()
	lo := a.InterleaveLo(b).BitsToFloat32() // a0 b0 a1 b1
	hi := a.InterleaveHi(b).BitsToFloat32() // a2 b2 a3 b3
	t := lo.Add(lo.ConcatPermuteScalars(2, 3, 6, 7, lo))
	t = t.Add(hi)
	return t.Add(hi.ConcatPermuteScalars(2, 3, 6, 7, hi))
}

// dlyHsumAvx2 reduces the AVX2 lane accumulators as hsum_ab does: pairwise
// sums of a = x2_sum_256 + x2_sum_256_8 and b = s_256 + s_256_8, the x2 sum
// in lane 0 and the s sum in lane 1.
func dlyHsumAvx2(s, s8, x2, x28 archsimd.Float32x8) archsimd.Float32x4 {
	p := x2.Add(x28).ConcatAddPairsGrouped(s.Add(s8)) // a01 a23 b01 b23 | a45 a67 b45 b67
	lo, hi := p.GetLo(), p.GetHi()
	// t lanes: (a01 + a23), (b01 + b23), (a45 + a67), (b45 + b67).
	t := lo.ConcatPermuteScalars(0, 2, 4, 6, hi).Add(lo.ConcatPermuteScalars(1, 3, 5, 7, hi))
	return t.Add(t.ConcatPermuteScalars(2, 3, 6, 7, t))
}

// The Sum kernels run over blocks of 64 elements and advance the slices
// block by block: the block pointer is the slice pointer, and every vector
// of a block is addressed as pointer plus constant. An array pointer taken
// at h[i : i+16] with a loop counter costs two address computations per
// access.

func dlyDotSse2Sum(h, x []float32) (s, x2Sum float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	var sv, s4, x2, x24 archsimd.Float32x4
	for len(h) >= 64 && len(x) >= 64 {
		hh := (*[64]float32)(h)
		xx := (*[64]float32)(x)
		step := func(o int) {
			xk := dlyLd4((*[4]float32)(xx[o : o+4]))
			xk4 := dlyLd4((*[4]float32)(xx[o+4 : o+8]))
			x2 = x2.Add(xk.Mul(xk))
			x24 = x24.Add(xk4.Mul(xk4))
			sv = sv.Add(dlyLd4((*[4]float32)(hh[o : o+4])).Mul(xk))
			s4 = s4.Add(dlyLd4((*[4]float32)(hh[o+4 : o+8])).Mul(xk4))
		}
		step(0)
		step(8)
		step(16)
		step(24)
		step(32)
		step(40)
		step(48)
		step(56)
		h = h[64:]
		x = x[64:]
	}
	for len(h) >= 8 && len(x) >= 8 {
		hh := (*[8]float32)(h)
		xx := (*[8]float32)(x)
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(xk))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4))
		h = h[8:]
		x = x[8:]
	}
	t := dlyHsumSse2(sv, s4, x2, x24)
	r0, r1 := t.GetElem(0), t.GetElem(1)
	return 0 + r1, 0 + r0
}

func dlyAxpyDotSse2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := kernSplat4(alpha)
	var sv, s4, x2, x24 archsimd.Float32x4
	for len(h) >= 64 && len(xa) >= 64 && len(xb) >= 64 {
		hh := (*[64]float32)(h)
		aa := (*[64]float32)(xa)
		bb := (*[64]float32)(xb)
		step := func(o int) {
			h0 := (*[4]float32)(hh[o : o+4])
			h4 := (*[4]float32)(hh[o+4 : o+8])
			hk := dlyLd4(h0).Add(a.Mul(dlyLd4((*[4]float32)(aa[o : o+4]))))
			hk4 := dlyLd4(h4).Add(a.Mul(dlyLd4((*[4]float32)(aa[o+4 : o+8]))))
			hk.StoreArray(h0)
			hk4.StoreArray(h4)
			xk := dlyLd4((*[4]float32)(bb[o : o+4]))
			xk4 := dlyLd4((*[4]float32)(bb[o+4 : o+8]))
			x2 = x2.Add(xk.Mul(xk))
			x24 = x24.Add(xk4.Mul(xk4))
			sv = sv.Add(hk.Mul(xk))
			s4 = s4.Add(hk4.Mul(xk4))
		}
		step(0)
		step(8)
		step(16)
		step(24)
		step(32)
		step(40)
		step(48)
		step(56)
		h = h[64:]
		xa = xa[64:]
		xb = xb[64:]
	}
	for len(h) >= 8 && len(xa) >= 8 && len(xb) >= 8 {
		hh := (*[8]float32)(h)
		aa := (*[8]float32)(xa)
		bb := (*[8]float32)(xb)
		h0 := (*[4]float32)(hh[0:4])
		h4 := (*[4]float32)(hh[4:8])
		hk := dlyLd4(h0).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4(h4).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray(h0)
		hk4.StoreArray(h4)
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
		h = h[8:]
		xa = xa[8:]
		xb = xb[8:]
	}
	t := dlyHsumSse2(sv, s4, x2, x24)
	r0, r1 := t.GetElem(0), t.GetElem(1)
	return 0 + r1, 0 + r0
}

func dlyDotAvx2Sum(h, x []float32) (s, x2Sum float32) {
	var r [4]float32
	dlyDotAvx2SumLanes(h, x, &r)
	return 0 + r[1], 0 + r[0]
}

// dlyDotAvx2SumLanes stores the reduced lane sums of dlyDotAvx2Sum in r. A
// float32 result would cross the VZEROUPPER.
//
//go:noinline
func dlyDotAvx2SumLanes(h, x []float32, r *[4]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	x = x[:n:n]
	var sv, s8, x2, x28 archsimd.Float32x8
	for len(h) >= 64 && len(x) >= 64 {
		hh := (*[64]float32)(h)
		xx := (*[64]float32)(x)
		step := func(o int) {
			xk := dlyLd8((*[8]float32)(xx[o : o+8]))
			xk8 := dlyLd8((*[8]float32)(xx[o+8 : o+16]))
			sv = dlyLd8((*[8]float32)(hh[o:o+8])).MulAdd(xk, sv)
			s8 = dlyLd8((*[8]float32)(hh[o+8:o+16])).MulAdd(xk8, s8)
			x2 = xk.MulAdd(xk, x2)
			x28 = xk8.MulAdd(xk8, x28)
		}
		step(0)
		step(16)
		step(32)
		step(48)
		h = h[64:]
		x = x[64:]
	}
	for len(h) >= 16 && len(x) >= 16 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		sv = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(xk, sv)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
		h = h[16:]
		x = x[16:]
	}
	dlyHsumAvx2(sv, s8, x2, x28).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

func dlyAxpyFmaDotAvx2Sum(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	var r [4]float32
	dlyAxpyFmaDotAvx2SumLanes(h, xa, xb, alpha, &r)
	return 0 + r[1], 0 + r[0]
}

// dlyAxpyFmaDotAvx2SumLanes stores the reduced lane sums of
// dlyAxpyFmaDotAvx2Sum in r.
//
//go:noinline
func dlyAxpyFmaDotAvx2SumLanes(h, xa, xb []float32, alpha float32, r *[4]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	var sv, s8, x2, x28 archsimd.Float32x8
	for len(h) >= 64 && len(xa) >= 64 && len(xb) >= 64 {
		hh := (*[64]float32)(h)
		aa := (*[64]float32)(xa)
		bb := (*[64]float32)(xb)
		step := func(o int) {
			h0 := (*[8]float32)(hh[o : o+8])
			h8 := (*[8]float32)(hh[o+8 : o+16])
			hk := dlyLd8((*[8]float32)(aa[o:o+8])).MulAdd(a, dlyLd8(h0))
			hk8 := dlyLd8((*[8]float32)(aa[o+8:o+16])).MulAdd(a, dlyLd8(h8))
			hk.StoreArray(h0)
			hk8.StoreArray(h8)
			xk := dlyLd8((*[8]float32)(bb[o : o+8]))
			xk8 := dlyLd8((*[8]float32)(bb[o+8 : o+16]))
			sv = hk.MulAdd(xk, sv)
			s8 = hk8.MulAdd(xk8, s8)
			x2 = xk.MulAdd(xk, x2)
			x28 = xk8.MulAdd(xk8, x28)
		}
		step(0)
		step(16)
		step(32)
		step(48)
		h = h[64:]
		xa = xa[64:]
		xb = xb[64:]
	}
	for len(h) >= 16 && len(xa) >= 16 && len(xb) >= 16 {
		hh := (*[16]float32)(h)
		aa := (*[16]float32)(xa)
		bb := (*[16]float32)(xb)
		h0 := (*[8]float32)(hh[0:8])
		h8 := (*[8]float32)(hh[8:16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8(h0))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8(h8))
		hk.StoreArray(h0)
		hk8.StoreArray(h8)
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		sv = hk.MulAdd(xk, sv)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
		h = h[16:]
		xa = xa[16:]
		xb = xb[16:]
	}
	dlyHsumAvx2(sv, s8, x2, x28).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

// Samples whose window wraps around the circular buffer or whose filter is
// not a whole number of groups.
//
// The C++ loop runs such a sample as two chunks, up to the end of the buffer
// and from its start: the whole groups of a chunk in the lane accumulators,
// the rest in two scalars. The pending update of the sample before it has
// its own two chunks, one element apart. dlyWrapSample applies that update
// to the scalar tails and to the groups the vector update of the lane does
// not cover, takes the scalar sums of the tails, and hands the groups to a
// kernel that keeps the accumulators of both chunks in registers and fuses
// the update into the groups it covers.

// fused returns the length n of the whole groups from a, within [a, b), that
// one stretch of the pending update covers with the vector update of the
// lane with group size g, and the offset off of that stretch: the update of
// h[k] reads x[k+off].
func (p *dlyPendingAxpy) fused(g, a, b int) (n, off int) {
	for i := 0; i < p.n; i++ {
		r := &p.runs[i]
		if r.fma == (g == 16) && r.start <= a && a < r.end {
			return (min(b, r.end) - a) &^ (g - 1), r.off
		}
	}
	return 0, 0
}

// applySmall is apply for stretches of a few elements.
func (p *dlyPendingAxpy) applySmall(h, x []float32, a, b int) {
	for i := 0; i < p.n; i++ {
		r := &p.runs[i]
		lo, hi := max(a, r.start), min(b, r.end)
		if lo >= hi {
			continue
		}
		if r.fma {
			dlyAxpyFma128(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		} else {
			dlyAxpy(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		}
	}
}

// dlyAxpyFma128 is dlyAxpyFma with 128-bit vectors, for a few elements. It
// leaves the upper halves of the registers clean. The last vector of a
// length that is not a multiple of four overlaps the one before it; both are
// computed from the elements as they were, so a shared element is updated
// once.
func dlyAxpyFma128(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	n := len(h)
	x = x[:n:n]
	if n < 4 {
		for i := range h {
			h[i] = fmaf(x[i], alpha, h[i])
		}
		return
	}
	a := kernSplat4(alpha)
	i := 0
	for ; i < n-7; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		dlyLd4((*[4]float32)(x[i:i+4])).MulAdd(a, dlyLd4(hp)).StoreArray(hp)
	}
	// Four to seven elements are left.
	h, x = h[i:], x[i:]
	m := len(h)
	if m < 4 || len(x) < m {
		return
	}
	h0 := (*[4]float32)(h[0:4])
	h1 := (*[4]float32)(h[m-4 : m])
	u0 := dlyLd4((*[4]float32)(x[0:4])).MulAdd(a, dlyLd4(h0))
	u1 := dlyLd4((*[4]float32)(x[m-4:m])).MulAdd(a, dlyLd4(h1))
	u0.StoreArray(h0)
	u1.StoreArray(h1)
}

// dlyWrapSample computes the filter output s and the x*x sum of the sample
// with start index xs. Its first chunk has c elements, v1 of them in whole
// groups; its second chunk has v2 elements in whole groups. It applies the
// pending update, of the sample with start index pendStart and a first chunk
// of pendChunk1 elements, to all of h.
func dlyWrapSample(g int, h, x []float32, xs, c, v1, v2 int, pend *dlyPendingAxpy, pendStart, pendChunk1 int) (s, x2Sum float32) {
	hSize := len(h)
	nA, offA, nB, offB := 0, 0, 0, 0
	var alpha float32
	if pend.active {
		alpha = pend.alpha
		if pendChunk1 == c-1 && pendStart == xs+1 && c >= 2 && c <= hSize && xs+c <= len(x) {
			// The sample before this one, one element further on in x.
			offA, offB = xs+1, 1-c
			if g == 8 {
				nA, nB = dlyWrapPendSse2(h, x, xs, c, v1, v2, alpha)
			} else {
				nA, nB = dlyWrapPendAvx2(h, x, xs, c, v1, v2, alpha)
			}
		} else {
			pend.set(pendStart, pendChunk1, hSize, alpha, g == 16)
			nA, offA = pend.fused(g, 0, v1)
			nB, offB = pend.fused(g, c, c+v2)
			pend.applySmall(h, x, nA, c)
			pend.applySmall(h, x, c+nB, hSize)
		}
	}
	for k := v1; k < c; k++ {
		xk := x[xs+k]
		x2Sum += float32(xk * xk)
		s += float32(h[k] * xk)
	}
	for k := c + v2; k < hSize; k++ {
		xk := x[k-c]
		x2Sum += float32(xk * xk)
		s += float32(h[k] * xk)
	}
	if g == 8 {
		r0, r1 := dlyWrapSse2(h, x, xs, c, v1, v2, nA, offA, nB, offB, alpha)
		return s + r1, x2Sum + r0
	}
	if dlyUseFma {
		var r [4]float32
		dlyWrapAvx2(h, x, xs, c, v1, v2, nA, offA, nB, offB, alpha, &r)
		return s + r[1], x2Sum + r[0]
	}
	return s, x2Sum
}

// dlyWrapPendSse2 applies the update of the sample before the one with start
// index xs and a first chunk of c elements to the elements of h that
// dlyWrapSse2 does not update. It returns the lengths nA and nB of the
// groups dlyWrapSse2 updates. The update of h[k] reads x[xs+1+k] below c-1
// and x[k-c+1] from there on.
func dlyWrapPendSse2(h, x []float32, xs, c, v1, v2 int, alpha float32) (nA, nB int) {
	nA = min(v1, (c-1)&^7)
	xa := x[xs+1 : xs+c]
	hh := h[:c-1]
	for k := nA; k < len(hh) && k < len(xa); k++ {
		hh[k] += float32(alpha * xa[k])
	}
	h[c-1] += float32(alpha * x[0])
	hh = h[c:]
	xa = x[1:]
	for k := v2; k < len(hh) && k < len(xa); k++ {
		hh[k] += float32(alpha * xa[k])
	}
	return nA, v2
}

// dlyWrapPendAvx2 is dlyWrapPendSse2 for the AVX2 lane, whose update is
// fused over the whole groups of eight of each of its chunks and unfused
// over the rest.
func dlyWrapPendAvx2(h, x []float32, xs, c, v1, v2 int, alpha float32) (nA, nB int) {
	p1 := (c - 1) &^ 7
	p2 := (len(h) - c + 1) &^ 7
	nA = min(v1, p1&^15)
	xa := x[xs+1 : xs+c]
	hh := h[:c-1]
	if nA < p1 {
		dlyAxpyFma128(hh[nA:p1], xa[nA:p1], alpha)
	}
	for k := p1; k < len(hh) && k < len(xa); k++ {
		hh[k] += float32(alpha * xa[k])
	}
	// The second chunk of the update starts at c-1. Its fused part has p2
	// elements.
	hh = h[c-1:]
	xa = x[:len(hh)]
	if p2 > 0 {
		nB = min(v2, (p2-1)&^15)
		hh[0] = fmaf(xa[0], alpha, hh[0])
		if 1+nB < p2 {
			dlyAxpyFma128(hh[1+nB:p2], xa[1+nB:p2], alpha)
		}
	}
	for k := p2; k < len(hh); k++ {
		hh[k] += float32(alpha * xa[k])
	}
	return nA, nB
}

// dlyWrapSse2 accumulates h*x and x*x over the whole groups of the two
// chunks of a sample: h[:v1] against x[xs:] and h[c:c+v2] against x[0:]. It
// first updates the first nA elements of the first stretch, h[k] +=
// alpha*x[k+offA], and the first nB of the second with offB. It returns the
// reduced x*x sum and the reduced h*x sum.
func dlyWrapSse2(h, x []float32, xs, c, v1, v2, nA, offA, nB, offB int, alpha float32) (r0, r1 float32) {
	hA, aA, bA := h[:nA:nA], x[offA:][:nA:nA], x[xs:][:nA:nA]
	hP, bP := h[nA:v1], x[xs+nA:xs+v1]
	hB, aB, bB := h[c:][:nB:nB], x[c+offB:][:nB:nB], x[:nB:nB]
	hQ, bQ := h[c+nB:c+v2], x[nB:v2]
	a := kernSplat4(alpha)
	var sv, s4, x2, x24 archsimd.Float32x4
	fused := func(hh, aa, bb *[8]float32) {
		h0 := (*[4]float32)(hh[0:4])
		h4 := (*[4]float32)(hh[4:8])
		hk := dlyLd4(h0).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4(h4).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray(h0)
		hk4.StoreArray(h4)
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
	}
	plain := func(hh, bb *[8]float32) {
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(xk))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4))
	}
	for len(hA) >= 32 && len(aA) >= 32 && len(bA) >= 32 {
		hh := (*[32]float32)(hA)
		aa := (*[32]float32)(aA)
		bb := (*[32]float32)(bA)
		fused((*[8]float32)(hh[0:8]), (*[8]float32)(aa[0:8]), (*[8]float32)(bb[0:8]))
		fused((*[8]float32)(hh[8:16]), (*[8]float32)(aa[8:16]), (*[8]float32)(bb[8:16]))
		fused((*[8]float32)(hh[16:24]), (*[8]float32)(aa[16:24]), (*[8]float32)(bb[16:24]))
		fused((*[8]float32)(hh[24:32]), (*[8]float32)(aa[24:32]), (*[8]float32)(bb[24:32]))
		hA, aA, bA = hA[32:], aA[32:], bA[32:]
	}
	for len(hA) >= 8 && len(aA) >= 8 && len(bA) >= 8 {
		fused((*[8]float32)(hA), (*[8]float32)(aA), (*[8]float32)(bA))
		hA, aA, bA = hA[8:], aA[8:], bA[8:]
	}
	for len(hP) >= 8 && len(bP) >= 8 {
		plain((*[8]float32)(hP), (*[8]float32)(bP))
		hP, bP = hP[8:], bP[8:]
	}
	for len(hB) >= 32 && len(aB) >= 32 && len(bB) >= 32 {
		hh := (*[32]float32)(hB)
		aa := (*[32]float32)(aB)
		bb := (*[32]float32)(bB)
		fused((*[8]float32)(hh[0:8]), (*[8]float32)(aa[0:8]), (*[8]float32)(bb[0:8]))
		fused((*[8]float32)(hh[8:16]), (*[8]float32)(aa[8:16]), (*[8]float32)(bb[8:16]))
		fused((*[8]float32)(hh[16:24]), (*[8]float32)(aa[16:24]), (*[8]float32)(bb[16:24]))
		fused((*[8]float32)(hh[24:32]), (*[8]float32)(aa[24:32]), (*[8]float32)(bb[24:32]))
		hB, aB, bB = hB[32:], aB[32:], bB[32:]
	}
	for len(hB) >= 8 && len(aB) >= 8 && len(bB) >= 8 {
		fused((*[8]float32)(hB), (*[8]float32)(aB), (*[8]float32)(bB))
		hB, aB, bB = hB[8:], aB[8:], bB[8:]
	}
	for len(hQ) >= 8 && len(bQ) >= 8 {
		plain((*[8]float32)(hQ), (*[8]float32)(bQ))
		hQ, bQ = hQ[8:], bQ[8:]
	}
	t := dlyHsumSse2(sv, s4, x2, x24)
	return t.GetElem(0), t.GetElem(1)
}

// dlyWrapAvx2 is dlyWrapSse2 for the AVX2 lane, with the update h[k] =
// fma(x[k+off], alpha, h[k]). It stores the reduced x*x sum in r[0] and the
// reduced h*x sum in r[1].
//
//go:noinline
func dlyWrapAvx2(h, x []float32, xs, c, v1, v2, nA, offA, nB, offB int, alpha float32, r *[4]float32) {
	hA, aA, bA := h[:nA:nA], x[offA:][:nA:nA], x[xs:][:nA:nA]
	hP, bP := h[nA:v1], x[xs+nA:xs+v1]
	hB, aB, bB := h[c:][:nB:nB], x[c+offB:][:nB:nB], x[:nB:nB]
	hQ, bQ := h[c+nB:c+v2], x[nB:v2]
	a := archsimd.BroadcastFloat32x8(alpha)
	var sv, s8, x2, x28 archsimd.Float32x8
	fused := func(hh, aa, bb *[16]float32) {
		h0 := (*[8]float32)(hh[0:8])
		h8 := (*[8]float32)(hh[8:16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8(h0))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8(h8))
		hk.StoreArray(h0)
		hk8.StoreArray(h8)
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		sv = hk.MulAdd(xk, sv)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	plain := func(hh, bb *[16]float32) {
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		sv = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(xk, sv)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	for len(hA) >= 64 && len(aA) >= 64 && len(bA) >= 64 {
		hh := (*[64]float32)(hA)
		aa := (*[64]float32)(aA)
		bb := (*[64]float32)(bA)
		fused((*[16]float32)(hh[0:16]), (*[16]float32)(aa[0:16]), (*[16]float32)(bb[0:16]))
		fused((*[16]float32)(hh[16:32]), (*[16]float32)(aa[16:32]), (*[16]float32)(bb[16:32]))
		fused((*[16]float32)(hh[32:48]), (*[16]float32)(aa[32:48]), (*[16]float32)(bb[32:48]))
		fused((*[16]float32)(hh[48:64]), (*[16]float32)(aa[48:64]), (*[16]float32)(bb[48:64]))
		hA, aA, bA = hA[64:], aA[64:], bA[64:]
	}
	for len(hA) >= 16 && len(aA) >= 16 && len(bA) >= 16 {
		fused((*[16]float32)(hA), (*[16]float32)(aA), (*[16]float32)(bA))
		hA, aA, bA = hA[16:], aA[16:], bA[16:]
	}
	for len(hP) >= 16 && len(bP) >= 16 {
		plain((*[16]float32)(hP), (*[16]float32)(bP))
		hP, bP = hP[16:], bP[16:]
	}
	for len(hB) >= 64 && len(aB) >= 64 && len(bB) >= 64 {
		hh := (*[64]float32)(hB)
		aa := (*[64]float32)(aB)
		bb := (*[64]float32)(bB)
		fused((*[16]float32)(hh[0:16]), (*[16]float32)(aa[0:16]), (*[16]float32)(bb[0:16]))
		fused((*[16]float32)(hh[16:32]), (*[16]float32)(aa[16:32]), (*[16]float32)(bb[16:32]))
		fused((*[16]float32)(hh[32:48]), (*[16]float32)(aa[32:48]), (*[16]float32)(bb[32:48]))
		fused((*[16]float32)(hh[48:64]), (*[16]float32)(aa[48:64]), (*[16]float32)(bb[48:64]))
		hB, aB, bB = hB[64:], aB[64:], bB[64:]
	}
	for len(hB) >= 16 && len(aB) >= 16 && len(bB) >= 16 {
		fused((*[16]float32)(hB), (*[16]float32)(aB), (*[16]float32)(bB))
		hB, aB, bB = hB[16:], aB[16:], bB[16:]
	}
	for len(hQ) >= 16 && len(bQ) >= 16 {
		plain((*[16]float32)(hQ), (*[16]float32)(bQ))
		hQ, bQ = hQ[16:], bQ[16:]
	}
	dlyHsumAvx2(sv, s8, x2, x28).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}
