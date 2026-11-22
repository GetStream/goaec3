//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import "simd/archsimd"

// archsimd kernel shared by the NEON (arm64) and SSE2 (amd64) lanes.
//
// ooura_fft_neon.cc and ooura_fft_sse2.cc perform the same IEEE operation
// sequence for every output element: both multiply by the {-1,1,-1,1} sign
// table where the other one does (vmlaq_f32 is unfused, so vmlaq(x, s, y)
// rounds s*y exactly like _mm_mul_ps followed by _mm_add_ps), and their
// shuffles only move lanes. This kernel keeps that per-element arithmetic
// and is free in how it moves lanes: it processes two cftmdl iterations per
// vector, loads and stores whole vectors, and uses whichever permutation the
// architecture offers (shuffles in fft_simd_{arm64,amd64}.go), which also
// folds the bit-reversal permutation into cft1st's own shuffles. Products are
// always Mul followed by Add/Sub, never MulAdd, so nothing is fused.
// TestSimdMatchesRef checks it bit-exactly against fftNeonRef and
// fftSSE2Ref.

type vec = archsimd.Float32x4

// Loads and stores go through array pointers converted from slices whose
// bounds the compiler proves: a constant window of the frame (or table)
// first, then four floats at an offset that is a constant or a bounded
// function of a loop counter. The helpers below inline into the passes, and
// no load or store of a pass carries a bounds check (TestNoBoundsChecks).

func ld(p *[4]float32) vec { return archsimd.LoadFloat32x4Array(p) }

// f4 is the four floats at the constant offset off of a window. Each pass
// takes one window per iteration (a[j:j+c] for its counter j) and reaches
// its vectors at constant offsets from it, which the compiler turns into
// one base register and displacement addressing.
func f4w60(p *[60]float32, off int) *[4]float32 { return (*[4]float32)(p[off : off+4]) }

func f4w100(p *[100]float32, off int) *[4]float32 { return (*[4]float32)(p[off : off+4]) }

// tab is t[k2:k2+4] for k2 in {0, 4, ..., 28}.
func tab(t *[32]float32, k2 int) vec {
	return archsimd.LoadFloat32x4Array((*[4]float32)(t[k2 : k2+4]))
}

// zipLo is {x0, y0, x1, y1}; zipHi is {x2, y2, x3, y3}.
func zipLo(x, y vec) vec { return x.ToBits().InterleaveLo(y.ToBits()).BitsToFloat32() }

func zipHi(x, y vec) vec { return x.ToBits().InterleaveHi(y.ToBits()).BitsToFloat32() }

// negEven flips the sign of lanes 0 and 2 (C unary minus).
func negEven(x vec) vec {
	return x.ToBits().Xor(archsimd.LoadUint32x4Array(&signEven)).BitsToFloat32()
}

var signEven = [4]uint32{0x80000000, 0, 0x80000000, 0}

// signIm is {1,-1,1,-1}; multiplying by it is exact negation of the
// imaginary lanes.
var signIm = [4]float32{1, -1, 1, -1}

// rftWkr[m-1] = 0.5f - c[32-m] for m = 1..28 (c = rdft_w + 32): the wkr
// operand of the vectorized rftfsub/rftbsub iterations in j1 order (the C
// code computes the same float subtraction and reverses the lanes).
var rftWkr [28]float32

func init() {
	for m := 1; m <= 28; m++ {
		rftWkr[m-1] = 0.5 - rdftW[64-m]
	}
}

func fftSimd(a *[128]float32) {
	var t [128]float32
	bitrv2Cft1stSimd(a, &t)
	cftmdlSimd(&t, a)
	cftfsubTailSimd(a)
	rftfsubSimd(a)
	xi := a[0] - a[1]
	a[0] += a[1]
	a[1] = xi
}

func inverseFftSimd(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	rftbsubSimd(a)
	var t [128]float32
	bitrv2Cft1stSimd(a, &t)
	cftmdlSimd(&t, a)
	cftbsubTailSimd(a)
}

// bitrv2Cft1stSimd is bitrv2_128 followed by cft1st_128, from a into t.
//
// bitrv2_128 swaps the complex values (float pairs) u[i] and u[rev6(i)],
// where rev6 reverses the six index bits. In vectors of two pairs
// (V[m] = {u[2m], u[2m+1]}) the permuted frame is, for m in [0, 16),
//
//	W[rev4(m)]    = {low pair of V[m], low pair of V[16+m]}
//	W[16+rev4(m)] = {high pair of V[m], high pair of V[16+m]}
//
// and cft1st block q (vectors W[4q..4q+3]) starts by regrouping its pairs
// into a01 = {u'[0], u'[4]}, a23 = {u'[1], u'[5]}, a45 = {u'[2], u'[6]} and
// a67 = {u'[3], u'[7]}. With r = rev2(q) and Vi = V[4i+r] those are
//
//	block q:   a01 = lo64(V0, V1), a23 = lo64(V4, V5), a45 = lo64(V2, V3), a67 = lo64(V6, V7)
//	block q+4: the same with hi64
//
// so the bit reversal costs no instruction: the eight loads of one r feed
// two cft1st blocks through the shuffles cft1st performs anyway. Only lanes
// move differently; every float operation is the one of cft1st_128.
func bitrv2Cft1stSimd(a, t *[128]float32) {
	swap := archsimd.LoadFloat32x4Array(&swapSign)
	thi := (*[64]float32)(t[64:])
	// r is the residue and q its two bits reversed.
	blocks := func(r, q int) {
		src := (*[116]float32)(a[4*r : 4*r+116])
		v0, v1 := ld((*[4]float32)(src[0:4])), ld((*[4]float32)(src[16:20]))
		v2, v3 := ld((*[4]float32)(src[32:36])), ld((*[4]float32)(src[48:52]))
		v4, v5 := ld((*[4]float32)(src[64:68])), ld((*[4]float32)(src[80:84]))
		v6, v7 := ld((*[4]float32)(src[96:100])), ld((*[4]float32)(src[112:116]))
		b01, b23, b45, b67 := cft1stBlock(lo64(v0, v1), lo64(v4, v5), lo64(v2, v3), lo64(v6, v7), swap, 4*q)
		d := (*[16]float32)(t[16*q : 16*q+16])
		lo64(b01, b23).StoreArray((*[4]float32)(d[0:4]))
		lo64(b45, b67).StoreArray((*[4]float32)(d[4:8]))
		hi64(b01, b23).StoreArray((*[4]float32)(d[8:12]))
		hi64(b45, b67).StoreArray((*[4]float32)(d[12:16]))
		b01, b23, b45, b67 = cft1stBlock(hi64(v0, v1), hi64(v4, v5), hi64(v2, v3), hi64(v6, v7), swap, 4*q+16)
		d = (*[16]float32)(thi[16*q : 16*q+16])
		lo64(b01, b23).StoreArray((*[4]float32)(d[0:4]))
		lo64(b45, b67).StoreArray((*[4]float32)(d[4:8]))
		hi64(b01, b23).StoreArray((*[4]float32)(d[8:12]))
		hi64(b45, b67).StoreArray((*[4]float32)(d[12:16]))
	}
	blocks(0, 0)
	blocks(1, 2)
	blocks(2, 1)
	blocks(3, 3)
}

// cft1stBlock is the arithmetic of the cft1st_128 iteration j = 4*k2
// (k2 in {0, 4, ..., 28}).
func cft1stBlock(a01, a23, a45, a67, swap vec, k2 int) (b01, b23, b45, b67 vec) {
	k2 &= 28
	x0 := a01.Add(a23)
	x1 := a01.Sub(a23)
	x2 := a45.Add(a67)
	x3 := a45.Sub(a67)
	x3s := rev64(x3).Mul(swap)
	b01 = x0.Add(x2)
	x0 = x0.Sub(x2)
	b45 = tab(&rdftWk2r, k2).Mul(x0).Add(tab(&rdftWk2i, k2).Mul(rev64(x0)))
	x0 = x1.Add(x3s)
	b23 = tab(&rdftWk1r, k2).Mul(x0).Add(tab(&rdftWk1i, k2).Mul(rev64(x0)))
	x0 = x1.Sub(x3s)
	b67 = tab(&rdftWk3r, k2).Mul(x0).Add(tab(&rdftWk3i, k2).Mul(rev64(x0)))
	return
}

// cftmdlSimd runs cftmdl_128 from s into a with two C iterations (j and
// j+2) per pass: the low 64 bits of each whole-vector load belong to
// iteration j, the high 64 bits to iteration j+2.
func cftmdlSimd(s, a *[128]float32) {
	swap := archsimd.LoadFloat32x4Array(&swapSign)
	wk1rv := archsimd.LoadFloat32x4Array(&cftmdlWk1r)
	first := func(j int) {
		sp := (*[60]float32)(s[j : j+60])
		ap := (*[60]float32)(a[j : j+60])
		A0, A8, A16, A24 := ld(f4w60(sp, 0)), ld(f4w60(sp, 8)), ld(f4w60(sp, 16)), ld(f4w60(sp, 24))
		A32, A40, A48, A56 := ld(f4w60(sp, 32)), ld(f4w60(sp, 40)), ld(f4w60(sp, 48)), ld(f4w60(sp, 56))
		xx0a, xx1a, adda, suba, yy4a := cftmdlFirst(lo64(A0, A32), lo64(A8, A40), lo64(A16, A48), lo64(A24, A56), swap, wk1rv)
		xx0b, xx1b, addb, subb, yy4b := cftmdlFirst(hi64(A0, A32), hi64(A8, A40), hi64(A16, A48), hi64(A24, A56), swap, wk1rv)
		lo64(xx0a, xx0b).StoreArray(f4w60(ap, 0))
		hi64(xx0a, xx0b).StoreArray(f4w60(ap, 32))
		lo64(xx1a, xx1b).StoreArray(f4w60(ap, 16))
		negEven(rev64(hi64(xx1a, xx1b))).StoreArray(f4w60(ap, 48))
		lo64(adda, addb).StoreArray(f4w60(ap, 8))
		lo64(suba, subb).StoreArray(f4w60(ap, 24))
		lo64(yy4a, yy4b).StoreArray(f4w60(ap, 40))
		rev64(hi64(yy4a, yy4b)).StoreArray(f4w60(ap, 56))
	}
	first(0)
	first(4)
	const k2 = 4
	wk2rv := tab(&rdftWk2r, k2)
	wk2iv := tab(&rdftWk2i, k2)
	wk1iv := tab(&rdftWk1i, k2)
	wk3rv := tab(&rdftWk3r, k2)
	wk3iv := tab(&rdftWk3i, k2)
	wk1rv = tab(&rdftWk1r, k2)
	shi, ahi := (*[64]float32)(s[64:]), (*[64]float32)(a[64:])
	second := func(j int) {
		sp := (*[60]float32)(shi[j : j+60])
		ap := (*[60]float32)(ahi[j : j+60])
		A0, A8, A16, A24 := ld(f4w60(sp, 0)), ld(f4w60(sp, 8)), ld(f4w60(sp, 16)), ld(f4w60(sp, 24))
		A32, A40, A48, A56 := ld(f4w60(sp, 32)), ld(f4w60(sp, 40)), ld(f4w60(sp, 48)), ld(f4w60(sp, 56))
		xxa, xx4a, xx12a, xx22a := cftmdlSecond(lo64(A0, A32), lo64(A8, A40), lo64(A16, A48), lo64(A24, A56), swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv)
		xxb, xx4b, xx12b, xx22b := cftmdlSecond(hi64(A0, A32), hi64(A8, A40), hi64(A16, A48), hi64(A24, A56), swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv)
		lo64(xxa, xxb).StoreArray(f4w60(ap, 0))
		hi64(xxa, xxb).StoreArray(f4w60(ap, 32))
		lo64(xx4a, xx4b).StoreArray(f4w60(ap, 16))
		hi64(xx4a, xx4b).StoreArray(f4w60(ap, 48))
		lo64(xx12a, xx12b).StoreArray(f4w60(ap, 8))
		hi64(xx12a, xx12b).StoreArray(f4w60(ap, 40))
		lo64(xx22a, xx22b).StoreArray(f4w60(ap, 24))
		hi64(xx22a, xx22b).StoreArray(f4w60(ap, 56))
	}
	second(0)
	second(4)
}

// cftmdlSecond is the body of the second cftmdl_128 loop for one
// iteration.
func cftmdlSecond(p0, p8, p16, p24, swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv vec) (xx, xx4, xx12, xx22 vec) {
	x0 := p0.Add(p8)
	x1 := p0.Sub(p8)
	x2 := p16.Add(p24)
	x3 := p16.Sub(p24)
	xx = x0.Add(x2)
	xx1 := x0.Sub(x2)
	x3s := swap.Mul(rev64(x3))
	add := x1.Add(x3s)
	sub := x1.Sub(x3s)
	xx4 = wk2rv.Mul(xx1).Add(wk2iv.Mul(rev64(xx1)))
	xx12 = wk1rv.Mul(add).Add(wk1iv.Mul(rev64(add)))
	xx22 = wk3rv.Mul(sub).Add(wk3iv.Mul(rev64(sub)))
	return
}

// cftmdlFirst is the body of the first cftmdl_128 loop for one iteration.
func cftmdlFirst(a0032, a0840, a1648, a2456, swap, wk1rv vec) (xx0, xx1, add, sub, yy4 vec) {
	x0 := a0032.Add(a0840)
	x1 := a0032.Sub(a0840)
	x2 := a1648.Add(a2456)
	x3 := a1648.Sub(a2456)
	xx0 = x0.Add(x2)
	xx1 = x0.Sub(x2)
	x3s := swap.Mul(rev64(x3))
	add = x1.Add(x3s)
	sub = x1.Sub(x3s)
	u := zipHi(add, sub) // add2 sub2 add3 sub3
	yy0as := zipLo(u, u) // add2 add2 sub2 sub2
	yy1as := zipHi(u, u) // add3 add3 sub3 sub3
	yy0 := yy0as.Add(swap.Mul(yy1as))
	yy4 = wk1rv.Mul(yy0)
	return
}

// cftfsubTailSimd is the scalar final pass of cftfsub_128 on two complex
// values per vector: x1r - x3i is computed as x1r + (-1*x3i), which is the
// same IEEE result.
func cftfsubTailSimd(a *[128]float32) {
	swap := archsimd.LoadFloat32x4Array(&swapSign)
	step := func(j int) {
		w := (*[100]float32)(a[j : j+100])
		q0, q1, q2, q3 := f4w100(w, 0), f4w100(w, 32), f4w100(w, 64), f4w100(w, 96)
		p0, p1, p2, p3 := ld(q0), ld(q1), ld(q2), ld(q3)
		x0 := p0.Add(p1)
		x1 := p0.Sub(p1)
		x2 := p2.Add(p3)
		x3s := rev64(p2.Sub(p3)).Mul(swap) // -x3i, x3r
		x0.Add(x2).StoreArray(q0)
		x0.Sub(x2).StoreArray(q2)
		x1.Add(x3s).StoreArray(q1)
		x1.Sub(x3s).StoreArray(q3)
	}
	step(0)
	step(4)
	step(8)
	step(12)
	step(16)
	step(20)
	step(24)
	step(28)
}

// cftbsubTailSimd is the scalar final pass of cftbsub_128. The C code's
// -a[j+1] - a[j1+1] equals (-a[j+1]) + (-a[j1+1]) exactly, so negating the
// imaginary lanes first reproduces every expression.
func cftbsubTailSimd(a *[128]float32) {
	sgn := archsimd.LoadFloat32x4Array(&signIm)
	step := func(j int) {
		w := (*[100]float32)(a[j : j+100])
		q0, q1, q2, q3 := f4w100(w, 0), f4w100(w, 32), f4w100(w, 64), f4w100(w, 96)
		n0, n1 := ld(q0).Mul(sgn), ld(q1).Mul(sgn)
		p2, p3 := ld(q2), ld(q3)
		x0 := n0.Add(n1)
		x1 := n0.Sub(n1)
		x2s := p2.Add(p3).Mul(sgn)
		x3w := rev64(p2.Sub(p3))
		x0.Add(x2s).StoreArray(q0)
		x0.Sub(x2s).StoreArray(q2)
		x1.Sub(x3w).StoreArray(q1)
		x1.Add(x3w).StoreArray(q3)
	}
	step(0)
	step(4)
	step(8)
	step(12)
	step(16)
	step(20)
	step(24)
	step(28)
}

// The vectorized rftfsub/rftbsub iterations m = 0..6 cover j2 = 2+8m:
// rftLo(a, m) is a[j2:j2+8] and rftHi(a, m) the mirrored a[122-j2:130-j2].

func rftLo(a *[128]float32, m int) *[8]float32 {
	return (*[8]float32)((*[56]float32)(a[2:58])[8*m : 8*m+8])
}

func rftHi(a *[128]float32, m int) *[8]float32 {
	n := 6 - m
	return (*[8]float32)((*[56]float32)(a[72:128])[8*n : 8*n+8])
}

// rftWkiv is c[j1:j1+4] for j1 = 1+4m, c = rdft_w + 32.
func rftWkiv(m int) vec {
	return archsimd.LoadFloat32x4Array((*[4]float32)((*[28]float32)(rdftW[33:61])[4*m : 4*m+4]))
}

// rftWkrv is the matching wkr vector.
func rftWkrv(m int) vec {
	return archsimd.LoadFloat32x4Array((*[4]float32)(rftWkr[4*m : 4*m+4]))
}

// rftLoad deinterleaves lo and the mirrored hi so that lane i of every
// vector holds the operands of C iteration j2+2i.
func rftLoad(lo, hi *[8]float32) (p0, p1, k0, k1 vec) {
	aj0, aj4 := ld((*[4]float32)(lo[0:4])), ld((*[4]float32)(lo[4:8]))
	ak0, ak4 := swap64(ld((*[4]float32)(hi[0:4]))), swap64(ld((*[4]float32)(hi[4:8])))
	return even(aj0, aj4), odd(aj0, aj4), even(ak4, ak0), odd(ak4, ak0)
}

func rftStore(lo, hi *[8]float32, p0, p1, k0, k1 vec) {
	zipLo(p0, p1).StoreArray((*[4]float32)(lo[0:4]))
	zipHi(p0, p1).StoreArray((*[4]float32)(lo[4:8]))
	swap64(zipLo(k0, k1)).StoreArray((*[4]float32)(hi[4:8]))
	swap64(zipHi(k0, k1)).StoreArray((*[4]float32)(hi[0:4]))
}

func rftfsubSimd(a *[128]float32) {
	for m := 0; m < 7; m++ {
		wkr := rftWkrv(m)
		wki := rftWkiv(m)
		lo, hi := rftLo(a, m), rftHi(a, m)
		p0, p1, k0, k1 := rftLoad(lo, hi)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		rftStore(lo, hi, p0.Sub(yr), p1.Sub(yi), k0.Add(yr), k1.Sub(yi))
	}
	c := (*[32]float32)(rdftW[32:])
	for j1 := 29; j1 < 32; j1++ {
		j2 := 2 * j1
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2] - a[k2]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) - float32(wki*xi)
		yi := float32(wkr*xi) + float32(wki*xr)
		a[j2] -= yr
		a[j2+1] -= yi
		a[k2] += yr
		a[k2+1] -= yi
	}
}

func rftbsubSimd(a *[128]float32) {
	a[1] = -a[1]
	for m := 0; m < 7; m++ {
		wkr := rftWkrv(m)
		wki := rftWkiv(m)
		lo, hi := rftLo(a, m), rftHi(a, m)
		p0, p1, k0, k1 := rftLoad(lo, hi)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		rftStore(lo, hi, p0.Sub(yr), yi.Sub(p1), k0.Add(yr), yi.Sub(k1))
	}
	c := (*[32]float32)(rdftW[32:])
	for j1 := 29; j1 < 32; j1++ {
		j2 := 2 * j1
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2] - a[k2]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) + float32(wki*xi)
		yi := float32(wkr*xi) - float32(wki*xr)
		a[j2] = a[j2] - yr
		a[j2+1] = yi - a[j2+1]
		a[k2] = yr + a[k2]
		a[k2+1] = yi - a[k2+1]
	}
	a[65] = -a[65]
}
