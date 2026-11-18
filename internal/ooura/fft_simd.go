//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import (
	"simd/archsimd"
	"unsafe"
)

// archsimd kernel shared by the NEON (arm64) and SSE2 (amd64) lanes.
//
// ooura_fft_neon.cc and ooura_fft_sse2.cc perform the same IEEE operation
// sequence for every output element: both multiply by the {-1,1,-1,1} sign
// table where the other one does (vmlaq_f32 is unfused, so vmlaq(x, s, y)
// rounds s*y exactly like _mm_mul_ps followed by _mm_add_ps), and their
// shuffles only move lanes. This kernel keeps that per-element arithmetic
// and is free in how it moves lanes: it processes two cftmdl iterations per
// vector, loads and stores whole vectors, and uses whichever permutation the
// architecture offers (shuffles in fft_simd_{arm64,amd64}.go). Products are
// always Mul followed by Add/Sub, never MulAdd, so nothing is fused.
// TestSimdMatchesRef checks it bit-exactly against fftNeonRef and
// fftSSE2Ref.

type vec = archsimd.Float32x4

func vld(a *[128]float32, i int) vec {
	return archsimd.LoadFloat32x4Array((*[4]float32)(a[i : i+4]))
}

func vst(a *[128]float32, i int, x vec) { x.StoreArray((*[4]float32)(a[i : i+4])) }

func vtab(t []float32, i int) vec { return archsimd.LoadFloat32x4Array((*[4]float32)(t[i : i+4])) }

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

// rftWkr[m-1] = 0.5f - c[32-m] for m = 1..28: the wkr operand of the
// vectorized rftfsub/rftbsub iterations in j1 order (the C code computes the
// same float subtraction and reverses the lanes).
var rftWkr [28]float32

func init() {
	c := rdftW[32:]
	for m := 1; m <= 28; m++ {
		rftWkr[m-1] = 0.5 - c[32-m]
	}
}

func fftSimd(a *[128]float32) {
	bitrv2Simd(a)
	cft1stSimd(a)
	cftmdlSimd(a)
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
	bitrv2Simd(a)
	cft1stSimd(a)
	cftmdlSimd(a)
	cftbsubTailSimd(a)
}

// bitrv2Simd is bitrv2_128 unrolled: the same 28 swaps of complex pairs,
// moved as 64-bit words (indices in units of two floats).
func bitrv2Simd(a *[128]float32) {
	u := (*[64]uint64)(unsafe.Pointer(a))
	u[4], u[8] = u[8], u[4]
	u[32], u[1] = u[1], u[32]
	u[36], u[9] = u[9], u[36]
	u[40], u[5] = u[5], u[40]
	u[44], u[13] = u[13], u[44]
	u[37], u[41] = u[41], u[37]
	u[16], u[2] = u[2], u[16]
	u[20], u[10] = u[10], u[20]
	u[24], u[6] = u[6], u[24]
	u[28], u[14] = u[14], u[28]
	u[17], u[34] = u[34], u[17]
	u[21], u[42] = u[42], u[21]
	u[25], u[38] = u[38], u[25]
	u[29], u[46] = u[46], u[29]
	u[22], u[26] = u[26], u[22]
	u[48], u[3] = u[3], u[48]
	u[52], u[11] = u[11], u[52]
	u[56], u[7] = u[7], u[56]
	u[60], u[15] = u[15], u[60]
	u[49], u[35] = u[35], u[49]
	u[53], u[43] = u[43], u[53]
	u[57], u[39] = u[39], u[57]
	u[61], u[47] = u[47], u[61]
	u[50], u[19] = u[19], u[50]
	u[54], u[27] = u[27], u[54]
	u[58], u[23] = u[23], u[58]
	u[62], u[31] = u[31], u[62]
	u[55], u[59] = u[59], u[55]
}

func cft1stSimd(a *[128]float32) {
	swap := archsimd.LoadFloat32x4Array(&swapSign)
	for j, k2 := 0, 0; j < 128; j, k2 = j+16, k2+4 {
		a00 := vld(a, j)
		a04 := vld(a, j+4)
		a08 := vld(a, j+8)
		a12 := vld(a, j+12)
		a01 := lo64(a00, a08)
		a23 := hi64(a00, a08)
		a45 := lo64(a04, a12)
		a67 := hi64(a04, a12)
		x0 := a01.Add(a23)
		x1 := a01.Sub(a23)
		x2 := a45.Add(a67)
		x3 := a45.Sub(a67)
		x3s := rev64(x3).Mul(swap)
		a01 = x0.Add(x2)
		x0 = x0.Sub(x2)
		a45 = vtab(rdftWk2r[:], k2).Mul(x0).Add(vtab(rdftWk2i[:], k2).Mul(rev64(x0)))
		x0 = x1.Add(x3s)
		a23 = vtab(rdftWk1r[:], k2).Mul(x0).Add(vtab(rdftWk1i[:], k2).Mul(rev64(x0)))
		x0 = x1.Sub(x3s)
		a67 = vtab(rdftWk3r[:], k2).Mul(x0).Add(vtab(rdftWk3i[:], k2).Mul(rev64(x0)))
		vst(a, j+0, lo64(a01, a23))
		vst(a, j+4, lo64(a45, a67))
		vst(a, j+8, hi64(a01, a23))
		vst(a, j+12, hi64(a45, a67))
	}
}

// cftmdlSimd runs cftmdl_128 with two C iterations (j and j+2) per pass:
// the low 64 bits of each whole-vector load belong to iteration j, the high
// 64 bits to iteration j+2.
func cftmdlSimd(a *[128]float32) {
	swap := archsimd.LoadFloat32x4Array(&swapSign)
	wk1rv := archsimd.LoadFloat32x4Array(&cftmdlWk1r)
	for j := 0; j < 8; j += 4 {
		A0, A8, A16, A24 := vld(a, j), vld(a, j+8), vld(a, j+16), vld(a, j+24)
		A32, A40, A48, A56 := vld(a, j+32), vld(a, j+40), vld(a, j+48), vld(a, j+56)
		xx0a, xx1a, adda, suba, yy4a := cftmdlFirst(lo64(A0, A32), lo64(A8, A40), lo64(A16, A48), lo64(A24, A56), swap, wk1rv)
		xx0b, xx1b, addb, subb, yy4b := cftmdlFirst(hi64(A0, A32), hi64(A8, A40), hi64(A16, A48), hi64(A24, A56), swap, wk1rv)
		vst(a, j+0, lo64(xx0a, xx0b))
		vst(a, j+32, hi64(xx0a, xx0b))
		vst(a, j+16, lo64(xx1a, xx1b))
		vst(a, j+48, negEven(rev64(hi64(xx1a, xx1b))))
		vst(a, j+8, lo64(adda, addb))
		vst(a, j+24, lo64(suba, subb))
		vst(a, j+40, lo64(yy4a, yy4b))
		vst(a, j+56, rev64(hi64(yy4a, yy4b)))
	}
	const k2 = 4
	wk2rv := vtab(rdftWk2r[:], k2)
	wk2iv := vtab(rdftWk2i[:], k2)
	wk1iv := vtab(rdftWk1i[:], k2)
	wk3rv := vtab(rdftWk3r[:], k2)
	wk3iv := vtab(rdftWk3i[:], k2)
	wk1rv = vtab(rdftWk1r[:], k2)
	for j := 64; j < 72; j += 4 {
		A0, A8, A16, A24 := vld(a, j), vld(a, j+8), vld(a, j+16), vld(a, j+24)
		A32, A40, A48, A56 := vld(a, j+32), vld(a, j+40), vld(a, j+48), vld(a, j+56)
		var xx, xx4, xx12, xx22 [2]vec
		for h := 0; h < 2; h++ {
			var p0, p8, p16, p24 vec
			if h == 0 {
				p0, p8, p16, p24 = lo64(A0, A32), lo64(A8, A40), lo64(A16, A48), lo64(A24, A56)
			} else {
				p0, p8, p16, p24 = hi64(A0, A32), hi64(A8, A40), hi64(A16, A48), hi64(A24, A56)
			}
			x0 := p0.Add(p8)
			x1 := p0.Sub(p8)
			x2 := p16.Add(p24)
			x3 := p16.Sub(p24)
			xx[h] = x0.Add(x2)
			xx1 := x0.Sub(x2)
			x3s := swap.Mul(rev64(x3))
			add := x1.Add(x3s)
			sub := x1.Sub(x3s)
			xx4[h] = wk2rv.Mul(xx1).Add(wk2iv.Mul(rev64(xx1)))
			xx12[h] = wk1rv.Mul(add).Add(wk1iv.Mul(rev64(add)))
			xx22[h] = wk3rv.Mul(sub).Add(wk3iv.Mul(rev64(sub)))
		}
		vst(a, j+0, lo64(xx[0], xx[1]))
		vst(a, j+32, hi64(xx[0], xx[1]))
		vst(a, j+16, lo64(xx4[0], xx4[1]))
		vst(a, j+48, hi64(xx4[0], xx4[1]))
		vst(a, j+8, lo64(xx12[0], xx12[1]))
		vst(a, j+40, hi64(xx12[0], xx12[1]))
		vst(a, j+24, lo64(xx22[0], xx22[1]))
		vst(a, j+56, hi64(xx22[0], xx22[1]))
	}
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
	for j := 0; j < 32; j += 4 {
		p0, p1, p2, p3 := vld(a, j), vld(a, j+32), vld(a, j+64), vld(a, j+96)
		x0 := p0.Add(p1)
		x1 := p0.Sub(p1)
		x2 := p2.Add(p3)
		x3s := rev64(p2.Sub(p3)).Mul(swap) // -x3i, x3r
		vst(a, j, x0.Add(x2))
		vst(a, j+64, x0.Sub(x2))
		vst(a, j+32, x1.Add(x3s))
		vst(a, j+96, x1.Sub(x3s))
	}
}

// cftbsubTailSimd is the scalar final pass of cftbsub_128. The C code's
// -a[j+1] - a[j1+1] equals (-a[j+1]) + (-a[j1+1]) exactly, so negating the
// imaginary lanes first reproduces every expression.
func cftbsubTailSimd(a *[128]float32) {
	sgn := archsimd.LoadFloat32x4Array(&signIm)
	for j := 0; j < 32; j += 4 {
		n0, n1 := vld(a, j).Mul(sgn), vld(a, j+32).Mul(sgn)
		p2, p3 := vld(a, j+64), vld(a, j+96)
		x0 := n0.Add(n1)
		x1 := n0.Sub(n1)
		x2s := p2.Add(p3).Mul(sgn)
		x3w := rev64(p2.Sub(p3))
		vst(a, j, x0.Add(x2s))
		vst(a, j+64, x0.Sub(x2s))
		vst(a, j+32, x1.Sub(x3w))
		vst(a, j+96, x1.Add(x3w))
	}
}

// rftLoad deinterleaves a[j2:j2+8] and the mirrored a[122-j2:130-j2] so
// that lane i of every vector holds the operands of C iteration j2+2i.
func rftLoad(a *[128]float32, j2 int) (p0, p1, k0, k1 vec) {
	aj0, aj4 := vld(a, j2), vld(a, j2+4)
	ak0, ak4 := swap64(vld(a, 122-j2)), swap64(vld(a, 126-j2))
	return even(aj0, aj4), odd(aj0, aj4), even(ak4, ak0), odd(ak4, ak0)
}

func rftStore(a *[128]float32, j2 int, p0, p1, k0, k1 vec) {
	vst(a, j2, zipLo(p0, p1))
	vst(a, j2+4, zipHi(p0, p1))
	vst(a, 126-j2, swap64(zipLo(k0, k1)))
	vst(a, 122-j2, swap64(zipHi(k0, k1)))
}

func rftfsubSimd(a *[128]float32) {
	c := rdftW[32:]
	j1, j2 := 1, 2
	for ; j2+7 < 64; j1, j2 = j1+4, j2+8 {
		wkr := vtab(rftWkr[:], j1-1)
		wki := vtab(c, j1)
		p0, p1, k0, k1 := rftLoad(a, j2)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		rftStore(a, j2, p0.Sub(yr), p1.Sub(yi), k0.Add(yr), k1.Sub(yi))
	}
	for ; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2+0] - a[k2+0]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) - float32(wki*xi)
		yi := float32(wkr*xi) + float32(wki*xr)
		a[j2+0] -= yr
		a[j2+1] -= yi
		a[k2+0] += yr
		a[k2+1] -= yi
	}
}

func rftbsubSimd(a *[128]float32) {
	c := rdftW[32:]
	a[1] = -a[1]
	j1, j2 := 1, 2
	for ; j2+7 < 64; j1, j2 = j1+4, j2+8 {
		wkr := vtab(rftWkr[:], j1-1)
		wki := vtab(c, j1)
		p0, p1, k0, k1 := rftLoad(a, j2)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		rftStore(a, j2, p0.Sub(yr), yi.Sub(p1), k0.Add(yr), yi.Sub(k1))
	}
	for ; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2+0] - a[k2+0]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) + float32(wki*xi)
		yi := float32(wkr*xi) - float32(wki*xr)
		a[j2+0] = a[j2+0] - yr
		a[j2+1] = yi - a[j2+1]
		a[k2+0] = yr + a[k2+0]
		a[k2+1] = yi - a[k2+1]
	}
	a[65] = -a[65]
}
