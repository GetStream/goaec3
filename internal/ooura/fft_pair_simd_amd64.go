//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// Two transforms in one 256-bit pass (AVX2).
//
// Every operation of the archsimd kernel in fft_simd.go is local to one
// 128-bit vector, so a 256-bit vector whose low half belongs to frame a and
// whose high half belongs to frame b runs both transforms with the same
// IEEE operation per element as two fftSimd calls: the grouped shuffles
// (VSHUFPS, VPUNPCK*DQ on ymm) permute each 128-bit half on its own and the
// tables repeat their four floats in both halves. TestPairMatchesSingle
// checks the pair bit-exactly against two single transforms.
//
// The work buffers hold the two frames interleaved four floats at a time:
// t[8v:8v+4] is a[4v:4v+4] and t[8v+4:8v+8] is b[4v:4v+4], so one 256-bit
// load or store moves the same vector of both frames. The first pass of each
// direction gathers the two frames into that layout and the last pass
// scatters its results into a and b; the passes between work on whole
// 256-bit vectors.

type vec8 = archsimd.Float32x8

// PairScratch is the work area of FftPair and InverseFftPair. It needs no
// initialization and carries nothing from one call to the next.
type PairScratch struct {
	t, u [256]float32
}

var usePair = archsimd.X86.AVX2()

func fftPair(a, b *[128]float32, s *PairScratch) {
	if usePair {
		fftPairSimd(a, b, s)
		return
	}
	fft(a)
	fft(b)
}

func inverseFftPair(a, b *[128]float32, s *PairScratch) {
	if usePair {
		inverseFftPairSimd(a, b, s)
		return
	}
	inverseFft(a)
	inverseFft(b)
}

func ld8(p *[8]float32) vec8 { return archsimd.LoadFloat32x8Array(p) }

// ldPair is {a, b}.
func ldPair(a, b *[4]float32) vec8 {
	var z vec8
	return z.SetLo(ld(a)).SetHi(ld(b))
}

// stPair stores the low half of x to a and the high half to b.
func stPair(x vec8, a, b *[4]float32) {
	x.GetLo().StoreArray(a)
	x.GetHi().StoreArray(b)
}

// The shuffles of fft_simd_amd64.go on each 128-bit half.

func rev64g(x vec8) vec8    { return x.ConcatPermuteScalarsGrouped(1, 0, 7, 6, x) }
func swap64g(x vec8) vec8   { return x.ConcatPermuteScalarsGrouped(2, 3, 4, 5, x) }
func lo64g(x, y vec8) vec8  { return x.ConcatPermuteScalarsGrouped(0, 1, 4, 5, y) }
func hi64g(x, y vec8) vec8  { return x.ConcatPermuteScalarsGrouped(2, 3, 6, 7, y) }
func eveng(x, y vec8) vec8  { return x.ConcatPermuteScalarsGrouped(0, 2, 4, 6, y) }
func oddg(x, y vec8) vec8   { return x.ConcatPermuteScalarsGrouped(1, 3, 5, 7, y) }
func zipLog(x, y vec8) vec8 { return x.ToBits().InterleaveLoGrouped(y.ToBits()).BitsToFloat32() }
func zipHig(x, y vec8) vec8 { return x.ToBits().InterleaveHiGrouped(y.ToBits()).BitsToFloat32() }

// mid64g is {x2, x3, y0, y1} per half: the four floats that follow x's low
// pair when y follows x in memory.
func mid64g(x, y vec8) vec8 { return x.ConcatPermuteScalarsGrouped(2, 3, 4, 5, y) }

func negEveng(x vec8) vec8 {
	return x.ToBits().Xor(archsimd.LoadUint32x8Array(&signEven8)).BitsToFloat32()
}

// The tables of the 128-bit kernel with every vector repeated.
var (
	signEven8           = [8]uint32{0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0}
	signIm8             = [8]float32{1, -1, 1, -1, 1, -1, 1, -1}
	swapSign8           = [8]float32{-1, 1, -1, 1, -1, 1, -1, 1}
	cftmdlWk1r8         [8]float32
	wk1r8, wk2r8, wk3r8 [64]float32
	wk1i8, wk2i8, wk3i8 [64]float32
	rftWkr8, rftWki8    [56]float32
)

// initPairTables runs once the negative zeros of the 128-bit tables are in
// place (tables.go).
func initPairTables() {
	dup := func(dst, src []float32) {
		for v := 0; 4*v < len(src); v++ {
			copy(dst[8*v:8*v+4], src[4*v:4*v+4])
			copy(dst[8*v+4:8*v+8], src[4*v:4*v+4])
		}
	}
	dup(cftmdlWk1r8[:], cftmdlWk1r[:])
	dup(wk1r8[:], rdftWk1r[:])
	dup(wk2r8[:], rdftWk2r[:])
	dup(wk3r8[:], rdftWk3r[:])
	dup(wk1i8[:], rdftWk1i[:])
	dup(wk2i8[:], rdftWk2i[:])
	dup(wk3i8[:], rdftWk3i[:])
	var wkr [28]float32
	for m := 1; m <= 28; m++ {
		wkr[m-1] = 0.5 - rdftW[64-m]
	}
	dup(rftWkr8[:], wkr[:])
	dup(rftWki8[:], rdftW[33:61])
}

// tab8 is the repeated t[k2:k2+4] of the 128-bit tables; k is 2*k2 in
// {0, 8, ..., 56}.
func tab8(t *[64]float32, k int) vec8 {
	k &= 56
	return archsimd.LoadFloat32x8Array((*[8]float32)(t[k : k+8]))
}

// p16 is the vector at offset j (0 or 8) of the 16-float window of t that
// starts at base; p64 the one at offset j (0, 8, ..., 56) of a 64-float
// window.
func p16(t *[256]float32, base, j int) *[8]float32 {
	return (*[8]float32)((*[16]float32)(t[base : base+16])[j : j+8])
}

func p64(t *[256]float32, base, j int) *[8]float32 {
	return (*[8]float32)((*[64]float32)(t[base : base+64])[j : j+8])
}

func fftPairSimd(a, b *[128]float32, s *PairScratch) {
	t := &s.t
	bitrv2Cft1stPair(a, b, nil, t)
	cftmdlPair(t)
	cftfsubTailPair(t)
	rftfsubScatter(t, a, b)
	archsimd.ClearAVXUpperBits()
	rftfsubEnd(a)
	rftfsubEnd(b)
}

func inverseFftPairSimd(a, b *[128]float32, s *PairScratch) {
	rftbsubBegin(a)
	rftbsubBegin(b)
	t, u := &s.t, &s.u
	rftbsubGather(a, b, t)
	bitrv2Cft1stPair(nil, nil, t, u)
	cftmdlPair(u)
	cftbsubTailScatter(u, a, b)
	archsimd.ClearAVXUpperBits()
}

// rftfsubEnd is what follows the vectorized rftfsub iterations in Fft: the
// three scalar iterations and the final butterfly of a[0] and a[1].
func rftfsubEnd(a *[128]float32) {
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
	xi := a[0] - a[1]
	a[0] += a[1]
	a[1] = xi
}

// rftbsubBegin is the scalar part of InverseFft ahead of the vectorized
// rftbsub iterations: the first butterfly, the sign changes of a[1] and
// a[65] and the three scalar iterations. None of it reads an element the
// vectorized iterations write, and those read none it writes.
func rftbsubBegin(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	a[1] = -a[1]
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

// bitrv2Cft1stPair is bitrv2Cft1stSimd into the interleaved t, either from
// the interleaved s or (s == nil) gathering the frames a and b.
func bitrv2Cft1stPair(a, b *[128]float32, s, t *[256]float32) {
	swap := ld8(&swapSign8)
	thi := (*[128]float32)(t[128:])
	for r := 0; r < 4; r++ {
		var v0, v1, v2, v3, v4, v5, v6, v7 vec8
		if s != nil {
			src := (*[232]float32)(s[8*r : 8*r+232])
			v0, v1 = ld8((*[8]float32)(src[0:8])), ld8((*[8]float32)(src[32:40]))
			v2, v3 = ld8((*[8]float32)(src[64:72])), ld8((*[8]float32)(src[96:104]))
			v4, v5 = ld8((*[8]float32)(src[128:136])), ld8((*[8]float32)(src[160:168]))
			v6, v7 = ld8((*[8]float32)(src[192:200])), ld8((*[8]float32)(src[224:232]))
		} else {
			sa := (*[116]float32)(a[4*r : 4*r+116])
			sb := (*[116]float32)(b[4*r : 4*r+116])
			v0 = ldPair((*[4]float32)(sa[0:4]), (*[4]float32)(sb[0:4]))
			v1 = ldPair((*[4]float32)(sa[16:20]), (*[4]float32)(sb[16:20]))
			v2 = ldPair((*[4]float32)(sa[32:36]), (*[4]float32)(sb[32:36]))
			v3 = ldPair((*[4]float32)(sa[48:52]), (*[4]float32)(sb[48:52]))
			v4 = ldPair((*[4]float32)(sa[64:68]), (*[4]float32)(sb[64:68]))
			v5 = ldPair((*[4]float32)(sa[80:84]), (*[4]float32)(sb[80:84]))
			v6 = ldPair((*[4]float32)(sa[96:100]), (*[4]float32)(sb[96:100]))
			v7 = ldPair((*[4]float32)(sa[112:116]), (*[4]float32)(sb[112:116]))
		}
		q := (r>>1 | r<<1) & 3
		b01, b23, b45, b67 := cft1stBlock8(lo64g(v0, v1), lo64g(v4, v5), lo64g(v2, v3), lo64g(v6, v7), swap, 8*q)
		d := (*[32]float32)(t[32*q : 32*q+32])
		lo64g(b01, b23).StoreArray((*[8]float32)(d[0:8]))
		lo64g(b45, b67).StoreArray((*[8]float32)(d[8:16]))
		hi64g(b01, b23).StoreArray((*[8]float32)(d[16:24]))
		hi64g(b45, b67).StoreArray((*[8]float32)(d[24:32]))
		b01, b23, b45, b67 = cft1stBlock8(hi64g(v0, v1), hi64g(v4, v5), hi64g(v2, v3), hi64g(v6, v7), swap, 8*q+32)
		d = (*[32]float32)(thi[32*q : 32*q+32])
		lo64g(b01, b23).StoreArray((*[8]float32)(d[0:8]))
		lo64g(b45, b67).StoreArray((*[8]float32)(d[8:16]))
		hi64g(b01, b23).StoreArray((*[8]float32)(d[16:24]))
		hi64g(b45, b67).StoreArray((*[8]float32)(d[24:32]))
	}
}

// cft1stBlock8 is cft1stBlock on both halves; k is twice its k2.
func cft1stBlock8(a01, a23, a45, a67, swap vec8, k int) (b01, b23, b45, b67 vec8) {
	x0 := a01.Add(a23)
	x1 := a01.Sub(a23)
	x2 := a45.Add(a67)
	x3 := a45.Sub(a67)
	x3s := rev64g(x3).Mul(swap)
	b01 = x0.Add(x2)
	x0 = x0.Sub(x2)
	b45 = tab8(&wk2r8, k).Mul(x0).Add(tab8(&wk2i8, k).Mul(rev64g(x0)))
	x0 = x1.Add(x3s)
	b23 = tab8(&wk1r8, k).Mul(x0).Add(tab8(&wk1i8, k).Mul(rev64g(x0)))
	x0 = x1.Sub(x3s)
	b67 = tab8(&wk3r8, k).Mul(x0).Add(tab8(&wk3i8, k).Mul(rev64g(x0)))
	return
}

// cftmdlPair is cftmdlSimd in place on the interleaved frames: each
// iteration loads its eight vectors before it stores into them.
func cftmdlPair(t *[256]float32) {
	swap := ld8(&swapSign8)
	wk1rv := ld8(&cftmdlWk1r8)
	for j := 0; j < 16; j += 8 {
		q0, q8, q16, q24 := p16(t, 0, j), p16(t, 16, j), p16(t, 32, j), p16(t, 48, j)
		q32, q40, q48, q56 := p16(t, 64, j), p16(t, 80, j), p16(t, 96, j), p16(t, 112, j)
		A0, A8, A16, A24 := ld8(q0), ld8(q8), ld8(q16), ld8(q24)
		A32, A40, A48, A56 := ld8(q32), ld8(q40), ld8(q48), ld8(q56)
		xx0a, xx1a, adda, suba, yy4a := cftmdlFirst8(lo64g(A0, A32), lo64g(A8, A40), lo64g(A16, A48), lo64g(A24, A56), swap, wk1rv)
		xx0b, xx1b, addb, subb, yy4b := cftmdlFirst8(hi64g(A0, A32), hi64g(A8, A40), hi64g(A16, A48), hi64g(A24, A56), swap, wk1rv)
		lo64g(xx0a, xx0b).StoreArray(q0)
		hi64g(xx0a, xx0b).StoreArray(q32)
		lo64g(xx1a, xx1b).StoreArray(q16)
		negEveng(rev64g(hi64g(xx1a, xx1b))).StoreArray(q48)
		lo64g(adda, addb).StoreArray(q8)
		lo64g(suba, subb).StoreArray(q24)
		lo64g(yy4a, yy4b).StoreArray(q40)
		rev64g(hi64g(yy4a, yy4b)).StoreArray(q56)
	}
	const k = 8
	wk2rv := tab8(&wk2r8, k)
	wk2iv := tab8(&wk2i8, k)
	wk1iv := tab8(&wk1i8, k)
	wk3rv := tab8(&wk3r8, k)
	wk3iv := tab8(&wk3i8, k)
	wk1rv = tab8(&wk1r8, k)
	for j := 0; j < 16; j += 8 {
		q0, q8, q16, q24 := p16(t, 128, j), p16(t, 144, j), p16(t, 160, j), p16(t, 176, j)
		q32, q40, q48, q56 := p16(t, 192, j), p16(t, 208, j), p16(t, 224, j), p16(t, 240, j)
		A0, A8, A16, A24 := ld8(q0), ld8(q8), ld8(q16), ld8(q24)
		A32, A40, A48, A56 := ld8(q32), ld8(q40), ld8(q48), ld8(q56)
		xxa, xx4a, xx12a, xx22a := cftmdlSecond8(lo64g(A0, A32), lo64g(A8, A40), lo64g(A16, A48), lo64g(A24, A56), swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv)
		xxb, xx4b, xx12b, xx22b := cftmdlSecond8(hi64g(A0, A32), hi64g(A8, A40), hi64g(A16, A48), hi64g(A24, A56), swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv)
		lo64g(xxa, xxb).StoreArray(q0)
		hi64g(xxa, xxb).StoreArray(q32)
		lo64g(xx4a, xx4b).StoreArray(q16)
		hi64g(xx4a, xx4b).StoreArray(q48)
		lo64g(xx12a, xx12b).StoreArray(q8)
		hi64g(xx12a, xx12b).StoreArray(q40)
		lo64g(xx22a, xx22b).StoreArray(q24)
		hi64g(xx22a, xx22b).StoreArray(q56)
	}
}

// cftmdlSecond8 is the body of the second cftmdl_128 loop for one
// iteration.
func cftmdlSecond8(p0, p8, p16, p24, swap, wk1rv, wk1iv, wk2rv, wk2iv, wk3rv, wk3iv vec8) (xx, xx4, xx12, xx22 vec8) {
	x0 := p0.Add(p8)
	x1 := p0.Sub(p8)
	x2 := p16.Add(p24)
	x3 := p16.Sub(p24)
	xx = x0.Add(x2)
	xx1 := x0.Sub(x2)
	x3s := swap.Mul(rev64g(x3))
	add := x1.Add(x3s)
	sub := x1.Sub(x3s)
	xx4 = wk2rv.Mul(xx1).Add(wk2iv.Mul(rev64g(xx1)))
	xx12 = wk1rv.Mul(add).Add(wk1iv.Mul(rev64g(add)))
	xx22 = wk3rv.Mul(sub).Add(wk3iv.Mul(rev64g(sub)))
	return
}

func cftmdlFirst8(a0032, a0840, a1648, a2456, swap, wk1rv vec8) (xx0, xx1, add, sub, yy4 vec8) {
	x0 := a0032.Add(a0840)
	x1 := a0032.Sub(a0840)
	x2 := a1648.Add(a2456)
	x3 := a1648.Sub(a2456)
	xx0 = x0.Add(x2)
	xx1 = x0.Sub(x2)
	x3s := swap.Mul(rev64g(x3))
	add = x1.Add(x3s)
	sub = x1.Sub(x3s)
	u := zipHig(add, sub)
	yy0as := zipLog(u, u)
	yy1as := zipHig(u, u)
	yy0 := yy0as.Add(swap.Mul(yy1as))
	yy4 = wk1rv.Mul(yy0)
	return
}

// cftfsubTailPair is cftfsubTailSimd in place on the interleaved frames.
func cftfsubTailPair(t *[256]float32) {
	swap := ld8(&swapSign8)
	for j := 0; j < 64; j += 8 {
		q0, q1, q2, q3 := p64(t, 0, j), p64(t, 64, j), p64(t, 128, j), p64(t, 192, j)
		p0, p1, p2, p3 := ld8(q0), ld8(q1), ld8(q2), ld8(q3)
		x0 := p0.Add(p1)
		x1 := p0.Sub(p1)
		x2 := p2.Add(p3)
		x3s := rev64g(p2.Sub(p3)).Mul(swap)
		x0.Add(x2).StoreArray(q0)
		x0.Sub(x2).StoreArray(q2)
		x1.Add(x3s).StoreArray(q1)
		x1.Sub(x3s).StoreArray(q3)
	}
}

// cftbsubTailScatter is cftbsubTailSimd from the interleaved frames into a
// and b.
func cftbsubTailScatter(t *[256]float32, a, b *[128]float32) {
	sgn := ld8(&signIm8)
	for j := 0; j < 32; j += 4 {
		n0, n1 := ld8(p64(t, 0, 2*j)).Mul(sgn), ld8(p64(t, 64, 2*j)).Mul(sgn)
		p2, p3 := ld8(p64(t, 128, 2*j)), ld8(p64(t, 192, 2*j))
		x0 := n0.Add(n1)
		x1 := n0.Sub(n1)
		x2s := p2.Add(p3).Mul(sgn)
		x3w := rev64g(p2.Sub(p3))
		stPair(x0.Add(x2s), w32(a, 0, j), w32(b, 0, j))
		stPair(x0.Sub(x2s), w32(a, 64, j), w32(b, 64, j))
		stPair(x1.Sub(x3w), w32(a, 32, j), w32(b, 32, j))
		stPair(x1.Add(x3w), w32(a, 96, j), w32(b, 96, j))
	}
}

// The vectorized rftfsub/rftbsub iteration m reads and writes
// a[2+8m:10+8m] and a[120-8m:128-8m]. In vectors of four (V[v] =
// a[4v:4v+4]) the first window is the high pair of V[2m], all of V[2m+1]
// and the low pair of V[2m+2]; the second is V[30-2m] and V[31-2m].

// rft8Lo is the interleaved V[2m], V[2m+1], V[2m+2].
func rft8Lo(t *[256]float32, m int) *[24]float32 {
	return (*[24]float32)((*[120]float32)(t[0:120])[16*m : 16*m+24])
}

// rft8Hi is the interleaved V[30-2m], V[31-2m].
func rft8Hi(t *[256]float32, m int) *[16]float32 {
	n := 6 - m
	return (*[16]float32)((*[112]float32)(t[144:256])[16*n : 16*n+16])
}

func rft8Tab(t *[56]float32, m int) vec8 {
	return archsimd.LoadFloat32x8Array((*[8]float32)(t[8*m : 8*m+8]))
}

// pairV is the interleaved V[v] of t for a constant v.
func pairV(t *[256]float32, v int) *[8]float32 { return (*[8]float32)(t[8*v : 8*v+8]) }

// rftfsubScatter is the vectorized part of rftfsubSimd from the interleaved
// frames into a and b, which also receive the elements those iterations
// leave alone (a[0:2] and a[58:72]).
func rftfsubScatter(t *[256]float32, a, b *[128]float32) {
	stPair(ld8(pairV(t, 0)), (*[4]float32)(a[0:4]), (*[4]float32)(b[0:4]))
	stPair(ld8(pairV(t, 14)), (*[4]float32)(a[56:60]), (*[4]float32)(b[56:60]))
	stPair(ld8(pairV(t, 15)), (*[4]float32)(a[60:64]), (*[4]float32)(b[60:64]))
	stPair(ld8(pairV(t, 16)), (*[4]float32)(a[64:68]), (*[4]float32)(b[64:68]))
	stPair(ld8(pairV(t, 17)), (*[4]float32)(a[68:72]), (*[4]float32)(b[68:72]))
	for m := 0; m < 7; m++ {
		wkr := rft8Tab(&rftWkr8, m)
		wki := rft8Tab(&rftWki8, m)
		lo, hi := rft8Lo(t, m), rft8Hi(t, m)
		v0, v1, v2 := ld8((*[8]float32)(lo[0:8])), ld8((*[8]float32)(lo[8:16])), ld8((*[8]float32)(lo[16:24]))
		aj0, aj4 := mid64g(v0, v1), mid64g(v1, v2)
		ak0, ak4 := swap64g(ld8((*[8]float32)(hi[0:8]))), swap64g(ld8((*[8]float32)(hi[8:16])))
		p0, p1, k0, k1 := eveng(aj0, aj4), oddg(aj0, aj4), eveng(ak4, ak0), oddg(ak4, ak0)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		p0, p1, k0, k1 = p0.Sub(yr), p1.Sub(yi), k0.Add(yr), k1.Sub(yi)
		la, lb, ha, hb := rftLo(a, m), rftLo(b, m), rftHi(a, m), rftHi(b, m)
		stPair(zipLog(p0, p1), (*[4]float32)(la[0:4]), (*[4]float32)(lb[0:4]))
		stPair(zipHig(p0, p1), (*[4]float32)(la[4:8]), (*[4]float32)(lb[4:8]))
		stPair(swap64g(zipLog(k0, k1)), (*[4]float32)(ha[4:8]), (*[4]float32)(hb[4:8]))
		stPair(swap64g(zipHig(k0, k1)), (*[4]float32)(ha[0:4]), (*[4]float32)(hb[0:4]))
	}
}

// rftbsubGather is the vectorized part of rftbsubSimd from a and b into the
// interleaved t, which also receives the elements those iterations leave
// alone. The results of iteration m are a[2+8m:6+8m] (w0) and a[6+8m:10+8m]
// (w1): V[2m+1] is the high pair of w0 and the low pair of w1, and V[2m+2]
// the high pair of w1 and the low pair of the next w0.
func rftbsubGather(a, b *[128]float32, t *[256]float32) {
	// {a[2], a[3], a[0], a[1]}: its high pair is the low pair of V[0].
	prev := swap64g(ldPair((*[4]float32)(a[0:4]), (*[4]float32)(b[0:4])))
	for m := 0; m < 7; m++ {
		wkr := rft8Tab(&rftWkr8, m)
		wki := rft8Tab(&rftWki8, m)
		la, lb, ha, hb := rftLo(a, m), rftLo(b, m), rftHi(a, m), rftHi(b, m)
		aj0 := ldPair((*[4]float32)(la[0:4]), (*[4]float32)(lb[0:4]))
		aj4 := ldPair((*[4]float32)(la[4:8]), (*[4]float32)(lb[4:8]))
		ak0 := swap64g(ldPair((*[4]float32)(ha[0:4]), (*[4]float32)(hb[0:4])))
		ak4 := swap64g(ldPair((*[4]float32)(ha[4:8]), (*[4]float32)(hb[4:8])))
		p0, p1, k0, k1 := eveng(aj0, aj4), oddg(aj0, aj4), eveng(ak4, ak0), oddg(ak4, ak0)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		p0, p1, k0, k1 = p0.Sub(yr), yi.Sub(p1), k0.Add(yr), yi.Sub(k1)
		w0, w1 := zipLog(p0, p1), zipHig(p0, p1)
		lo, hi := rft8Lo(t, m), rft8Hi(t, m)
		mid64g(prev, w0).StoreArray((*[8]float32)(lo[0:8]))
		mid64g(w0, w1).StoreArray((*[8]float32)(lo[8:16]))
		prev = w1
		swap64g(zipLog(k0, k1)).StoreArray((*[8]float32)(hi[8:16]))
		swap64g(zipHig(k0, k1)).StoreArray((*[8]float32)(hi[0:8]))
	}
	// V[14] is the high pair of the last w1 and a[58:60].
	v14 := ldPair((*[4]float32)(a[56:60]), (*[4]float32)(b[56:60]))
	hi64g(prev, v14).StoreArray(pairV(t, 14))
	ldPair((*[4]float32)(a[60:64]), (*[4]float32)(b[60:64])).StoreArray(pairV(t, 15))
	ldPair((*[4]float32)(a[64:68]), (*[4]float32)(b[64:68])).StoreArray(pairV(t, 16))
	ldPair((*[4]float32)(a[68:72]), (*[4]float32)(b[68:72])).StoreArray(pairV(t, 17))
}
