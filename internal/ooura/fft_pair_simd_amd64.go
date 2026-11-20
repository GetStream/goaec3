//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// Two split transforms in one 256-bit pass (AVX2).
//
// Every operation of the archsimd kernel in fft_simd.go is local to one
// 128-bit vector, so a 256-bit vector whose low half belongs to frame a and
// whose high half belongs to frame b runs both transforms with the same
// IEEE operation per element as two single transforms: the grouped shuffles
// (VSHUFPS, VPUNPCK*DQ, VPALIGNR on ymm) permute each 128-bit half on its
// own and the tables repeat their four floats in both halves.
// TestPairMatchesSingle checks the pair bit-exactly against two single
// transforms.
//
// The work buffers hold the two frames interleaved four floats at a time:
// t[8v:8v+4] is a[4v:4v+4] and t[8v+4:8v+8] is b[4v:4v+4], so one 256-bit
// load or store moves the same vector of both frames. The first pass of each
// direction gathers its two inputs into that layout and the last pass
// separates the results again: it pairs the vectors of two consecutive
// iterations into one 256-bit store per frame (VPERM2F128), so the 256-bit
// loads of whatever reads the results next find each of them in one store.
//
// Go's scalar float code is legacy-SSE encoded and must not run between the
// first 256-bit operation and ClearAVXUpperBits. The passes take the few
// scalar values they need from vectors, or compute them before the first
// 256-bit operation or after the last.

type vec8 = archsimd.Float32x8

// PairScratch is the work area of FftSplitPair and InverseFftSplitPair. It
// needs no initialization and carries nothing from one call to the next.
type PairScratch struct {
	t, u [256]float32
	s    Scratch
}

var usePair = archsimd.X86.AVX2()

func fftSplitPair(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch) {
	if usePair {
		fftSplitPairSoa(a, b, re0, im0, re1, im1, s)
		return
	}
	fftSplit(a, re0, im0, s.single())
	fftSplit(b, re1, im1, s.single())
}

func inverseFftSplitPair(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch) {
	if usePair {
		inverseFftSplitPairSoa(re0, im0, re1, im1, a, b, s)
		return
	}
	inverseFftSplit(re0, im0, a, s.single())
	inverseFftSplit(re1, im1, b, s.single())
}

func ld8(p *[8]float32) vec8 { return archsimd.LoadFloat32x8Array(p) }

// ldPair is {a, b}.
func ldPair(a, b *[4]float32) vec8 {
	var z vec8
	return z.SetLo(ld(a)).SetHi(ld(b))
}

// The shuffles of fft_simd_amd64.go on each 128-bit half.

func rev64g(x vec8) vec8        { return x.ConcatPermuteScalarsGrouped(1, 0, 7, 6, x) }
func rev4g(x vec8) vec8         { return x.ConcatPermuteScalarsGrouped(3, 2, 5, 4, x) }
func lo64g(x, y vec8) vec8      { return x.ConcatPermuteScalarsGrouped(0, 1, 4, 5, y) }
func hi64g(x, y vec8) vec8      { return x.ConcatPermuteScalarsGrouped(2, 3, 6, 7, y) }
func loHig(x, y vec8) vec8      { return x.ConcatPermuteScalarsGrouped(0, 1, 6, 7, y) }
func eveng(x, y vec8) vec8      { return x.ConcatPermuteScalarsGrouped(0, 2, 4, 6, y) }
func oddg(x, y vec8) vec8       { return x.ConcatPermuteScalarsGrouped(1, 3, 5, 7, y) }
func revEveng(hi, lo vec8) vec8 { return hi.ConcatPermuteScalarsGrouped(2, 0, 6, 4, lo) }
func revOddg(hi, lo vec8) vec8  { return hi.ConcatPermuteScalarsGrouped(3, 1, 7, 5, lo) }
func zipLog(x, y vec8) vec8     { return x.ToBits().InterleaveLoGrouped(y.ToBits()).BitsToFloat32() }
func zipHig(x, y vec8) vec8     { return x.ToBits().InterleaveHiGrouped(y.ToBits()).BitsToFloat32() }

func ext3g(x, y vec8) vec8 {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return yb.ConcatShiftBytesRightGrouped(xb, 12).ReshapeToUint32s().BitsToFloat32()
}

func negEveng(x vec8) vec8 {
	return x.ToBits().Xor(archsimd.LoadUint32x8Array(&signEven8)).BitsToFloat32()
}

// lane0g is y with lane 0 of each half taken from x.
func lane0g(x, y vec8) vec8 {
	return x.ConcatPermuteScalarsGrouped(0, 0, 5, 5, y).ConcatPermuteScalarsGrouped(0, 2, 6, 7, y)
}

// frameA is {x.lo, y.lo}: eight consecutive floats of frame a from the
// interleaved vectors x and y. frameB is {x.hi, y.hi}, the same of frame b.
func frameA(x, y vec8) vec8 { return x.ConcatPermute128Scalars(0, 2, y) }
func frameB(x, y vec8) vec8 { return x.ConcatPermute128Scalars(1, 3, y) }

// The tables of the 128-bit kernel with every vector repeated.
var (
	signEven8           = [8]uint32{0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0}
	signIm8             = [8]float32{1, -1, 1, -1, 1, -1, 1, -1}
	swapSign8           = [8]float32{-1, 1, -1, 1, -1, 1, -1, 1}
	cftmdlWk1r8         [8]float32
	wk1r8, wk2r8, wk3r8 [64]float32
	wk1i8, wk2i8, wk3i8 [64]float32
	rftWkr8, rftWki8    [64]float32
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
	var wkr [32]float32
	for j1 := 1; j1 < 32; j1++ {
		wkr[j1] = 0.5 - rdftW[64-j1]
	}
	dup(rftWkr8[:], wkr[:])
	dup(rftWki8[:], rdftW[32:64])
	initSoaTables()
	initSoaPairTables()
}

// tab8 is the repeated t[k2:k2+4] of the 128-bit tables; k is 2*k2 in
// {0, 8, ..., 56}.
func tab8(t *[64]float32, k int) vec8 {
	k &= 56
	return archsimd.LoadFloat32x8Array((*[8]float32)(t[k : k+8]))
}

// The passes reach their vectors at constant offsets of one window per
// iteration, like the 128-bit kernel (f4w60): f8w120 and f8w200 are the
// eight floats at offset off of such a window.
func f8w120(p *[120]float32, off int) *[8]float32 { return (*[8]float32)(p[off : off+8]) }

func f8w200(p *[200]float32, off int) *[8]float32 { return (*[8]float32)(p[off : off+8]) }

// fftSplitPairSimd is fftSplitSimd for the frames a and b.
func fftSplitPairSimd(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch) {
	t := &s.t
	bitrv2Cft1stPair(a, b, nil, t)
	cftmdlPair(t)
	cftfsubTailPair(t)
	rftfsubSplitPair(t, re0, im0, re1, im1)
	archsimd.ClearAVXUpperBits()
	// Bin 64 is a[0] - a[1] of the frame the last pass read.
	re0[64] = t[0] - t[1]
	im0[64] = 0
	re1[64] = t[4] - t[5]
	im1[64] = 0
}

// inverseFftSplitPairSimd is inverseFftSplitSimd for two sets of bins.
func inverseFftSplitPairSimd(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch) {
	t, u := &s.t, &s.u
	rftbsubSplitPair(re0, im0, re1, im1, t)
	bitrv2Cft1stPair(nil, nil, t, u)
	cftmdlPair(u)
	cftbsubTailScatter(u, a, b)
	archsimd.ClearAVXUpperBits()
}

// bitrv2Cft1stPair is bitrv2Cft1stSimd into the interleaved t, either from
// the interleaved s or (s == nil) gathering the frames a and b.
func bitrv2Cft1stPair(a, b *[128]float32, s, t *[256]float32) {
	swap := ld8(&swapSign8)
	thi := (*[128]float32)(t[128:])
	// r is the residue and q its two bits reversed.
	blocks := func(r, q int) {
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
	blocks(0, 0)
	blocks(1, 2)
	blocks(2, 1)
	blocks(3, 3)
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
	first := func(j int) {
		w := (*[120]float32)(t[j : j+120])
		q0, q8, q16, q24 := f8w120(w, 0), f8w120(w, 16), f8w120(w, 32), f8w120(w, 48)
		q32, q40, q48, q56 := f8w120(w, 64), f8w120(w, 80), f8w120(w, 96), f8w120(w, 112)
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
	first(0)
	first(8)
	const k = 8
	wk2rv := tab8(&wk2r8, k)
	wk2iv := tab8(&wk2i8, k)
	wk1iv := tab8(&wk1i8, k)
	wk3rv := tab8(&wk3r8, k)
	wk3iv := tab8(&wk3i8, k)
	wk1rv = tab8(&wk1r8, k)
	thi := (*[128]float32)(t[128:])
	second := func(j int) {
		w := (*[120]float32)(thi[j : j+120])
		q0, q8, q16, q24 := f8w120(w, 0), f8w120(w, 16), f8w120(w, 32), f8w120(w, 48)
		q32, q40, q48, q56 := f8w120(w, 64), f8w120(w, 80), f8w120(w, 96), f8w120(w, 112)
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
	second(0)
	second(8)
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
	step := func(j int) {
		w := (*[200]float32)(t[j : j+200])
		q0, q1, q2, q3 := f8w200(w, 0), f8w200(w, 64), f8w200(w, 128), f8w200(w, 192)
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
	step(0)
	step(8)
	step(16)
	step(24)
	step(32)
	step(40)
	step(48)
	step(56)
}

// cftbsubTail8 is one iteration of cftbsubTailSimd: r0..r3 belong at the
// iteration's offset in a[0:32], a[32:64], a[64:96] and a[96:128].
func cftbsubTail8(q0, q1, q2, q3 *[8]float32, sgn vec8) (r0, r1, r2, r3 vec8) {
	n0, n1 := ld8(q0).Mul(sgn), ld8(q1).Mul(sgn)
	p2, p3 := ld8(q2), ld8(q3)
	x0 := n0.Add(n1)
	x1 := n0.Sub(n1)
	x2s := p2.Add(p3).Mul(sgn)
	x3w := rev64g(p2.Sub(p3))
	return x0.Add(x2s), x1.Sub(x3w), x0.Sub(x2s), x1.Add(x3w)
}

// cftbsubTailScatter is cftbsubTailSimd from the interleaved frames into a
// and b, two iterations (r and r+1) per round.
func cftbsubTailScatter(t *[256]float32, a, b *[128]float32) {
	sgn := ld8(&signIm8)
	round := func(r int) {
		w := (*[208]float32)(t[8*r : 8*r+208])
		r0, r1, r2, r3 := cftbsubTail8((*[8]float32)(w[0:8]), (*[8]float32)(w[64:72]), (*[8]float32)(w[128:136]), (*[8]float32)(w[192:200]), sgn)
		s0, s1, s2, s3 := cftbsubTail8((*[8]float32)(w[8:16]), (*[8]float32)(w[72:80]), (*[8]float32)(w[136:144]), (*[8]float32)(w[200:208]), sgn)
		wa := (*[104]float32)(a[4*r : 4*r+104])
		wb := (*[104]float32)(b[4*r : 4*r+104])
		frameA(r0, s0).StoreArray((*[8]float32)(wa[0:8]))
		frameB(r0, s0).StoreArray((*[8]float32)(wb[0:8]))
		frameA(r1, s1).StoreArray((*[8]float32)(wa[32:40]))
		frameB(r1, s1).StoreArray((*[8]float32)(wb[32:40]))
		frameA(r2, s2).StoreArray((*[8]float32)(wa[64:72]))
		frameB(r2, s2).StoreArray((*[8]float32)(wb[64:72]))
		frameA(r3, s3).StoreArray((*[8]float32)(wa[96:104]))
		frameB(r3, s3).StoreArray((*[8]float32)(wb[96:104]))
	}
	round(0)
	round(2)
	round(4)
	round(6)
}

// rftfArith8 and rftbArith8 are the float operations of one rftfsub
// (rftbsub) iteration on the bins p and their partners k.
func rftfArith8(p0, p1, k0, k1, wkr, wki vec8) (vec8, vec8, vec8, vec8) {
	xr := p0.Sub(k0)
	xi := p1.Add(k1)
	yr := wkr.Mul(xr).Sub(wki.Mul(xi))
	yi := wkr.Mul(xi).Add(wki.Mul(xr))
	return p0.Sub(yr), p1.Sub(yi), k0.Add(yr), k1.Sub(yi)
}

func rftbArith8(p0, p1, k0, k1, wkr, wki vec8) (vec8, vec8, vec8, vec8) {
	xr := p0.Sub(k0)
	xi := p1.Add(k1)
	yr := wkr.Mul(xr).Add(wki.Mul(xi))
	yi := wkr.Mul(xi).Sub(wki.Mul(xr))
	return p0.Sub(yr), yi.Sub(p1), k0.Add(yr), yi.Sub(k1)
}

// b8 is bins 8q..8q+7 of a 65-bin array (q in 0..3), h8 bins 32+8u..39+8u.
func b8(x *[65]float32, q int) *[8]float32 { return (*[8]float32)(x[8*q : 8*q+8]) }

func h8(x *[65]float32, u int) *[8]float32 {
	return (*[8]float32)((*[32]float32)(x[32:64])[8*u : 8*u+8])
}

// rftfsubSplitPair is rftfsubSplit on the interleaved frames, two
// iterations per round, except for bin 64 (see fftSplitPairSimd). Bin 0,
// a[0] + a[1], comes from the first vector plus itself with its pairs
// swapped.
func rftfsubSplitPair(t *[256]float32, re0, im0, re1, im1 *[65]float32) {
	thi := (*[128]float32)(t[128:])
	v0 := ld8((*[8]float32)(t[0:8]))
	dc := v0.Add(rev64g(v0))
	v16 := ld8((*[8]float32)(thi[0:8]))
	re32 := v16.ConcatPermuteScalarsGrouped(0, 0, 4, 4, v16)
	im32 := v16.ConcatPermuteScalarsGrouped(1, 1, 5, 5, v16)
	var zero, prevR, prevI, r0, r1, hR, hI vec8
	round := func(q int) {
		u := 3 - q
		lo := (*[32]float32)(t[32*q : 32*q+32])
		hi := (*[32]float32)(thi[32*u : 32*u+32])
		wr := (*[16]float32)(rftWkr8[16*q : 16*q+16])
		wi := (*[16]float32)(rftWki8[16*q : 16*q+16])

		// Iteration m = 2q.
		a0, a1 := ld8((*[8]float32)(lo[0:8])), ld8((*[8]float32)(lo[8:16]))
		g0, g1 := ld8((*[8]float32)(hi[16:24])), ld8((*[8]float32)(hi[24:32]))
		rhR, rhI := revEveng(g1, g0), revOddg(g1, g0)
		p0, p1, k0, k1 := rftfArith8(eveng(a0, a1), oddg(a0, a1), ext3g(prevR, rhR), ext3g(prevI, rhI),
			ld8((*[8]float32)(wr[0:8])), ld8((*[8]float32)(wi[0:8])))
		prevR, prevI = rhR, rhI
		k0, k1 = rev4g(k0), rev4g(k1)
		if q == 0 {
			p0 = lane0g(dc, p0)
			p1 = lane0g(zero, p1)
		} else {
			// Bins 64-8q..71-8q.
			xr, xi := ext3g(k0, r0), ext3g(k1, r1)
			frameA(xr, hR).StoreArray(h8(re0, u+1))
			frameB(xr, hR).StoreArray(h8(re1, u+1))
			frameA(xi, hI).StoreArray(h8(im0, u+1))
			frameB(xi, hI).StoreArray(h8(im1, u+1))
		}
		r0, r1 = k0, k1

		// Iteration m = 2q+1.
		a0, a1 = ld8((*[8]float32)(lo[16:24])), ld8((*[8]float32)(lo[24:32]))
		g0, g1 = ld8((*[8]float32)(hi[0:8])), ld8((*[8]float32)(hi[8:16]))
		rhR, rhI = revEveng(g1, g0), revOddg(g1, g0)
		s0, s1, k0, k1 := rftfArith8(eveng(a0, a1), oddg(a0, a1), ext3g(prevR, rhR), ext3g(prevI, rhI),
			ld8((*[8]float32)(wr[8:16])), ld8((*[8]float32)(wi[8:16])))
		prevR, prevI = rhR, rhI
		k0, k1 = rev4g(k0), rev4g(k1)
		frameA(p0, s0).StoreArray(b8(re0, q))
		frameB(p0, s0).StoreArray(b8(re1, q))
		frameA(p1, s1).StoreArray(b8(im0, q))
		frameB(p1, s1).StoreArray(b8(im1, q))
		hR, hI = ext3g(k0, r0), ext3g(k1, r1)
		r0, r1 = k0, k1
	}
	round(0)
	round(1)
	round(2)
	round(3)
	// Bins 32..39: bin 32 as the frame has it, then the last results.
	xr, xi := ext3g(re32, r0), ext3g(im32, r1)
	frameA(xr, hR).StoreArray(h8(re0, 0))
	frameB(xr, hR).StoreArray(h8(re1, 0))
	frameA(xi, hI).StoreArray(h8(im0, 0))
	frameB(xi, hI).StoreArray(h8(im1, 0))
}

// rftbsubSplitPair is rftbsubSplit for two sets of bins, into the
// interleaved t. The scalar values of bins 0, 32 and 64 are computed before
// the first 256-bit operation.
func rftbsubSplitPair(re0, im0, re1, im1 *[65]float32, t *[256]float32) {
	a1 := float32(0.5 * (re0[0] - re0[64]))
	a0 := re0[0] - a1
	a1 = -a1
	b1 := float32(0.5 * (re1[0] - re1[64]))
	b0 := re1[0] - b1
	b1 = -b1
	var z vec
	dcA := z.SetElem(0, a0).SetElem(1, a1)
	dcB := z.SetElem(0, b0).SetElem(1, b1)
	xA := z.SetElem(0, re0[32]).SetElem(1, -im0[32])
	xB := z.SetElem(0, re1[32]).SetElem(1, -im1[32])
	var z8 vec8
	dc := z8.SetLo(dcA).SetHi(dcB)
	x32 := z8.SetLo(xA).SetHi(xB)

	thi := (*[128]float32)(t[128:])
	var prevR, prevI, pzh vec8
	step := func(m int) {
		n := 7 - m
		p0 := ldPair(lo4(re0, m), lo4(re1, m))
		p1 := ldPair(lo4(im0, m), lo4(im1, m))
		rhR := rev4g(ldPair(hi4(re0, n), hi4(re1, n)))
		rhI := rev4g(ldPair(hi4(im0, n), hi4(im1, n)))
		wkr := ld8((*[8]float32)(rftWkr8[8*m : 8*m+8]))
		wki := ld8((*[8]float32)(rftWki8[8*m : 8*m+8]))
		p0, p1, k0, k1 := rftbArith8(p0, p1, ext3g(prevR, rhR), ext3g(prevI, rhI), wkr, wki)
		prevR, prevI = rhR, rhI
		z0, z1 := zipLog(p0, p1), zipHig(p0, p1)
		if m == 0 {
			z0 = loHig(dc, z0)
		}
		lo := (*[16]float32)(t[16*m : 16*m+16])
		z0.StoreArray((*[8]float32)(lo[0:8]))
		z1.StoreArray((*[8]float32)(lo[8:16]))
		zl, zh := zipLog(k0, k1), zipHig(k0, k1)
		hi := (*[16]float32)(thi[16*n : 16*n+16])
		loHig(zh, zl).StoreArray((*[8]float32)(hi[8:16]))
		if m > 0 {
			// The mask changes nothing for n < 7.
			j := (16*n + 16) & 112
			loHig(zl, pzh).StoreArray((*[8]float32)(thi[j : j+8]))
		}
		pzh = zh
	}
	step(0)
	step(1)
	step(2)
	step(3)
	step(4)
	step(5)
	step(6)
	step(7)
	loHig(x32, pzh).StoreArray((*[8]float32)(thi[0:8]))
}
