//go:build goexperiment.simd && arm64 && !purego

package ooura

// The split transforms of the NEON lane with the real and the imaginary
// parts in separate vectors.
//
// ooura_fft_neon.cc keeps a complex value in two adjacent lanes, so a
// rotation w*x is wr*x + wi*rev64(x) and x1 +- i*x3 goes through rev64 and
// the {-1, 1} sign table as well; with the lanes regrouped between the
// passes, close to half of the vector operations of the passes of
// fft_simd.go are shuffles. A core with two vector pipes executes the
// transform at the pace of its vector operations, shuffles included.
//
// Here a vector holds the real (or the imaginary) parts of four complex
// values. Every float operation is the one of the C code for that element:
//
//   - a rotation is (wrr*xr + wri*xi, wir*xi + wii*xr) with the four table
//     values the two lanes of the C vectors hold;
//   - x + s*y for s in {1, -1} is x + y and x - y: s*y is y or its negation,
//     exactly, and adding a negation is subtracting.
//
// The frame is 64 complex values. After the bit reversal the value of
// position p = 16*B + 4*c1 + c0 comes from index 16*rev2(c0) + 4*rev2(c1) +
// rev2(B), and the three radix-4 passes combine the four values that differ
// in c0 (cft1st), in c1 (cftmdl) and in B (the last pass of cftfsub and
// cftbsub). The first pass takes its four vectors from four contiguous
// loads (lanes over B), a 4x4 transposition turns its results into lanes
// over c0, and the other two passes and rftfsub/rftbsub work on lanes over
// c0, the order of the bins.
//
// TestSoaMatchesRef checks the transforms bit-exactly against the lane
// reference.

// soaS1[m] holds the twiddle factors of the cft1st butterflies of the load
// group m, one lane per butterfly, as twelve vectors: for wk1, wk2 and wk3
// the factor of xr and of xi in the real part and the factor of xi and of
// xr in the imaginary part.
var soaS1 [4][12][4]float32

// soaS2[b] holds the same twelve factors of the cftmdl butterflies of block
// 2+b, in every lane.
var soaS2 [2][12][4]float32

// soaWk1[i] is cftmdlWk1r[i] in every lane (block 1 of cftmdl).
var soaWk1 [4][4]float32

func rev2(x int) int { return (x&1)<<1 | x>>1 }

// initPairTables runs once the negative zeros of the 128-bit tables are in
// place.
func initPairTables() {
	tabs := [3][2]*[32]float32{{&rdftWk1r, &rdftWk1i}, {&rdftWk2r, &rdftWk2i}, {&rdftWk3r, &rdftWk3i}}
	fill := func(dst *[12][4]float32, lane, idx int) {
		for w, t := range tabs {
			dst[4*w+0][lane] = t[0][idx]   // xr in the real part
			dst[4*w+1][lane] = t[1][idx]   // xi in the real part
			dst[4*w+2][lane] = t[0][idx+1] // xi in the imaginary part
			dst[4*w+3][lane] = t[1][idx+1] // xr in the imaginary part
		}
	}
	for m := 0; m < 4; m++ {
		for k := 0; k < 4; k++ {
			// Lane k of group m is the butterfly of positions 4*i..4*i+3
			// with i = 4*rev2(k) + rev2(m): cft1st iteration i/2 (table
			// index 4*(i/2)), its first or second complex value.
			i := 4*rev2(k) + rev2(m)
			fill(&soaS1[m], k, 4*(i/2)+2*(i&1))
		}
	}
	for b := 0; b < 2; b++ {
		for lane := 0; lane < 4; lane++ {
			fill(&soaS2[b], lane, 4+2*b)
		}
	}
	for i, w := range cftmdlWk1r {
		soaWk1[i] = [4]float32{w, w, w, w}
	}
}

// PairScratch is the work area of FftSplitPair and InverseFftSplitPair; this
// build runs the two transforms one after the other.
type PairScratch struct {
	s Scratch
}

func fftSplitPair(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch) {
	fftSplit(a, re0, im0, s.single())
	fftSplit(b, re1, im1, s.single())
}

func inverseFftSplitPair(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch) {
	inverseFftSplit(re0, im0, a, s.single())
	inverseFftSplit(re1, im1, b, s.single())
}

// fftSplitSoa is fftSplitSimd: the transform of a into the bins re and im.
// It leaves a in an intermediate state; t is work space.
func fftSplitSoa(a *[128]float32, re, im *[65]float32, t *[128]float32) {
	soaCft1stPacked(a, t)
	soaCftmdl(t, a)
	soaCftfsubTail(a, t)
	soaRftfsub(t, re, im)
}

// inverseFftSplitSoa is inverseFftSplitSimd: the inverse transform of the
// bins re and im into a; t is work space.
func inverseFftSplitSoa(re, im *[65]float32, a *[128]float32, t *[128]float32) {
	soaRftbsub(re, im, t)
	soaCft1stSplit(t, a)
	soaCftmdl(a, t)
	soaCftbsubTail(t, a)
}

// A frame in split form holds the real parts of its 64 complex values in
// its first half and the imaginary parts in its second half.

func soaHalves(t *[128]float32) (r, i *[64]float32) {
	return (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
}

// q4 is four floats at the constant offset off of a window.
func q4w52(p *[52]float32, off int) *[4]float32 { return (*[4]float32)(p[off : off+4]) }

func q4w16(p *[16]float32, off int) *[4]float32 { return (*[4]float32)(p[off : off+4]) }

func tab12(t *[12][4]float32, i int) vec { return ld(&t[i]) }

// trn1 is {x0, y0, x2, y2}; trn2 is {x1, y1, x3, y3}.
func trn1(x, y vec) vec { return x.ToBits().InterleaveEven(y.ToBits()).BitsToFloat32() }

func trn2(x, y vec) vec { return x.ToBits().InterleaveOdd(y.ToBits()).BitsToFloat32() }

// soaButterfly is the part every radix-4 butterfly of the transform starts
// with, on the values a, b, c and d: x0 = a + b, x1 = a - b, x2 = c + d and
// x3 = c - d, then s = x0 + x2, t = x0 - x2, y = x1 + i*x3 and
// z = x1 - i*x3.
func soaButterfly(ar, ai, br, bi, cr, ci, dr, di vec) (sr, si, tr, ti, yr, yi, zr, zi vec) {
	x0r, x0i := ar.Add(br), ai.Add(bi)
	x1r, x1i := ar.Sub(br), ai.Sub(bi)
	x2r, x2i := cr.Add(dr), ci.Add(di)
	x3r, x3i := cr.Sub(dr), ci.Sub(di)
	return x0r.Add(x2r), x0i.Add(x2i), x0r.Sub(x2r), x0i.Sub(x2i),
		x1r.Sub(x3i), x1i.Add(x3r), x1r.Add(x3i), x1i.Sub(x3r)
}

// soaRot is the rotation of x by the factors w, w+1, w+2 and w+3 of tab.
func soaRot(tab *[12][4]float32, w int, xr, xi vec) (yr, yi vec) {
	return tab12(tab, w).Mul(xr).Add(tab12(tab, w+1).Mul(xi)), tab12(tab, w+2).Mul(xi).Add(tab12(tab, w+3).Mul(xr))
}

// soaTransposeStore stores lane k of r0..r3 as four consecutive floats, for
// k = 0, 2, 1, 3 at w, w+16, w+32 and w+48: lane k of the cft1st results
// o0..o3 of a load group m is positions 16*rev2(k) + 4*rev2(m) + 0..3.
func soaTransposeStore(w *[52]float32, r0, r1, r2, r3 vec) {
	e0, e1, e2, e3 := trn1(r0, r1), trn2(r0, r1), trn1(r2, r3), trn2(r2, r3)
	lo64(e0, e2).StoreArray(q4w52(w, 0))
	hi64(e0, e2).StoreArray(q4w52(w, 16))
	lo64(e1, e3).StoreArray(q4w52(w, 32))
	hi64(e1, e3).StoreArray(q4w52(w, 48))
}

// The cft1st passes are bitrv2_128 and cft1st_128 by load group: the four
// inputs of group m hold, in lane k, the values of index
// 16*rev2(c) + 4*m + k for c = 0..3, which are positions 4*i..4*i+3 with
// i = 4*rev2(k) + rev2(m). The results go to t in split form, at their
// positions.

// soaCft1stPacked is bitrv2_128 and cft1st_128 from the frame a, whose
// complex values are pairs of floats, into t in split form.
func soaCft1stPacked(a, t *[128]float32) {
	outR, outI := soaHalves(t)
	var zero vec
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		// Index 16*rev2(c) + 4*m is float 32*rev2(c) + 8*m.
		w := (*[104]float32)(a[8*m : 8*m+104])
		a0, a1 := ld((*[4]float32)(w[0:4])), ld((*[4]float32)(w[4:8]))
		b0, b1 := ld((*[4]float32)(w[64:68])), ld((*[4]float32)(w[68:72]))
		c0, c1 := ld((*[4]float32)(w[32:36])), ld((*[4]float32)(w[36:40]))
		d0, d1 := ld((*[4]float32)(w[96:100])), ld((*[4]float32)(w[100:104]))
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly(even(a0, a1), odd(a0, a1), even(b0, b1), odd(b0, b1),
			even(c0, c1), odd(c0, c1), even(d0, d1), odd(d0, d1))
		tab := &soaS1[m]
		o2r, o2i := soaRot(tab, 4, tr, ti)
		o1r, o1i := soaRot(tab, 0, yr, yi)
		o3r, o3i := soaRot(tab, 8, zr, zi)
		soaTransposeStore((*[52]float32)(outR[p:p+52]), o0r, o1r, o2r, o3r)
		soaTransposeStore((*[52]float32)(outI[p:p+52]), o0i, o1i, o2i, o3i)
	}
	// p = 4*rev2(m).
	group(0, 0)
	group(1, 8)
	group(2, 4)
	group(3, 12)
}

// soaCft1stSplit is bitrv2_128 and cft1st_128 from s to t, both in split
// form.
func soaCft1stSplit(s, t *[128]float32) {
	inR, inI := soaHalves(s)
	outR, outI := soaHalves(t)
	var zero vec
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		wr := (*[52]float32)(inR[4*m : 4*m+52])
		wi := (*[52]float32)(inI[4*m : 4*m+52])
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly(ld(q4w52(wr, 0)), ld(q4w52(wi, 0)), ld(q4w52(wr, 32)), ld(q4w52(wi, 32)),
			ld(q4w52(wr, 16)), ld(q4w52(wi, 16)), ld(q4w52(wr, 48)), ld(q4w52(wi, 48)))
		tab := &soaS1[m]
		o2r, o2i := soaRot(tab, 4, tr, ti)
		o1r, o1i := soaRot(tab, 0, yr, yi)
		o3r, o3i := soaRot(tab, 8, zr, zi)
		soaTransposeStore((*[52]float32)(outR[p:p+52]), o0r, o1r, o2r, o3r)
		soaTransposeStore((*[52]float32)(outI[p:p+52]), o0i, o1i, o2i, o3i)
	}
	// p = 4*rev2(m).
	group(0, 0)
	group(1, 8)
	group(2, 4)
	group(3, 12)
}

// soaLoad16 loads the four vectors of a block of sixteen values.
func soaLoad16(p *[16]float32) (v0, v1, v2, v3 vec) {
	return ld(q4w16(p, 0)), ld(q4w16(p, 4)), ld(q4w16(p, 8)), ld(q4w16(p, 12))
}

func soaStore16(p *[16]float32, v0, v1, v2, v3 vec) {
	v0.StoreArray(q4w16(p, 0))
	v1.StoreArray(q4w16(p, 4))
	v2.StoreArray(q4w16(p, 8))
	v3.StoreArray(q4w16(p, 12))
}

// soaCftmdl is cftmdl_128 from s to t in split form. Block b is positions
// 16*b..16*b+15, and a butterfly combines the four values that are four
// positions apart: the four vectors of a block.
func soaCftmdl(s, t *[128]float32) {
	inR, inI := soaHalves(s)
	outR, outI := soaHalves(t)

	// Block 0: no rotation.
	{
		ar, br, cr, dr := soaLoad16((*[16]float32)(inR[0:16]))
		ai, bi, ci, di := soaLoad16((*[16]float32)(inI[0:16]))
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(ar, ai, br, bi, cr, ci, dr, di)
		soaStore16((*[16]float32)(outR[0:16]), sr, yr, tr, zr)
		soaStore16((*[16]float32)(outI[0:16]), si, yi, ti, zi)
	}

	// Block 1: t turns by a quarter, y and z by wk1r.
	{
		ar, br, cr, dr := soaLoad16((*[16]float32)(inR[16:32]))
		ai, bi, ci, di := soaLoad16((*[16]float32)(inI[16:32]))
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(ar, ai, br, bi, cr, ci, dr, di)
		g0, g1, g2, g3 := ld(&soaWk1[0]), ld(&soaWk1[1]), ld(&soaWk1[2]), ld(&soaWk1[3])
		soaStore16((*[16]float32)(outR[16:32]), sr, g0.Mul(yr.Sub(yi)), ti.Neg(), g3.Mul(zr.Add(zi)))
		soaStore16((*[16]float32)(outI[16:32]), si, g1.Mul(yr.Add(yi)), tr, g2.Mul(zr.Sub(zi)))
	}

	// Blocks 2 and 3: the rotations of soaS2.
	var zero vec
	rotated := func(b int) {
		_ = zero // a captured vector makes the compiler inline the closure
		tab := &soaS2[b]
		wr := (*[16]float32)(inR[32+16*b : 48+16*b])
		wi := (*[16]float32)(inI[32+16*b : 48+16*b])
		ar, br, cr, dr := soaLoad16(wr)
		ai, bi, ci, di := soaLoad16(wi)
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(ar, ai, br, bi, cr, ci, dr, di)
		o2r, o2i := soaRot(tab, 4, tr, ti)
		o1r, o1i := soaRot(tab, 0, yr, yi)
		o3r, o3i := soaRot(tab, 8, zr, zi)
		soaStore16((*[16]float32)(outR[32+16*b:48+16*b]), sr, o1r, o2r, o3r)
		soaStore16((*[16]float32)(outI[32+16*b:48+16*b]), si, o1i, o2i, o3i)
	}
	rotated(0)
	rotated(1)
}

// soaCftfsubTail is the last pass of cftfsub_128 from s to t in split form:
// a butterfly combines the four values that are sixteen positions apart.
func soaCftfsubTail(s, t *[128]float32) {
	inR, inI := soaHalves(s)
	outR, outI := soaHalves(t)
	var zero vec
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		ir := (*[52]float32)(inR[j : j+52])
		ii := (*[52]float32)(inI[j : j+52])
		or := (*[52]float32)(outR[j : j+52])
		oi := (*[52]float32)(outI[j : j+52])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(ld(q4w52(ir, 0)), ld(q4w52(ii, 0)), ld(q4w52(ir, 16)), ld(q4w52(ii, 16)),
			ld(q4w52(ir, 32)), ld(q4w52(ii, 32)), ld(q4w52(ir, 48)), ld(q4w52(ii, 48)))
		sr.StoreArray(q4w52(or, 0))
		si.StoreArray(q4w52(oi, 0))
		yr.StoreArray(q4w52(or, 16))
		yi.StoreArray(q4w52(oi, 16))
		tr.StoreArray(q4w52(or, 32))
		ti.StoreArray(q4w52(oi, 32))
		zr.StoreArray(q4w52(or, 48))
		zi.StoreArray(q4w52(oi, 48))
	}
	step(0)
	step(4)
	step(8)
	step(12)
}

// soaCftbsubTail is the last pass of cftbsub_128 from s in split form to
// the frame a, whose complex values are pairs of floats. The C code works
// on the conjugates of the first two values and conjugates x0 + x2 again;
// its x1 - i*x3 and x1 + i*x3 take the parts of x3 exchanged.
func soaCftbsubTail(s, a *[128]float32) {
	inR, inI := soaHalves(s)
	var zero vec
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		ir := (*[52]float32)(inR[j : j+52])
		ii := (*[52]float32)(inI[j : j+52])
		ar, ai := ld(q4w52(ir, 0)), ld(q4w52(ii, 0)).Neg()
		br, bi := ld(q4w52(ir, 16)), ld(q4w52(ii, 16)).Neg()
		cr, ci := ld(q4w52(ir, 32)), ld(q4w52(ii, 32))
		dr, di := ld(q4w52(ir, 48)), ld(q4w52(ii, 48))
		x0r, x0i := ar.Add(br), ai.Add(bi)
		x1r, x1i := ar.Sub(br), ai.Sub(bi)
		x2r, x2i := cr.Add(dr), ci.Add(di).Neg()
		x3r, x3i := cr.Sub(dr), ci.Sub(di)
		o := (*[104]float32)(a[2*j : 2*j+104])
		soaStorePacked((*[8]float32)(o[0:8]), x0r.Add(x2r), x0i.Add(x2i))
		soaStorePacked((*[8]float32)(o[32:40]), x1r.Sub(x3i), x1i.Sub(x3r))
		soaStorePacked((*[8]float32)(o[64:72]), x0r.Sub(x2r), x0i.Sub(x2i))
		soaStorePacked((*[8]float32)(o[96:104]), x1r.Add(x3i), x1i.Add(x3r))
	}
	step(0)
	step(4)
	step(8)
	step(12)
}

// soaStorePacked stores four complex values as pairs of floats.
func soaStorePacked(p *[8]float32, r, i vec) {
	zipLo(r, i).StoreArray((*[4]float32)(p[0:4]))
	zipHi(r, i).StoreArray((*[4]float32)(p[4:8]))
}

// s4 is positions 4*m..4*m+3 of the first 32 of a half, s4h positions
// 32+4*n..35+4*n.
func s4(x *[64]float32, m int) *[4]float32 {
	return (*[4]float32)((*[32]float32)(x[0:32])[4*m : 4*m+4])
}

func s4h(x *[64]float32, n int) *[4]float32 {
	return (*[4]float32)((*[32]float32)(x[32:64])[4*n : 4*n+4])
}

// soaRftfsub is rftfsubSplit on a frame in split form: rftfsub_128 and the
// final butterfly of Fft, with the result written as bins. Position k of
// the frame is bin k, and the real part of bin 64 is the imaginary part of
// position 0.
func soaRftfsub(t *[128]float32, re, im *[65]float32) {
	tr, ti := soaHalves(t)
	a0, a1 := tr[0], ti[0]
	re32, im32 := tr[32], ti[32]
	// prevR and prevI are the reversed bins 64-4m..67-4m, whose last lane is
	// bin 64-4m; r0 and r1 the results for bins 65-4m..68-4m.
	var prevR, prevI, r0, r1 vec
	step := func(m int) {
		n := 7 - m
		p0, p1 := ld(s4(tr, m)), ld(s4(ti, m))
		rhR, rhI := rev4(ld(s4h(tr, n))), rev4(ld(s4h(ti, n)))
		k0, k1 := ext3(prevR, rhR), ext3(prevI, rhI)
		prevR, prevI = rhR, rhI
		wkr, wki := rftSplitTabs(m)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		p0, p1, k0, k1 = p0.Sub(yr), p1.Sub(yi), k0.Add(yr), k1.Sub(yi)
		if m == 0 {
			p0 = p0.SetElem(0, a0+a1)
			p1 = p1.SetElem(0, 0)
		}
		p0.StoreArray(lo4(re, m))
		p1.StoreArray(lo4(im, m))
		k0, k1 = rev4(k0), rev4(k1) // bins 61-4m..64-4m
		if m > 0 {
			ext3(k0, r0).StoreArray(hi4(re, n+1))
			ext3(k1, r1).StoreArray(hi4(im, n+1))
		}
		r0, r1 = k0, k1
	}
	step(0)
	step(1)
	step(2)
	step(3)
	step(4)
	step(5)
	step(6)
	step(7)
	// Bins 32..35: bin 32 as the frame has it, then the last results.
	ext3(r0, r0).SetElem(0, re32).StoreArray(hi4(re, 0))
	ext3(r1, r1).SetElem(0, im32).StoreArray(hi4(im, 0))
	re[64] = a0 - a1
	im[64] = 0
}

// soaRftbsub is rftbsubSplit into a frame in split form: the first
// butterfly of InverseFft and rftbsub_128 on the frame that packs the bins
// re and im.
func soaRftbsub(re, im *[65]float32, t *[128]float32) {
	tr, ti := soaHalves(t)
	a1 := float32(0.5 * (re[0] - re[64]))
	a0 := re[0] - a1
	a1 = -a1
	re32, im32 := re[32], -im[32]
	// prevR and prevI are as in soaRftfsub, r0 and r1 the results for bins
	// 65-4m..68-4m.
	var prevR, prevI, r0, r1 vec
	step := func(m int) {
		n := 7 - m
		p0, p1 := ld(lo4(re, m)), ld(lo4(im, m))
		rhR, rhI := rev4(ld(hi4(re, n))), rev4(ld(hi4(im, n)))
		k0, k1 := ext3(prevR, rhR), ext3(prevI, rhI)
		prevR, prevI = rhR, rhI
		wkr, wki := rftSplitTabs(m)
		xr := p0.Sub(k0)
		xi := p1.Add(k1)
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		p0, p1, k0, k1 = p0.Sub(yr), yi.Sub(p1), k0.Add(yr), yi.Sub(k1)
		if m == 0 {
			p0 = p0.SetElem(0, a0)
			p1 = p1.SetElem(0, a1)
		}
		p0.StoreArray(s4(tr, m))
		p1.StoreArray(s4(ti, m))
		k0, k1 = rev4(k0), rev4(k1) // bins 61-4m..64-4m
		if m > 0 {
			ext3(k0, r0).StoreArray(s4h(tr, (n+1)&7))
			ext3(k1, r1).StoreArray(s4h(ti, (n+1)&7))
		}
		r0, r1 = k0, k1
	}
	step(0)
	step(1)
	step(2)
	step(3)
	step(4)
	step(5)
	step(6)
	step(7)
	// Positions 32..35: bin 32 with its imaginary part negated, then the
	// last results.
	ext3(r0, r0).SetElem(0, re32).StoreArray(s4h(tr, 0))
	ext3(r1, r1).SetElem(0, im32).StoreArray(s4h(ti, 0))
}
