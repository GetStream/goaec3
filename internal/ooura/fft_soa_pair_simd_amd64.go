//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// Two split-form transforms in one 256-bit pass (AVX2): the passes of
// fft_soa_simd.go on vectors whose low half belongs to frame a and whose
// high half belongs to frame b. Every operation of those passes is local to
// one 128-bit vector, the grouped shuffles permute each half on its own and
// the tables repeat their four floats in both halves, so each half is one
// transform with the IEEE operations of the single one.
//
// The work buffer holds the real parts of the two frames in its first half
// and the imaginary parts in its second, each interleaved four floats at a
// time: positions 4v..4v+3 of frame a are r[8v:8v+4], those of frame b
// r[8v+4:8v+8]. The first pass of each direction gathers its two inputs
// into that layout and the last one separates the results into one 256-bit
// store per frame (VPERM2F128), as the passes of fft_pair_simd_amd64.go do.
//
// TestSoaPairMatchesSingle checks the pair bit-exactly against two single
// transforms.

// The tables of fft_soa_simd.go with every vector repeated.
var (
	soaS1x8  [4][12][8]float32
	soaS2x8  [2][12][8]float32
	soaWk1x8 [4][8]float32
	signAll8 = [8]uint32{0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000}
)

func initSoaPairTables() {
	dup := func(v [4]float32) [8]float32 {
		return [8]float32{v[0], v[1], v[2], v[3], v[0], v[1], v[2], v[3]}
	}
	for m := range soaS1 {
		for i, v := range soaS1[m] {
			soaS1x8[m][i] = dup(v)
		}
	}
	for b := range soaS2 {
		for i, v := range soaS2[b] {
			soaS2x8[b][i] = dup(v)
		}
	}
	for i, v := range soaWk1 {
		soaWk1x8[i] = dup(v)
	}
}

// fftSplitPairSoa is fftSplitSoa for the frames a and b.
func fftSplitPairSoa(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch) {
	t := &s.t
	soaCft1stPackedPair(a, b, t)
	soaCftmdlPair(t)
	soaCftfsubTailPair(t)
	soaRftfsubPair(t, re0, im0, re1, im1)
	archsimd.ClearAVXUpperBits()
	// Bin 64 is a[0] - a[1] of the frame the last pass read: the real and
	// the imaginary part of position 0.
	re0[64] = t[0] - t[128]
	im0[64] = 0
	re1[64] = t[4] - t[132]
	im1[64] = 0
}

// inverseFftSplitPairSoa is inverseFftSplitSoa for two sets of bins.
func inverseFftSplitPairSoa(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch) {
	t, u := &s.t, &s.u
	soaRftbsubPair(re0, im0, re1, im1, t)
	soaCft1stSplitPair(t, u)
	soaCftmdlPair(u)
	soaCftbsubTailPair(u, a, b)
	archsimd.ClearAVXUpperBits()
}

func soaHalves8(t *[256]float32) (r, i *[128]float32) {
	return (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
}

// q8 is eight floats at the constant offset off of a window.
func q8w104(p *[104]float32, off int) *[8]float32 { return (*[8]float32)(p[off : off+8]) }

func q8w32(p *[32]float32, off int) *[8]float32 { return (*[8]float32)(p[off : off+8]) }

func tab12g(t *[12][8]float32, i int) vec8 { return ld8(&t[i]) }

func neg8(x vec8) vec8 {
	return x.ToBits().Xor(archsimd.LoadUint32x8Array(&signAll8)).BitsToFloat32()
}

// soaButterfly8 is soaButterfly on both halves.
func soaButterfly8(ar, ai, br, bi, cr, ci, dr, di vec8) (sr, si, tr, ti, yr, yi, zr, zi vec8) {
	x0r, x0i := ar.Add(br), ai.Add(bi)
	x1r, x1i := ar.Sub(br), ai.Sub(bi)
	x2r, x2i := cr.Add(dr), ci.Add(di)
	x3r, x3i := cr.Sub(dr), ci.Sub(di)
	return x0r.Add(x2r), x0i.Add(x2i), x0r.Sub(x2r), x0i.Sub(x2i),
		x1r.Sub(x3i), x1i.Add(x3r), x1r.Add(x3i), x1i.Sub(x3r)
}

// soaRot8 is soaRot on both halves.
func soaRot8(tab *[12][8]float32, w int, xr, xi vec8) (yr, yi vec8) {
	return tab12g(tab, w).Mul(xr).Add(tab12g(tab, w+1).Mul(xi)), tab12g(tab, w+2).Mul(xi).Add(tab12g(tab, w+3).Mul(xr))
}

// soaTransposeStore8 is soaTransposeStore on both halves.
func soaTransposeStore8(w *[104]float32, r0, r1, r2, r3 vec8) {
	e0, e1, e2, e3 := lo64g(r0, r1), hi64g(r0, r1), lo64g(r2, r3), hi64g(r2, r3)
	eveng(e0, e2).StoreArray(q8w104(w, 0))
	eveng(e1, e3).StoreArray(q8w104(w, 32))
	oddg(e0, e2).StoreArray(q8w104(w, 64))
	oddg(e1, e3).StoreArray(q8w104(w, 96))
}

// soaCft1stPackedPair is soaCft1stPacked from the frames a and b into t.
func soaCft1stPackedPair(a, b *[128]float32, t *[256]float32) {
	outR, outI := soaHalves8(t)
	var zero vec8
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		wa := (*[104]float32)(a[8*m : 8*m+104])
		wb := (*[104]float32)(b[8*m : 8*m+104])
		a0, a1 := ldPair((*[4]float32)(wa[0:4]), (*[4]float32)(wb[0:4])), ldPair((*[4]float32)(wa[4:8]), (*[4]float32)(wb[4:8]))
		b0, b1 := ldPair((*[4]float32)(wa[64:68]), (*[4]float32)(wb[64:68])), ldPair((*[4]float32)(wa[68:72]), (*[4]float32)(wb[68:72]))
		c0, c1 := ldPair((*[4]float32)(wa[32:36]), (*[4]float32)(wb[32:36])), ldPair((*[4]float32)(wa[36:40]), (*[4]float32)(wb[36:40]))
		d0, d1 := ldPair((*[4]float32)(wa[96:100]), (*[4]float32)(wb[96:100])), ldPair((*[4]float32)(wa[100:104]), (*[4]float32)(wb[100:104]))
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly8(eveng(a0, a1), oddg(a0, a1), eveng(b0, b1), oddg(b0, b1),
			eveng(c0, c1), oddg(c0, c1), eveng(d0, d1), oddg(d0, d1))
		tab := &soaS1x8[m]
		o2r, o2i := soaRot8(tab, 4, tr, ti)
		o1r, o1i := soaRot8(tab, 0, yr, yi)
		o3r, o3i := soaRot8(tab, 8, zr, zi)
		soaTransposeStore8((*[104]float32)(outR[p:p+104]), o0r, o1r, o2r, o3r)
		soaTransposeStore8((*[104]float32)(outI[p:p+104]), o0i, o1i, o2i, o3i)
	}
	// p = 8*rev2(m).
	group(0, 0)
	group(1, 16)
	group(2, 8)
	group(3, 24)
}

// soaCft1stSplitPair is soaCft1stSplit from s to t.
func soaCft1stSplitPair(s, t *[256]float32) {
	inR, inI := soaHalves8(s)
	outR, outI := soaHalves8(t)
	var zero vec8
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		wr := (*[104]float32)(inR[8*m : 8*m+104])
		wi := (*[104]float32)(inI[8*m : 8*m+104])
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly8(ld8(q8w104(wr, 0)), ld8(q8w104(wi, 0)), ld8(q8w104(wr, 64)), ld8(q8w104(wi, 64)),
			ld8(q8w104(wr, 32)), ld8(q8w104(wi, 32)), ld8(q8w104(wr, 96)), ld8(q8w104(wi, 96)))
		tab := &soaS1x8[m]
		o2r, o2i := soaRot8(tab, 4, tr, ti)
		o1r, o1i := soaRot8(tab, 0, yr, yi)
		o3r, o3i := soaRot8(tab, 8, zr, zi)
		soaTransposeStore8((*[104]float32)(outR[p:p+104]), o0r, o1r, o2r, o3r)
		soaTransposeStore8((*[104]float32)(outI[p:p+104]), o0i, o1i, o2i, o3i)
	}
	// p = 8*rev2(m).
	group(0, 0)
	group(1, 16)
	group(2, 8)
	group(3, 24)
}

func soaLoad32(p *[32]float32) (v0, v1, v2, v3 vec8) {
	return ld8(q8w32(p, 0)), ld8(q8w32(p, 8)), ld8(q8w32(p, 16)), ld8(q8w32(p, 24))
}

func soaStore32(p *[32]float32, v0, v1, v2, v3 vec8) {
	v0.StoreArray(q8w32(p, 0))
	v1.StoreArray(q8w32(p, 8))
	v2.StoreArray(q8w32(p, 16))
	v3.StoreArray(q8w32(p, 24))
}

// soaCftmdlPair is soaCftmdl in place: a block loads its eight vectors
// before it stores into them.
func soaCftmdlPair(t *[256]float32) {
	r, i := soaHalves8(t)

	// Block 0: no rotation.
	{
		pr, pi := (*[32]float32)(r[0:32]), (*[32]float32)(i[0:32])
		ar, br, cr, dr := soaLoad32(pr)
		ai, bi, ci, di := soaLoad32(pi)
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(ar, ai, br, bi, cr, ci, dr, di)
		soaStore32(pr, sr, yr, tr, zr)
		soaStore32(pi, si, yi, ti, zi)
	}

	// Block 1: t turns by a quarter, y and z by wk1r.
	{
		pr, pi := (*[32]float32)(r[32:64]), (*[32]float32)(i[32:64])
		ar, br, cr, dr := soaLoad32(pr)
		ai, bi, ci, di := soaLoad32(pi)
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(ar, ai, br, bi, cr, ci, dr, di)
		g0, g1, g2, g3 := ld8(&soaWk1x8[0]), ld8(&soaWk1x8[1]), ld8(&soaWk1x8[2]), ld8(&soaWk1x8[3])
		soaStore32(pr, sr, g0.Mul(yr.Sub(yi)), neg8(ti), g3.Mul(zr.Add(zi)))
		soaStore32(pi, si, g1.Mul(yr.Add(yi)), tr, g2.Mul(zr.Sub(zi)))
	}

	// Blocks 2 and 3: the rotations of soaS2.
	var zero vec8
	rotated := func(b int) {
		_ = zero // a captured vector makes the compiler inline the closure
		tab := &soaS2x8[b]
		pr := (*[32]float32)(r[64+32*b : 96+32*b])
		pi := (*[32]float32)(i[64+32*b : 96+32*b])
		ar, br, cr, dr := soaLoad32(pr)
		ai, bi, ci, di := soaLoad32(pi)
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(ar, ai, br, bi, cr, ci, dr, di)
		o2r, o2i := soaRot8(tab, 4, tr, ti)
		o1r, o1i := soaRot8(tab, 0, yr, yi)
		o3r, o3i := soaRot8(tab, 8, zr, zi)
		soaStore32(pr, sr, o1r, o2r, o3r)
		soaStore32(pi, si, o1i, o2i, o3i)
	}
	rotated(0)
	rotated(1)
}

// soaCftfsubTailPair is soaCftfsubTail in place.
func soaCftfsubTailPair(t *[256]float32) {
	r, i := soaHalves8(t)
	var zero vec8
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr := (*[104]float32)(r[j : j+104])
		pi := (*[104]float32)(i[j : j+104])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(ld8(q8w104(pr, 0)), ld8(q8w104(pi, 0)), ld8(q8w104(pr, 32)), ld8(q8w104(pi, 32)),
			ld8(q8w104(pr, 64)), ld8(q8w104(pi, 64)), ld8(q8w104(pr, 96)), ld8(q8w104(pi, 96)))
		sr.StoreArray(q8w104(pr, 0))
		si.StoreArray(q8w104(pi, 0))
		yr.StoreArray(q8w104(pr, 32))
		yi.StoreArray(q8w104(pi, 32))
		tr.StoreArray(q8w104(pr, 64))
		ti.StoreArray(q8w104(pi, 64))
		zr.StoreArray(q8w104(pr, 96))
		zi.StoreArray(q8w104(pi, 96))
	}
	step(0)
	step(8)
	step(16)
	step(24)
}

// soaStorePackedPair stores four complex values of each frame as pairs of
// floats.
func soaStorePackedPair(pa, pb *[8]float32, r, i vec8) {
	zl, zh := zipLog(r, i), zipHig(r, i)
	frameA(zl, zh).StoreArray(pa)
	frameB(zl, zh).StoreArray(pb)
}

// soaCftbsubTailPair is soaCftbsubTail from s into the frames a and b.
func soaCftbsubTailPair(s *[256]float32, a, b *[128]float32) {
	inR, inI := soaHalves8(s)
	var zero vec8
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr := (*[104]float32)(inR[j : j+104])
		pi := (*[104]float32)(inI[j : j+104])
		ar, ai := ld8(q8w104(pr, 0)), neg8(ld8(q8w104(pi, 0)))
		br, bi := ld8(q8w104(pr, 32)), neg8(ld8(q8w104(pi, 32)))
		cr, ci := ld8(q8w104(pr, 64)), ld8(q8w104(pi, 64))
		dr, di := ld8(q8w104(pr, 96)), ld8(q8w104(pi, 96))
		x0r, x0i := ar.Add(br), ai.Add(bi)
		x1r, x1i := ar.Sub(br), ai.Sub(bi)
		x2r, x2i := cr.Add(dr), neg8(ci.Add(di))
		x3r, x3i := cr.Sub(dr), ci.Sub(di)
		oa := (*[104]float32)(a[j : j+104])
		ob := (*[104]float32)(b[j : j+104])
		soaStorePackedPair((*[8]float32)(oa[0:8]), (*[8]float32)(ob[0:8]), x0r.Add(x2r), x0i.Add(x2i))
		soaStorePackedPair((*[8]float32)(oa[32:40]), (*[8]float32)(ob[32:40]), x1r.Sub(x3i), x1i.Sub(x3r))
		soaStorePackedPair((*[8]float32)(oa[64:72]), (*[8]float32)(ob[64:72]), x0r.Sub(x2r), x0i.Sub(x2i))
		soaStorePackedPair((*[8]float32)(oa[96:104]), (*[8]float32)(ob[96:104]), x1r.Add(x3i), x1i.Add(x3r))
	}
	step(0)
	step(8)
	step(16)
	step(24)
}

// soaRftfsubPair is soaRftfsub on the interleaved frames, two iterations per
// round, except for bin 64 (see fftSplitPairSoa).
func soaRftfsubPair(t *[256]float32, re0, im0, re1, im1 *[65]float32) {
	tr, ti := soaHalves8(t)
	trh, tih := (*[64]float32)(tr[64:]), (*[64]float32)(ti[64:])
	// Lane 0 of dc is a[0] + a[1].
	dc := ld8((*[8]float32)(tr[0:8])).Add(ld8((*[8]float32)(ti[0:8])))
	var zero, prevR, prevI, r0, r1, hR, hI vec8
	round := func(q int) {
		u := 3 - q
		loR := (*[16]float32)(tr[16*q : 16*q+16])
		loI := (*[16]float32)(ti[16*q : 16*q+16])
		hiR := (*[16]float32)(trh[16*u : 16*u+16])
		hiI := (*[16]float32)(tih[16*u : 16*u+16])
		wr := (*[16]float32)(rftWkr8[16*q : 16*q+16])
		wi := (*[16]float32)(rftWki8[16*q : 16*q+16])

		// Iteration m = 2q.
		rhR, rhI := rev4g(ld8((*[8]float32)(hiR[8:16]))), rev4g(ld8((*[8]float32)(hiI[8:16])))
		p0, p1, k0, k1 := rftfArith8(ld8((*[8]float32)(loR[0:8])), ld8((*[8]float32)(loI[0:8])), ext3g(prevR, rhR), ext3g(prevI, rhI),
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
		rhR, rhI = rev4g(ld8((*[8]float32)(hiR[0:8]))), rev4g(ld8((*[8]float32)(hiI[0:8])))
		s0, s1, k0, k1 := rftfArith8(ld8((*[8]float32)(loR[8:16])), ld8((*[8]float32)(loI[8:16])), ext3g(prevR, rhR), ext3g(prevI, rhI),
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
	// Bins 32..39: bin 32 as the frame has it (the last lane of the reversed
	// positions 32..35), then the last results.
	xr, xi := ext3g(prevR, r0), ext3g(prevI, r1)
	frameA(xr, hR).StoreArray(h8(re0, 0))
	frameB(xr, hR).StoreArray(h8(re1, 0))
	frameA(xi, hI).StoreArray(h8(im0, 0))
	frameB(xi, hI).StoreArray(h8(im1, 0))
}

// soaRftbsubPair is soaRftbsub for two sets of bins, into the interleaved
// t. The scalar values of bins 0, 32 and 64 are computed before the first
// 256-bit operation.
func soaRftbsubPair(re0, im0, re1, im1 *[65]float32, t *[256]float32) {
	a1 := float32(0.5 * (re0[0] - re0[64]))
	a0 := re0[0] - a1
	a1 = -a1
	b1 := float32(0.5 * (re1[0] - re1[64]))
	b0 := re1[0] - b1
	b1 = -b1
	// Lane 0 of dcR and dcI is position 0; lane 3 of x32R and x32I is
	// position 32, bin 32 with its imaginary part negated.
	var z vec
	dcRa, dcRb := z.SetElem(0, a0), z.SetElem(0, b0)
	dcIa, dcIb := z.SetElem(0, a1), z.SetElem(0, b1)
	xRa, xRb := z.SetElem(3, re0[32]), z.SetElem(3, re1[32])
	xIa, xIb := z.SetElem(3, -im0[32]), z.SetElem(3, -im1[32])
	var z8 vec8
	dcR, dcI := z8.SetLo(dcRa).SetHi(dcRb), z8.SetLo(dcIa).SetHi(dcIb)
	x32R, x32I := z8.SetLo(xRa).SetHi(xRb), z8.SetLo(xIa).SetHi(xIb)

	tr, ti := soaHalves8(t)
	trh, tih := (*[64]float32)(tr[64:]), (*[64]float32)(ti[64:])
	var prevR, prevI, r0, r1 vec8
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
		if m == 0 {
			p0 = lane0g(dcR, p0)
			p1 = lane0g(dcI, p1)
		}
		p0.StoreArray((*[8]float32)(tr[8*m : 8*m+8]))
		p1.StoreArray((*[8]float32)(ti[8*m : 8*m+8]))
		k0, k1 = rev4g(k0), rev4g(k1) // bins 61-4m..64-4m
		if m > 0 {
			// Positions 36+4n..39+4n; the mask changes nothing for n < 7.
			j := (8*n + 8) & 56
			ext3g(k0, r0).StoreArray((*[8]float32)(trh[j : j+8]))
			ext3g(k1, r1).StoreArray((*[8]float32)(tih[j : j+8]))
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
	// Positions 32..35: bin 32, then the last results.
	ext3g(x32R, r0).StoreArray((*[8]float32)(trh[0:8]))
	ext3g(x32I, r1).StoreArray((*[8]float32)(tih[0:8]))
}
