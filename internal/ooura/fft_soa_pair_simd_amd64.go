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
// As in fft_soa_simd.go, every line that calls a helper has an archsimd
// operation of its own; two 128-bit loads are gathered into a vector with
// SetLo and SetHi on the line (ldPair written out).
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

func neg8(x vec8) vec8 {
	return x.ToBits().Xor(archsimd.LoadUint32x8Array(&signAll8)).BitsToFloat32()
}

// soaLane0g..3g are soaLane0..3 on both halves.
func soaLane0g(r0, r1, r2, r3 vec8) vec8 {
	return r0.ConcatPermuteScalarsGrouped(0, 1, 4, 5, r1).ConcatPermuteScalarsGrouped(0, 2, 4, 6, r2.ConcatPermuteScalarsGrouped(0, 1, 4, 5, r3))
}

func soaLane1g(r0, r1, r2, r3 vec8) vec8 {
	return r0.ConcatPermuteScalarsGrouped(0, 1, 4, 5, r1).ConcatPermuteScalarsGrouped(1, 3, 5, 7, r2.ConcatPermuteScalarsGrouped(0, 1, 4, 5, r3))
}

func soaLane2g(r0, r1, r2, r3 vec8) vec8 {
	return r0.ConcatPermuteScalarsGrouped(2, 3, 6, 7, r1).ConcatPermuteScalarsGrouped(0, 2, 4, 6, r2.ConcatPermuteScalarsGrouped(2, 3, 6, 7, r3))
}

func soaLane3g(r0, r1, r2, r3 vec8) vec8 {
	return r0.ConcatPermuteScalarsGrouped(2, 3, 6, 7, r1).ConcatPermuteScalarsGrouped(1, 3, 5, 7, r2.ConcatPermuteScalarsGrouped(2, 3, 6, 7, r3))
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

// soaCft1stPackedPair is soaCft1stPacked from the frames a and b into t.
func soaCft1stPackedPair(a, b *[128]float32, t *[256]float32) {
	outR, outI := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
	var zero vec8
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		wa, wb := (*[104]float32)(a[8*m:8*m+104]), (*[104]float32)(b[8*m:8*m+104])
		wr, wi := (*[104]float32)(outR[p:p+104]), (*[104]float32)(outI[p:p+104])
		tab := &soaS1x8[m]
		a0, a1, b0, b1 := zero.SetLo(ld((*[4]float32)(wa[0:4]))).SetHi(ld((*[4]float32)(wb[0:4]))), zero.SetLo(ld((*[4]float32)(wa[4:8]))).SetHi(ld((*[4]float32)(wb[4:8]))), zero.SetLo(ld((*[4]float32)(wa[64:68]))).SetHi(ld((*[4]float32)(wb[64:68]))), zero.SetLo(ld((*[4]float32)(wa[68:72]))).SetHi(ld((*[4]float32)(wb[68:72])))
		c0, c1, d0, d1 := zero.SetLo(ld((*[4]float32)(wa[32:36]))).SetHi(ld((*[4]float32)(wb[32:36]))), zero.SetLo(ld((*[4]float32)(wa[36:40]))).SetHi(ld((*[4]float32)(wb[36:40]))), zero.SetLo(ld((*[4]float32)(wa[96:100]))).SetHi(ld((*[4]float32)(wb[96:100]))), zero.SetLo(ld((*[4]float32)(wa[100:104]))).SetHi(ld((*[4]float32)(wb[100:104])))
		x0r, x0i := eveng(a0, a1).Add(eveng(b0, b1)), oddg(a0, a1).Add(oddg(b0, b1))
		x1r, x1i := eveng(a0, a1).Sub(eveng(b0, b1)), oddg(a0, a1).Sub(oddg(b0, b1))
		x2r, x2i := eveng(c0, c1).Add(eveng(d0, d1)), oddg(c0, c1).Add(oddg(d0, d1))
		x3r, x3i := eveng(c0, c1).Sub(eveng(d0, d1)), oddg(c0, c1).Sub(oddg(d0, d1))
		o0r, o0i, tr, ti := x0r.Add(x2r), x0i.Add(x2i), x0r.Sub(x2r), x0i.Sub(x2i)
		yr, yi, zr, zi := x1r.Sub(x3i), x1i.Add(x3r), x1r.Add(x3i), x1i.Sub(x3r)
		o1r, o1i := ld8(&tab[0]).Mul(yr).Add(ld8(&tab[1]).Mul(yi)), ld8(&tab[2]).Mul(yi).Add(ld8(&tab[3]).Mul(yr))
		o2r, o2i := ld8(&tab[4]).Mul(tr).Add(ld8(&tab[5]).Mul(ti)), ld8(&tab[6]).Mul(ti).Add(ld8(&tab[7]).Mul(tr))
		o3r, o3i := ld8(&tab[8]).Mul(zr).Add(ld8(&tab[9]).Mul(zi)), ld8(&tab[10]).Mul(zi).Add(ld8(&tab[11]).Mul(zr))
		soaLane0g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[0:8]))
		soaLane2g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[32:40]))
		soaLane1g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[64:72]))
		soaLane3g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[96:104]))
		soaLane0g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[0:8]))
		soaLane2g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[32:40]))
		soaLane1g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[64:72]))
		soaLane3g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[96:104]))
	}
	// p = 8*rev2(m).
	group(0, 0)
	group(1, 16)
	group(2, 8)
	group(3, 24)
}

// soaCft1stSplitPair is soaCft1stSplit from s to t.
func soaCft1stSplitPair(s, t *[256]float32) {
	inR, inI := (*[128]float32)(s[:128]), (*[128]float32)(s[128:])
	outR, outI := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
	var zero vec8
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		sr, si := (*[104]float32)(inR[8*m:8*m+104]), (*[104]float32)(inI[8*m:8*m+104])
		wr, wi := (*[104]float32)(outR[p:p+104]), (*[104]float32)(outI[p:p+104])
		tab := &soaS1x8[m]
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly8(archsimd.LoadFloat32x8Array((*[8]float32)(sr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(si[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(sr[64:72])), archsimd.LoadFloat32x8Array((*[8]float32)(si[64:72])), archsimd.LoadFloat32x8Array((*[8]float32)(sr[32:40])), archsimd.LoadFloat32x8Array((*[8]float32)(si[32:40])), archsimd.LoadFloat32x8Array((*[8]float32)(sr[96:104])), archsimd.LoadFloat32x8Array((*[8]float32)(si[96:104])))
		o1r, o1i := ld8(&tab[0]).Mul(yr).Add(ld8(&tab[1]).Mul(yi)), ld8(&tab[2]).Mul(yi).Add(ld8(&tab[3]).Mul(yr))
		o2r, o2i := ld8(&tab[4]).Mul(tr).Add(ld8(&tab[5]).Mul(ti)), ld8(&tab[6]).Mul(ti).Add(ld8(&tab[7]).Mul(tr))
		o3r, o3i := ld8(&tab[8]).Mul(zr).Add(ld8(&tab[9]).Mul(zi)), ld8(&tab[10]).Mul(zi).Add(ld8(&tab[11]).Mul(zr))
		soaLane0g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[0:8]))
		soaLane2g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[32:40]))
		soaLane1g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[64:72]))
		soaLane3g(o0r, o1r, o2r, o3r).StoreArray((*[8]float32)(wr[96:104]))
		soaLane0g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[0:8]))
		soaLane2g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[32:40]))
		soaLane1g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[64:72]))
		soaLane3g(o0i, o1i, o2i, o3i).StoreArray((*[8]float32)(wi[96:104]))
	}
	// p = 8*rev2(m).
	group(0, 0)
	group(1, 16)
	group(2, 8)
	group(3, 24)
}

// soaCftmdlPair is soaCftmdl in place: a block loads its eight vectors
// before it stores into them.
func soaCftmdlPair(t *[256]float32) {
	r, i := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])

	// Block 0: no rotation.
	{
		pr, pi := (*[32]float32)(r[0:32]), (*[32]float32)(i[0:32])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(archsimd.LoadFloat32x8Array((*[8]float32)(pr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[24:32])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[24:32])))
		sr.StoreArray((*[8]float32)(pr[0:8]))
		yr.StoreArray((*[8]float32)(pr[8:16]))
		tr.StoreArray((*[8]float32)(pr[16:24]))
		zr.StoreArray((*[8]float32)(pr[24:32]))
		si.StoreArray((*[8]float32)(pi[0:8]))
		yi.StoreArray((*[8]float32)(pi[8:16]))
		ti.StoreArray((*[8]float32)(pi[16:24]))
		zi.StoreArray((*[8]float32)(pi[24:32]))
	}

	// Block 1: t turns by a quarter, y and z by wk1r.
	{
		pr, pi := (*[32]float32)(r[32:64]), (*[32]float32)(i[32:64])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(archsimd.LoadFloat32x8Array((*[8]float32)(pr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[24:32])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[24:32])))
		sr.StoreArray((*[8]float32)(pr[0:8]))
		ld8(&soaWk1x8[0]).Mul(yr.Sub(yi)).StoreArray((*[8]float32)(pr[8:16]))
		neg8(ti).StoreArray((*[8]float32)(pr[16:24]))
		ld8(&soaWk1x8[3]).Mul(zr.Add(zi)).StoreArray((*[8]float32)(pr[24:32]))
		si.StoreArray((*[8]float32)(pi[0:8]))
		ld8(&soaWk1x8[1]).Mul(yr.Add(yi)).StoreArray((*[8]float32)(pi[8:16]))
		tr.StoreArray((*[8]float32)(pi[16:24]))
		ld8(&soaWk1x8[2]).Mul(zr.Sub(zi)).StoreArray((*[8]float32)(pi[24:32]))
	}

	// Blocks 2 and 3: the rotations of soaS2.
	var zero vec8
	rotated := func(b int) {
		_ = zero // a captured vector makes the compiler inline the closure
		tab := &soaS2x8[b]
		pr, pi := (*[32]float32)(r[64+32*b:96+32*b]), (*[32]float32)(i[64+32*b:96+32*b])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(archsimd.LoadFloat32x8Array((*[8]float32)(pr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[16:24])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[24:32])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[24:32])))
		o1r, o1i := ld8(&tab[0]).Mul(yr).Add(ld8(&tab[1]).Mul(yi)), ld8(&tab[2]).Mul(yi).Add(ld8(&tab[3]).Mul(yr))
		o2r, o2i := ld8(&tab[4]).Mul(tr).Add(ld8(&tab[5]).Mul(ti)), ld8(&tab[6]).Mul(ti).Add(ld8(&tab[7]).Mul(tr))
		o3r, o3i := ld8(&tab[8]).Mul(zr).Add(ld8(&tab[9]).Mul(zi)), ld8(&tab[10]).Mul(zi).Add(ld8(&tab[11]).Mul(zr))
		sr.StoreArray((*[8]float32)(pr[0:8]))
		o1r.StoreArray((*[8]float32)(pr[8:16]))
		o2r.StoreArray((*[8]float32)(pr[16:24]))
		o3r.StoreArray((*[8]float32)(pr[24:32]))
		si.StoreArray((*[8]float32)(pi[0:8]))
		o1i.StoreArray((*[8]float32)(pi[8:16]))
		o2i.StoreArray((*[8]float32)(pi[16:24]))
		o3i.StoreArray((*[8]float32)(pi[24:32]))
	}
	rotated(0)
	rotated(1)
}

// soaCftfsubTailPair is soaCftfsubTail in place.
func soaCftfsubTailPair(t *[256]float32) {
	r, i := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
	var zero vec8
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr, pi := (*[104]float32)(r[j:j+104]), (*[104]float32)(i[j:j+104])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly8(archsimd.LoadFloat32x8Array((*[8]float32)(pr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[32:40])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[32:40])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[64:72])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[64:72])), archsimd.LoadFloat32x8Array((*[8]float32)(pr[96:104])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[96:104])))
		sr.StoreArray((*[8]float32)(pr[0:8]))
		si.StoreArray((*[8]float32)(pi[0:8]))
		yr.StoreArray((*[8]float32)(pr[32:40]))
		yi.StoreArray((*[8]float32)(pi[32:40]))
		tr.StoreArray((*[8]float32)(pr[64:72]))
		ti.StoreArray((*[8]float32)(pi[64:72]))
		zr.StoreArray((*[8]float32)(pr[96:104]))
		zi.StoreArray((*[8]float32)(pi[96:104]))
	}
	step(0)
	step(8)
	step(16)
	step(24)
}

// soaStorePackedPair stores four complex values of each frame as pairs of
// floats.
func soaStorePackedPair(pa, pb *[8]float32, r, i vec8) {
	frameA(zipLog(r, i), zipHig(r, i)).StoreArray(pa)
	frameB(zipLog(r, i), zipHig(r, i)).StoreArray(pb)
}

// soaCftbsubTailPair is soaCftbsubTail from s into the frames a and b.
func soaCftbsubTailPair(s *[256]float32, a, b *[128]float32) {
	inR, inI := (*[128]float32)(s[:128]), (*[128]float32)(s[128:])
	var zero vec8
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr, pi := (*[104]float32)(inR[j:j+104]), (*[104]float32)(inI[j:j+104])
		ar, ai := archsimd.LoadFloat32x8Array((*[8]float32)(pr[0:8])), neg8(archsimd.LoadFloat32x8Array((*[8]float32)(pi[0:8])))
		br, bi := archsimd.LoadFloat32x8Array((*[8]float32)(pr[32:40])), neg8(archsimd.LoadFloat32x8Array((*[8]float32)(pi[32:40])))
		cr, ci := archsimd.LoadFloat32x8Array((*[8]float32)(pr[64:72])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[64:72]))
		dr, di := archsimd.LoadFloat32x8Array((*[8]float32)(pr[96:104])), archsimd.LoadFloat32x8Array((*[8]float32)(pi[96:104]))
		x0r, x0i := ar.Add(br), ai.Add(bi)
		x1r, x1i := ar.Sub(br), ai.Sub(bi)
		x2r, x2i := cr.Add(dr), neg8(ci.Add(di))
		x3r, x3i := cr.Sub(dr), ci.Sub(di)
		oa, ob := (*[104]float32)(a[j:j+104]), (*[104]float32)(b[j:j+104])
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
// round, except for bin 64 (see fftSplitPairSoa). The rounds pair the
// results of two iterations into one 256-bit store per frame.
func soaRftfsubPair(t *[256]float32, re0, im0, re1, im1 *[65]float32) {
	tr, ti := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
	trh, tih := (*[64]float32)(tr[64:]), (*[64]float32)(ti[64:])
	// Lane 0 of dc is a[0] + a[1].
	dc := archsimd.LoadFloat32x8Array((*[8]float32)(tr[0:8])).Add(archsimd.LoadFloat32x8Array((*[8]float32)(ti[0:8])))
	// r0 and r1 are the reversed results of the iteration before (bins
	// 65-4m..68-4m), g0 and g1 those of the one before that.
	var zero, prevR, prevI, r0, r1, g0, g1 vec8
	round := func(q int) {
		u := 3 - q
		loR, loI := (*[16]float32)(tr[16*q:16*q+16]), (*[16]float32)(ti[16*q:16*q+16])
		hiR, hiI := (*[16]float32)(trh[16*u:16*u+16]), (*[16]float32)(tih[16*u:16*u+16])
		wr, wi := (*[16]float32)(rftWkr8[16*q:16*q+16]), (*[16]float32)(rftWki8[16*q:16*q+16])

		// Iteration m = 2q: the bins p and their partners
		// k = ext3g(prev, rh).
		p0, p1 := archsimd.LoadFloat32x8Array((*[8]float32)(loR[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(loI[0:8]))
		rhR, rhI := rev4g(archsimd.LoadFloat32x8Array((*[8]float32)(hiR[8:16]))), rev4g(archsimd.LoadFloat32x8Array((*[8]float32)(hiI[8:16])))
		wkr, wki := archsimd.LoadFloat32x8Array((*[8]float32)(wr[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(wi[0:8]))
		xr, xi := p0.Sub(ext3g(prevR, rhR)), p1.Add(ext3g(prevI, rhI))
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		p0, p1 = p0.Sub(yr), p1.Sub(yi)
		k0, k1 := rev4g(ext3g(prevR, rhR).Add(yr)), rev4g(ext3g(prevI, rhI).Sub(yi))
		prevR, prevI = rhR, rhI
		if q == 0 {
			// Bin 0 is a[0] + a[1].
			p0 = dc.ConcatPermuteScalarsGrouped(0, 0, 5, 5, p0).ConcatPermuteScalarsGrouped(0, 2, 6, 7, p0)
			p1 = zero.ConcatPermuteScalarsGrouped(0, 0, 5, 5, p1).ConcatPermuteScalarsGrouped(0, 2, 6, 7, p1)
		} else {
			// Bins 64-8q..71-8q.
			frameA(ext3g(k0, r0), ext3g(r0, g0)).StoreArray(h8(re0, u+1))
			frameB(ext3g(k0, r0), ext3g(r0, g0)).StoreArray(h8(re1, u+1))
			frameA(ext3g(k1, r1), ext3g(r1, g1)).StoreArray(h8(im0, u+1))
			frameB(ext3g(k1, r1), ext3g(r1, g1)).StoreArray(h8(im1, u+1))
		}
		g0, g1 = k0, k1

		// Iteration m = 2q+1.
		s0, s1 := archsimd.LoadFloat32x8Array((*[8]float32)(loR[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(loI[8:16]))
		rhR, rhI = rev4g(archsimd.LoadFloat32x8Array((*[8]float32)(hiR[0:8]))), rev4g(archsimd.LoadFloat32x8Array((*[8]float32)(hiI[0:8])))
		wkr, wki = archsimd.LoadFloat32x8Array((*[8]float32)(wr[8:16])), archsimd.LoadFloat32x8Array((*[8]float32)(wi[8:16]))
		xr, xi = s0.Sub(ext3g(prevR, rhR)), s1.Add(ext3g(prevI, rhI))
		yr = wkr.Mul(xr).Sub(wki.Mul(xi))
		yi = wkr.Mul(xi).Add(wki.Mul(xr))
		s0, s1 = s0.Sub(yr), s1.Sub(yi)
		r0, r1 = rev4g(ext3g(prevR, rhR).Add(yr)), rev4g(ext3g(prevI, rhI).Sub(yi))
		prevR, prevI = rhR, rhI
		frameA(p0, s0).StoreArray(b8(re0, q))
		frameB(p0, s0).StoreArray(b8(re1, q))
		frameA(p1, s1).StoreArray(b8(im0, q))
		frameB(p1, s1).StoreArray(b8(im1, q))
	}
	round(0)
	round(1)
	round(2)
	round(3)
	// Bins 32..39: bin 32 as the frame has it (the last lane of the reversed
	// positions 32..35), then the last results.
	frameA(ext3g(prevR, r0), ext3g(r0, g0)).StoreArray(h8(re0, 0))
	frameB(ext3g(prevR, r0), ext3g(r0, g0)).StoreArray(h8(re1, 0))
	frameA(ext3g(prevI, r1), ext3g(r1, g1)).StoreArray(h8(im0, 0))
	frameB(ext3g(prevI, r1), ext3g(r1, g1)).StoreArray(h8(im1, 0))
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

	tr, ti := (*[128]float32)(t[:128]), (*[128]float32)(t[128:])
	trh, tih := (*[64]float32)(tr[64:]), (*[64]float32)(ti[64:])
	var prevR, prevI, r0, r1 vec8
	step := func(m int) {
		n := 7 - m
		p0, p1 := z8.SetLo(ld(lo4(re0, m))).SetHi(ld(lo4(re1, m))), z8.SetLo(ld(lo4(im0, m))).SetHi(ld(lo4(im1, m)))
		rhR, rhI := rev4g(z8.SetLo(ld(hi4(re0, n))).SetHi(ld(hi4(re1, n)))), rev4g(z8.SetLo(ld(hi4(im0, n))).SetHi(ld(hi4(im1, n))))
		wkr, wki := archsimd.LoadFloat32x8Array((*[8]float32)(rftWkr8[8*m:8*m+8])), archsimd.LoadFloat32x8Array((*[8]float32)(rftWki8[8*m:8*m+8]))
		// The partners are k0 = ext3g(prevR, rhR) and k1 = ext3g(prevI, rhI).
		xr, xi := p0.Sub(ext3g(prevR, rhR)), p1.Add(ext3g(prevI, rhI))
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		p0, p1 = p0.Sub(yr), yi.Sub(p1)
		// Their results, reversed: bins 61-4m..64-4m.
		k0, k1 := rev4g(ext3g(prevR, rhR).Add(yr)), rev4g(yi.Sub(ext3g(prevI, rhI)))
		prevR, prevI = rhR, rhI
		if m == 0 {
			p0 = dcR.ConcatPermuteScalarsGrouped(0, 0, 5, 5, p0).ConcatPermuteScalarsGrouped(0, 2, 6, 7, p0)
			p1 = dcI.ConcatPermuteScalarsGrouped(0, 0, 5, 5, p1).ConcatPermuteScalarsGrouped(0, 2, 6, 7, p1)
		}
		p0.StoreArray((*[8]float32)(tr[8*m : 8*m+8]))
		p1.StoreArray((*[8]float32)(ti[8*m : 8*m+8]))
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
