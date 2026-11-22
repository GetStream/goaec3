//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import "simd/archsimd"

// The split transforms of the NEON and SSE2 lanes, with the real and the
// imaginary parts in separate vectors.
//
// ooura_fft_neon.cc and ooura_fft_sse2.cc keep a complex value in two
// adjacent lanes, so a rotation w*x is wr*x + wi*rev64(x), and x1 ± i*x3
// goes through rev64 and the {-1, 1} sign table: close to half of their
// vector operations are shuffles. Here a vector holds the real or the
// imaginary parts of four complex values, and every float operation is the
// one of the C code for that element. A rotation is
// (wrr*xr + wri*xi, wir*xi + wii*xr) with the four table values the two
// lanes of the C vectors hold. x + s*y for s in {1, -1} is x + y or x - y:
// s*y is y or its negation, exactly, and adding a negation is subtracting.
// Products are Mul followed by Add or Sub, never MulAdd.
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
// A pass with a fixed trip count is a closure called once per iteration
// with constant arguments, so that every load and store is base plus
// displacement and carries no bounds check (TestNoBoundsChecks).
//
// An inlined call leaves a NOP that marks its place unless its line has an
// instruction of its own, and a NOP takes a place in the pipeline. Every
// line of the passes that calls a helper therefore has an archsimd
// operation of its own (a load written as archsimd.LoadFloat32x4Array
// instead of ld, a shuffle next to the addition or the store that takes
// it), and a value that two lines need is written out in both: the
// compiler computes it once.
//
// TestSoaMatchesRef checks the transforms against the lane reference.

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

type vec = archsimd.Float32x4

func ld(p *[4]float32) vec { return archsimd.LoadFloat32x4Array(p) }

// zipLo is {x0, y0, x1, y1}; zipHi is {x2, y2, x3, y3}.
func zipLo(x, y vec) vec { return x.ToBits().InterleaveLo(y.ToBits()).BitsToFloat32() }

func zipHi(x, y vec) vec { return x.ToBits().InterleaveHi(y.ToBits()).BitsToFloat32() }

// Scratch is the work area of FftSplit and InverseFftSplit. It needs no
// initialization and carries nothing from one call to the next. It lives
// with the caller so that no transform clears a frame on its stack.
type Scratch struct {
	t [128]float32
}

// rftSplitWkr[j1] = 0.5f - c[32-j1] for j1 = 1..31 (c = rdft_w + 32); the
// matching wki is c[j1] = rdftW[32+j1].
var rftSplitWkr [32]float32

func init() {
	for j1 := 1; j1 < 32; j1++ {
		rftSplitWkr[j1] = 0.5 - rdftW[64-j1]
	}
}

func rev2(x int) int { return (x&1)<<1 | x>>1 }

// initSoaTables fills the tables. It runs after the init of tables.go,
// which sets the negative zeros of the tables it reads.
func initSoaTables() {
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

// fftSplitSoa is Fft followed by FftData::CopyFromPackedArray into re and
// im. It leaves a in an intermediate state; t is work space.
func fftSplitSoa(a *[128]float32, re, im *[65]float32, t *[128]float32) {
	soaCft1stPacked(a, t)
	soaCftmdl(t, a)
	soaCftfsubTail(a, t)
	soaRftfsub(t, re, im)
}

// inverseFftSplitSoa is FftData::CopyToPackedArray of re and im followed by
// InverseFft, into a; t is work space.
func inverseFftSplitSoa(re, im *[65]float32, a *[128]float32, t *[128]float32) {
	soaRftbsub(re, im, t)
	soaCft1stSplit(t, a)
	soaCftmdl(a, t)
	soaCftbsubTail(t, a)
}

// A frame in split form holds the real parts of its 64 complex values in
// its first half and the imaginary parts in its second half.

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

// The cft1st passes are bitrv2_128 and cft1st_128 by load group: the four
// inputs of group m hold, in lane k, the values of index
// 16*rev2(c) + 4*m + k for c = 0..3, which are positions 4*i..4*i+3 with
// i = 4*rev2(k) + rev2(m). The results o0..o3 go to t in split form, at
// their positions: lane k of the results (soaLane0..3, the transposition)
// is positions 16*rev2(k) + 4*rev2(m) + 0..3. A rotation by the factors w,
// w+1, w+2 and w+3 of a table is written out.

// soaCft1stPacked is bitrv2_128 and cft1st_128 from the frame a, whose
// complex values are pairs of floats, into t in split form.
func soaCft1stPacked(a, t *[128]float32) {
	outR, outI := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
	var zero vec
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		// Index 16*rev2(c) + 4*m is float 32*rev2(c) + 8*m.
		w := (*[104]float32)(a[8*m : 8*m+104])
		wr, wi := (*[52]float32)(outR[p:p+52]), (*[52]float32)(outI[p:p+52])
		tab := &soaS1[m]
		// The values a, b, c and d of the butterfly (soaButterfly), each the
		// even and the odd floats of two loads.
		a0, a1, b0, b1 := archsimd.LoadFloat32x4Array((*[4]float32)(w[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(w[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(w[64:68])), archsimd.LoadFloat32x4Array((*[4]float32)(w[68:72]))
		c0, c1, d0, d1 := archsimd.LoadFloat32x4Array((*[4]float32)(w[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(w[36:40])), archsimd.LoadFloat32x4Array((*[4]float32)(w[96:100])), archsimd.LoadFloat32x4Array((*[4]float32)(w[100:104]))
		x0r, x0i := even(a0, a1).Add(even(b0, b1)), odd(a0, a1).Add(odd(b0, b1))
		x1r, x1i := even(a0, a1).Sub(even(b0, b1)), odd(a0, a1).Sub(odd(b0, b1))
		x2r, x2i := even(c0, c1).Add(even(d0, d1)), odd(c0, c1).Add(odd(d0, d1))
		x3r, x3i := even(c0, c1).Sub(even(d0, d1)), odd(c0, c1).Sub(odd(d0, d1))
		o0r, o0i, tr, ti := x0r.Add(x2r), x0i.Add(x2i), x0r.Sub(x2r), x0i.Sub(x2i)
		yr, yi, zr, zi := x1r.Sub(x3i), x1i.Add(x3r), x1r.Add(x3i), x1i.Sub(x3r)
		o1r, o1i := ld(&tab[0]).Mul(yr).Add(ld(&tab[1]).Mul(yi)), ld(&tab[2]).Mul(yi).Add(ld(&tab[3]).Mul(yr))
		o2r, o2i := ld(&tab[4]).Mul(tr).Add(ld(&tab[5]).Mul(ti)), ld(&tab[6]).Mul(ti).Add(ld(&tab[7]).Mul(tr))
		o3r, o3i := ld(&tab[8]).Mul(zr).Add(ld(&tab[9]).Mul(zi)), ld(&tab[10]).Mul(zi).Add(ld(&tab[11]).Mul(zr))
		soaLane0(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[0:4]))
		soaLane2(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[16:20]))
		soaLane1(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[32:36]))
		soaLane3(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[48:52]))
		soaLane0(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[0:4]))
		soaLane2(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[16:20]))
		soaLane1(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[32:36]))
		soaLane3(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[48:52]))
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
	inR, inI := (*[64]float32)(s[:64]), (*[64]float32)(s[64:])
	outR, outI := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
	var zero vec
	group := func(m, p int) {
		_ = zero // a captured vector makes the compiler inline the closure
		sr, si := (*[52]float32)(inR[4*m:4*m+52]), (*[52]float32)(inI[4*m:4*m+52])
		wr, wi := (*[52]float32)(outR[p:p+52]), (*[52]float32)(outI[p:p+52])
		tab := &soaS1[m]
		o0r, o0i, tr, ti, yr, yi, zr, zi := soaButterfly(archsimd.LoadFloat32x4Array((*[4]float32)(sr[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(si[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(sr[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(si[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(sr[16:20])), archsimd.LoadFloat32x4Array((*[4]float32)(si[16:20])), archsimd.LoadFloat32x4Array((*[4]float32)(sr[48:52])), archsimd.LoadFloat32x4Array((*[4]float32)(si[48:52])))
		o1r, o1i := ld(&tab[0]).Mul(yr).Add(ld(&tab[1]).Mul(yi)), ld(&tab[2]).Mul(yi).Add(ld(&tab[3]).Mul(yr))
		o2r, o2i := ld(&tab[4]).Mul(tr).Add(ld(&tab[5]).Mul(ti)), ld(&tab[6]).Mul(ti).Add(ld(&tab[7]).Mul(tr))
		o3r, o3i := ld(&tab[8]).Mul(zr).Add(ld(&tab[9]).Mul(zi)), ld(&tab[10]).Mul(zi).Add(ld(&tab[11]).Mul(zr))
		soaLane0(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[0:4]))
		soaLane2(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[16:20]))
		soaLane1(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[32:36]))
		soaLane3(o0r, o1r, o2r, o3r).StoreArray((*[4]float32)(wr[48:52]))
		soaLane0(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[0:4]))
		soaLane2(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[16:20]))
		soaLane1(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[32:36]))
		soaLane3(o0i, o1i, o2i, o3i).StoreArray((*[4]float32)(wi[48:52]))
	}
	// p = 4*rev2(m).
	group(0, 0)
	group(1, 8)
	group(2, 4)
	group(3, 12)
}

// soaCftmdl is cftmdl_128 from s to t in split form. Block b is positions
// 16*b..16*b+15, and a butterfly combines the four values that are four
// positions apart: the four vectors of a block.
func soaCftmdl(s, t *[128]float32) {
	inR, inI := (*[64]float32)(s[:64]), (*[64]float32)(s[64:])
	outR, outI := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])

	// Block 0: no rotation.
	{
		pr, pi := (*[16]float32)(inR[0:16]), (*[16]float32)(inI[0:16])
		qr, qi := (*[16]float32)(outR[0:16]), (*[16]float32)(outI[0:16])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(archsimd.LoadFloat32x4Array((*[4]float32)(pr[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[12:16])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[12:16])))
		sr.StoreArray((*[4]float32)(qr[0:4]))
		yr.StoreArray((*[4]float32)(qr[4:8]))
		tr.StoreArray((*[4]float32)(qr[8:12]))
		zr.StoreArray((*[4]float32)(qr[12:16]))
		si.StoreArray((*[4]float32)(qi[0:4]))
		yi.StoreArray((*[4]float32)(qi[4:8]))
		ti.StoreArray((*[4]float32)(qi[8:12]))
		zi.StoreArray((*[4]float32)(qi[12:16]))
	}

	// Block 1: t turns by a quarter, y and z by wk1r.
	{
		pr, pi := (*[16]float32)(inR[16:32]), (*[16]float32)(inI[16:32])
		qr, qi := (*[16]float32)(outR[16:32]), (*[16]float32)(outI[16:32])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(archsimd.LoadFloat32x4Array((*[4]float32)(pr[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[12:16])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[12:16])))
		sr.StoreArray((*[4]float32)(qr[0:4]))
		ld(&soaWk1[0]).Mul(yr.Sub(yi)).StoreArray((*[4]float32)(qr[4:8]))
		neg(ti).StoreArray((*[4]float32)(qr[8:12]))
		ld(&soaWk1[3]).Mul(zr.Add(zi)).StoreArray((*[4]float32)(qr[12:16]))
		si.StoreArray((*[4]float32)(qi[0:4]))
		ld(&soaWk1[1]).Mul(yr.Add(yi)).StoreArray((*[4]float32)(qi[4:8]))
		tr.StoreArray((*[4]float32)(qi[8:12]))
		ld(&soaWk1[2]).Mul(zr.Sub(zi)).StoreArray((*[4]float32)(qi[12:16]))
	}

	// Blocks 2 and 3: the rotations of soaS2.
	var zero vec
	rotated := func(b int) {
		_ = zero // a captured vector makes the compiler inline the closure
		tab := &soaS2[b]
		pr, pi := (*[16]float32)(inR[32+16*b:48+16*b]), (*[16]float32)(inI[32+16*b:48+16*b])
		qr, qi := (*[16]float32)(outR[32+16*b:48+16*b]), (*[16]float32)(outI[32+16*b:48+16*b])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(archsimd.LoadFloat32x4Array((*[4]float32)(pr[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[4:8])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[12:16])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[12:16])))
		o1r, o1i := ld(&tab[0]).Mul(yr).Add(ld(&tab[1]).Mul(yi)), ld(&tab[2]).Mul(yi).Add(ld(&tab[3]).Mul(yr))
		o2r, o2i := ld(&tab[4]).Mul(tr).Add(ld(&tab[5]).Mul(ti)), ld(&tab[6]).Mul(ti).Add(ld(&tab[7]).Mul(tr))
		o3r, o3i := ld(&tab[8]).Mul(zr).Add(ld(&tab[9]).Mul(zi)), ld(&tab[10]).Mul(zi).Add(ld(&tab[11]).Mul(zr))
		sr.StoreArray((*[4]float32)(qr[0:4]))
		o1r.StoreArray((*[4]float32)(qr[4:8]))
		o2r.StoreArray((*[4]float32)(qr[8:12]))
		o3r.StoreArray((*[4]float32)(qr[12:16]))
		si.StoreArray((*[4]float32)(qi[0:4]))
		o1i.StoreArray((*[4]float32)(qi[4:8]))
		o2i.StoreArray((*[4]float32)(qi[8:12]))
		o3i.StoreArray((*[4]float32)(qi[12:16]))
	}
	rotated(0)
	rotated(1)
}

// soaCftfsubTail is the last pass of cftfsub_128 from s to t in split form:
// a butterfly combines the four values that are sixteen positions apart.
func soaCftfsubTail(s, t *[128]float32) {
	inR, inI := (*[64]float32)(s[:64]), (*[64]float32)(s[64:])
	outR, outI := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
	var zero vec
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr, pi := (*[52]float32)(inR[j:j+52]), (*[52]float32)(inI[j:j+52])
		qr, qi := (*[52]float32)(outR[j:j+52]), (*[52]float32)(outI[j:j+52])
		sr, si, tr, ti, yr, yi, zr, zi := soaButterfly(archsimd.LoadFloat32x4Array((*[4]float32)(pr[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[16:20])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[16:20])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(pr[48:52])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[48:52])))
		sr.StoreArray((*[4]float32)(qr[0:4]))
		si.StoreArray((*[4]float32)(qi[0:4]))
		yr.StoreArray((*[4]float32)(qr[16:20]))
		yi.StoreArray((*[4]float32)(qi[16:20]))
		tr.StoreArray((*[4]float32)(qr[32:36]))
		ti.StoreArray((*[4]float32)(qi[32:36]))
		zr.StoreArray((*[4]float32)(qr[48:52]))
		zi.StoreArray((*[4]float32)(qi[48:52]))
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
	inR, inI := (*[64]float32)(s[:64]), (*[64]float32)(s[64:])
	var zero vec
	step := func(j int) {
		_ = zero // a captured vector makes the compiler inline the closure
		pr, pi := (*[52]float32)(inR[j:j+52]), (*[52]float32)(inI[j:j+52])
		ar, ai := archsimd.LoadFloat32x4Array((*[4]float32)(pr[0:4])), neg(archsimd.LoadFloat32x4Array((*[4]float32)(pi[0:4])))
		br, bi := archsimd.LoadFloat32x4Array((*[4]float32)(pr[16:20])), neg(archsimd.LoadFloat32x4Array((*[4]float32)(pi[16:20])))
		cr, ci := archsimd.LoadFloat32x4Array((*[4]float32)(pr[32:36])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[32:36]))
		dr, di := archsimd.LoadFloat32x4Array((*[4]float32)(pr[48:52])), archsimd.LoadFloat32x4Array((*[4]float32)(pi[48:52]))
		x0r, x0i := ar.Add(br), ai.Add(bi)
		x1r, x1i := ar.Sub(br), ai.Sub(bi)
		x2r, x2i := cr.Add(dr), neg(ci.Add(di))
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

// soaRftfsub is rftfsub_128 and the final butterfly of Fft on a frame in
// split form, with the result written as bins. Position k of the frame is
// bin k, and the real part of bin 64 is the imaginary part of position 0.
//
// rftfsub and rftbsub combine bin j1 with bin 64-j1 for j1 = 1..31, by the
// same float operations in the vectorized and in the scalar iterations of
// the C sources, so the passes run all of them in vectors: iteration m
// takes bins 4m..4m+3 (lane 0 of iteration 0 is bin 0, which the passes
// compute apart) with their partners 64-4m, 63-4m, 62-4m, 61-4m, the last
// three from the reversed bins 60-4m..63-4m and the first from the
// iteration before.
func soaRftfsub(t *[128]float32, re, im *[65]float32) {
	tr, ti := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
	trh, tih := (*[32]float32)(tr[32:64]), (*[32]float32)(ti[32:64])
	reh, imh := (*[32]float32)(re[32:64]), (*[32]float32)(im[32:64])
	wr, wi := (*[32]float32)(rftSplitWkr[:]), (*[32]float32)(rdftW[32:64])
	a0, a1 := tr[0], ti[0]
	re32, im32 := tr[32], ti[32]
	// prevR and prevI are the reversed bins 64-4m..67-4m, whose last lane is
	// bin 64-4m; r0 and r1 the results for bins 65-4m..68-4m.
	var prevR, prevI, r0, r1 vec
	step := func(m int) {
		n := 7 - m
		p0, p1 := archsimd.LoadFloat32x4Array((*[4]float32)(tr[4*m:4*m+4])), archsimd.LoadFloat32x4Array((*[4]float32)(ti[4*m:4*m+4]))
		rhR, rhI := rev4(archsimd.LoadFloat32x4Array((*[4]float32)(trh[4*n:4*n+4]))), rev4(archsimd.LoadFloat32x4Array((*[4]float32)(tih[4*n:4*n+4])))
		wkr, wki := archsimd.LoadFloat32x4Array((*[4]float32)(wr[4*m:4*m+4])), archsimd.LoadFloat32x4Array((*[4]float32)(wi[4*m:4*m+4]))
		// The partners are k0 = ext3(prevR, rhR) and k1 = ext3(prevI, rhI).
		xr, xi := p0.Sub(ext3(prevR, rhR)), p1.Add(ext3(prevI, rhI))
		yr := wkr.Mul(xr).Sub(wki.Mul(xi))
		yi := wkr.Mul(xi).Add(wki.Mul(xr))
		p0, p1 = p0.Sub(yr), p1.Sub(yi)
		// Their results, reversed: bins 61-4m..64-4m.
		k0, k1 := rev4(ext3(prevR, rhR).Add(yr)), rev4(ext3(prevI, rhI).Sub(yi))
		prevR, prevI = rhR, rhI
		if m == 0 {
			p0 = p0.SetElem(0, a0+a1)
			p1 = p1.SetElem(0, 0)
		}
		p0.StoreArray((*[4]float32)(re[4*m : 4*m+4]))
		p1.StoreArray((*[4]float32)(im[4*m : 4*m+4]))
		if m > 0 {
			// Bins 64-4m..67-4m; the mask changes nothing for n < 7.
			j := (4*n + 4) & 28
			ext3(k0, r0).StoreArray((*[4]float32)(reh[j : j+4]))
			ext3(k1, r1).StoreArray((*[4]float32)(imh[j : j+4]))
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
	ext3(r0, r0).SetElem(0, re32).StoreArray((*[4]float32)(reh[0:4]))
	ext3(r1, r1).SetElem(0, im32).StoreArray((*[4]float32)(imh[0:4]))
	re[64] = a0 - a1
	im[64] = 0
}

// soaRftbsub is the first butterfly of InverseFft and rftbsub_128 on the
// frame that packs the bins re and im, into a frame in split form.
func soaRftbsub(re, im *[65]float32, t *[128]float32) {
	tr, ti := (*[64]float32)(t[:64]), (*[64]float32)(t[64:])
	trh, tih := (*[32]float32)(tr[32:64]), (*[32]float32)(ti[32:64])
	reh, imh := (*[32]float32)(re[32:64]), (*[32]float32)(im[32:64])
	wr, wi := (*[32]float32)(rftSplitWkr[:]), (*[32]float32)(rdftW[32:64])
	a1 := float32(0.5 * (re[0] - re[64]))
	a0 := re[0] - a1
	a1 = -a1
	re32, im32 := re[32], -im[32]
	// prevR and prevI are as in soaRftfsub, r0 and r1 the results for bins
	// 65-4m..68-4m.
	var prevR, prevI, r0, r1 vec
	step := func(m int) {
		n := 7 - m
		p0, p1 := archsimd.LoadFloat32x4Array((*[4]float32)(re[4*m:4*m+4])), archsimd.LoadFloat32x4Array((*[4]float32)(im[4*m:4*m+4]))
		rhR, rhI := rev4(archsimd.LoadFloat32x4Array((*[4]float32)(reh[4*n:4*n+4]))), rev4(archsimd.LoadFloat32x4Array((*[4]float32)(imh[4*n:4*n+4])))
		wkr, wki := archsimd.LoadFloat32x4Array((*[4]float32)(wr[4*m:4*m+4])), archsimd.LoadFloat32x4Array((*[4]float32)(wi[4*m:4*m+4]))
		// The partners are k0 = ext3(prevR, rhR) and k1 = ext3(prevI, rhI).
		xr, xi := p0.Sub(ext3(prevR, rhR)), p1.Add(ext3(prevI, rhI))
		yr := wkr.Mul(xr).Add(wki.Mul(xi))
		yi := wkr.Mul(xi).Sub(wki.Mul(xr))
		p0, p1 = p0.Sub(yr), yi.Sub(p1)
		// Their results, reversed: bins 61-4m..64-4m.
		k0, k1 := rev4(ext3(prevR, rhR).Add(yr)), rev4(yi.Sub(ext3(prevI, rhI)))
		prevR, prevI = rhR, rhI
		if m == 0 {
			p0 = p0.SetElem(0, a0)
			p1 = p1.SetElem(0, a1)
		}
		p0.StoreArray((*[4]float32)(tr[4*m : 4*m+4]))
		p1.StoreArray((*[4]float32)(ti[4*m : 4*m+4]))
		if m > 0 {
			// Positions 36+4n..39+4n; the mask changes nothing for n < 7.
			j := (4*n + 4) & 28
			ext3(k0, r0).StoreArray((*[4]float32)(trh[j : j+4]))
			ext3(k1, r1).StoreArray((*[4]float32)(tih[j : j+4]))
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
	ext3(r0, r0).SetElem(0, re32).StoreArray((*[4]float32)(trh[0:4]))
	ext3(r1, r1).SetElem(0, im32).StoreArray((*[4]float32)(tih[0:4]))
}
