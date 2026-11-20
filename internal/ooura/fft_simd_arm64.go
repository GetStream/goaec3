//go:build goexperiment.simd && arm64 && !purego

package ooura

// useSimd reports whether the archsimd kernels can run: NEON is always
// present on arm64.
const useSimd = true

func fft(a *[128]float32)        { fftSimd(a) }
func inverseFft(a *[128]float32) { inverseFftSimd(a) }

func fftSplit(a *[128]float32, re, im *[65]float32, s *Scratch) { fftSplitSoa(a, re, im, &s.t) }

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

func initPairTables() { initSoaTables() }

func inverseFftSplit(re, im *[65]float32, a *[128]float32, s *Scratch) {
	inverseFftSplitSoa(re, im, a, &s.t)
}

// rev64 is vrev64q_f32 = {x1, x0, x3, x2}. archsimd has no REV64, so it is
// TRN2(x, TRN1(x, x)).
func rev64(x vec) vec {
	u := x.ToBits()
	return u.InterleaveOdd(u.InterleaveEven(u)).BitsToFloat32()
}

// swap64 exchanges the 64-bit halves: {x2, x3, x0, x1} (EXT #8).
func swap64(x vec) vec {
	b := x.ToBits().ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint32s().BitsToFloat32()
}

// even is {x0, x2, y0, y2} (UZP1); odd is {x1, x3, y1, y3} (UZP2).
func even(x, y vec) vec { return x.ToBits().ConcatEven(y.ToBits()).BitsToFloat32() }

func odd(x, y vec) vec { return x.ToBits().ConcatOdd(y.ToBits()).BitsToFloat32() }

// lo64 is vcombine(low(x), low(y)) = {x0, x1, y0, y1}.
func lo64(x, y vec) vec {
	return x.ToBits().ReshapeToUint64s().InterleaveLo(y.ToBits().ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}

// hi64 is vcombine(high(x), high(y)) = {x2, x3, y2, y3}.
func hi64(x, y vec) vec {
	return x.ToBits().ReshapeToUint64s().InterleaveHi(y.ToBits().ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}

// rev4 is {x3, x2, x1, x0}: swap64(rev64(x)), written without calls so that
// no line of it is a call alone (fft_soa_simd.go).
func rev4(x vec) vec {
	u := x.ToBits()
	b := u.InterleaveOdd(u.InterleaveEven(u)).ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint32s().BitsToFloat32()
}

// revEven is {hi2, hi0, lo2, lo0}; revOdd is {hi3, hi1, lo3, lo1}: the even
// (odd) lanes of lo followed by hi, reversed.
func revEven(hi, lo vec) vec { return even(swap64(hi), swap64(lo)) }

func revOdd(hi, lo vec) vec { return odd(swap64(hi), swap64(lo)) }

// ext3 is {x3, y0, y1, y2} (EXT #12; ConcatShiftBytesRight shifts its
// argument down and its receiver in).
func ext3(x, y vec) vec {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return yb.ConcatShiftBytesRight(xb, 12).ReshapeToUint32s().BitsToFloat32()
}

// loHi is {x0, x1, y2, y3}: EXT #8 gives {y2, y3, x0, x1}.
func loHi(x, y vec) vec {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return swap64(xb.ConcatShiftBytesRight(yb, 8).ReshapeToUint32s().BitsToFloat32())
}

// neg is the negation of every lane.
func neg(x vec) vec { return x.Neg() }

// soaLane0..3 are the transposition of r0..r3: soaLanek is {r0[k], r1[k],
// r2[k], r3[k]}. TRN1 of two vectors is {x0, y0, x2, y2} and TRN2 is {x1,
// y1, x3, y3}; the lanes are the low or the high halves of two of them.
func soaLane0(r0, r1, r2, r3 vec) vec {
	return r0.ToBits().InterleaveEven(r1.ToBits()).ReshapeToUint64s().InterleaveLo(r2.ToBits().InterleaveEven(r3.ToBits()).ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}

func soaLane1(r0, r1, r2, r3 vec) vec {
	return r0.ToBits().InterleaveOdd(r1.ToBits()).ReshapeToUint64s().InterleaveLo(r2.ToBits().InterleaveOdd(r3.ToBits()).ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}

func soaLane2(r0, r1, r2, r3 vec) vec {
	return r0.ToBits().InterleaveEven(r1.ToBits()).ReshapeToUint64s().InterleaveHi(r2.ToBits().InterleaveEven(r3.ToBits()).ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}

func soaLane3(r0, r1, r2, r3 vec) vec {
	return r0.ToBits().InterleaveOdd(r1.ToBits()).ReshapeToUint64s().InterleaveHi(r2.ToBits().InterleaveOdd(r3.ToBits()).ReshapeToUint64s()).ReshapeToUint32s().BitsToFloat32()
}
