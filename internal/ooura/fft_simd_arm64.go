//go:build goexperiment.simd && arm64 && !purego

package ooura

// useSimd reports whether the archsimd kernels can run: NEON is always
// present on arm64.
const useSimd = true

func initSimdTables() { initSoaTables() }

func fftSplit(a *[128]float32, re, im *[65]float32, s *Scratch) { fftSplitSoa(a, re, im, &s.t) }

func inverseFftSplit(re, im *[65]float32, a *[128]float32, s *Scratch) {
	inverseFftSplitSoa(re, im, a, &s.t)
}

// even is {x0, x2, y0, y2} (UZP1); odd is {x1, x3, y1, y3} (UZP2).
func even(x, y vec) vec { return x.ToBits().ConcatEven(y.ToBits()).BitsToFloat32() }

func odd(x, y vec) vec { return x.ToBits().ConcatOdd(y.ToBits()).BitsToFloat32() }

// rev4 is {x3, x2, x1, x0}: the pairs exchanged (TRN2 of x and TRN1(x, x)),
// then the halves (EXT #8).
func rev4(x vec) vec {
	u := x.ToBits()
	b := u.InterleaveOdd(u.InterleaveEven(u)).ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint32s().BitsToFloat32()
}

// ext3 is {x3, y0, y1, y2} (EXT #12; ConcatShiftBytesRight shifts its
// argument down and its receiver in).
func ext3(x, y vec) vec {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return yb.ConcatShiftBytesRight(xb, 12).ReshapeToUint32s().BitsToFloat32()
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
