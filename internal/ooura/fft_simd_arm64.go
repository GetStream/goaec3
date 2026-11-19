//go:build goexperiment.simd && arm64 && !purego

package ooura

func fft(a *[128]float32)        { fftSimd(a) }
func inverseFft(a *[128]float32) { inverseFftSimd(a) }

func fftSplit(a *[128]float32, re, im *[65]float32)        { fftSplitSimd(a, re, im) }
func inverseFftSplit(re, im *[65]float32, a *[128]float32) { inverseFftSplitSimd(re, im, a) }

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

// rev4 is {x3, x2, x1, x0}.
func rev4(x vec) vec { return swap64(rev64(x)) }

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
