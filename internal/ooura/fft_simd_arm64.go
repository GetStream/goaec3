//go:build goexperiment.simd && arm64 && !purego

package ooura

func fft(a *[128]float32)        { fftSimd(a) }
func inverseFft(a *[128]float32) { inverseFftSimd(a) }

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
