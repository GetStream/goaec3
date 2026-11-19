//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// archsimd's 128-bit float ops on amd64 are VEX-encoded (AVX). They compute
// the same IEEE results as the SSE2 instructions libwebrtc uses; CPUs
// without AVX run the pure-Go lane code (fft_fast.go).
var useSimd = archsimd.X86.AVX()

func fft(a *[128]float32) {
	if useSimd {
		fftSimd(a)
	} else {
		fftFast(a)
	}
}

func inverseFft(a *[128]float32) {
	if useSimd {
		inverseFftSimd(a)
	} else {
		inverseFftFast(a)
	}
}

func fftSplit(a *[128]float32, re, im *[65]float32, s *Scratch) {
	if useSimd {
		fftSplitSimd(a, re, im, &s.t)
	} else {
		fftSplitFast(a, re, im)
	}
}

func inverseFftSplit(re, im *[65]float32, a *[128]float32, s *Scratch) {
	if useSimd {
		inverseFftSplitSimd(re, im, a, &s.t)
	} else {
		inverseFftSplitFast(re, im, a)
	}
}

// The shuffles below are SHUFPS, which stays in the floating-point domain
// (an integer shuffle such as PSHUFD or PUNPCKLQDQ between float operations
// costs a bypass delay on some cores).

// rev64 is {x1, x0, x3, x2}.
func rev64(x vec) vec { return x.ConcatPermuteScalars(1, 0, 7, 6, x) }

// swap64 is {x2, x3, x0, x1}.
func swap64(x vec) vec { return x.ConcatPermuteScalars(2, 3, 4, 5, x) }

// lo64 is {x0, x1, y0, y1}.
func lo64(x, y vec) vec { return x.ConcatPermuteScalars(0, 1, 4, 5, y) }

// hi64 is {x2, x3, y2, y3}.
func hi64(x, y vec) vec { return x.ConcatPermuteScalars(2, 3, 6, 7, y) }

// even is {x0, x2, y0, y2}; odd is {x1, x3, y1, y3} (SHUFPS).
func even(x, y vec) vec { return x.ConcatPermuteScalars(0, 2, 4, 6, y) }

func odd(x, y vec) vec { return x.ConcatPermuteScalars(1, 3, 5, 7, y) }

// rev4 is {x3, x2, x1, x0}.
func rev4(x vec) vec { return x.ConcatPermuteScalars(3, 2, 5, 4, x) }

// revEven is {hi2, hi0, lo2, lo0}; revOdd is {hi3, hi1, lo3, lo1}: the even
// (odd) lanes of lo followed by hi, reversed.
func revEven(hi, lo vec) vec { return hi.ConcatPermuteScalars(2, 0, 6, 4, lo) }

func revOdd(hi, lo vec) vec { return hi.ConcatPermuteScalars(3, 1, 7, 5, lo) }

// loHi is {x0, x1, y2, y3}.
func loHi(x, y vec) vec { return x.ConcatPermuteScalars(0, 1, 6, 7, y) }

// ext3 is {x3, y0, y1, y2} (VPALIGNR; ConcatShiftBytesRight shifts its
// argument down and its receiver in).
func ext3(x, y vec) vec {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return yb.ConcatShiftBytesRight(xb, 12).ReshapeToUint32s().BitsToFloat32()
}
