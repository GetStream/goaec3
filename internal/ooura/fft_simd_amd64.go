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
