//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// archsimd's 128-bit float ops on amd64 are VEX-encoded (AVX). They compute
// the same IEEE results as the SSE2 instructions libwebrtc uses; CPUs
// without AVX run the pure-Go SSE2 reference.
var useSimd = archsimd.X86.AVX()

func fft(a *[128]float32) {
	if useSimd {
		fftSimd(a)
	} else {
		fftSSE2Ref(a)
	}
}

func inverseFft(a *[128]float32) {
	if useSimd {
		inverseFftSimd(a)
	} else {
		inverseFftSSE2Ref(a)
	}
}

// rev64 is {x1, x0, x3, x2} (PSHUFD).
func rev64(x vec) vec { return x.ToBits().PermuteScalars(1, 0, 3, 2).BitsToFloat32() }

// swap64 is {x2, x3, x0, x1} (PSHUFD).
func swap64(x vec) vec { return x.ToBits().PermuteScalars(2, 3, 0, 1).BitsToFloat32() }

// even is {x0, x2, y0, y2}; odd is {x1, x3, y1, y3} (SHUFPS).
func even(x, y vec) vec { return x.ConcatPermuteScalars(0, 2, 4, 6, y) }

func odd(x, y vec) vec { return x.ConcatPermuteScalars(1, 3, 5, 7, y) }
