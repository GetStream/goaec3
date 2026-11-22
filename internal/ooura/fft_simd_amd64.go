//go:build goexperiment.simd && amd64 && !purego

package ooura

import "simd/archsimd"

// useSimd reports whether the archsimd kernels can run. Their 128-bit
// operations are VEX-encoded and need AVX; they compute what the SSE2
// instructions of libwebrtc compute. A CPU without AVX runs fft_fast.go.
var useSimd = archsimd.X86.AVX()

func fftSplit(a *[128]float32, re, im *[65]float32, s *Scratch) {
	if useSimd {
		fftSplitSoa(a, re, im, &s.t)
	} else {
		fftSplitFast(a, re, im)
	}
}

func inverseFftSplit(re, im *[65]float32, a *[128]float32, s *Scratch) {
	if useSimd {
		inverseFftSplitSoa(re, im, a, &s.t)
	} else {
		inverseFftSplitFast(re, im, a)
	}
}

// The shuffles are SHUFPS, which stays in the floating-point domain: an
// integer shuffle between float operations costs a bypass delay on some
// cores.

// even is {x0, x2, y0, y2}; odd is {x1, x3, y1, y3} (SHUFPS).
func even(x, y vec) vec { return x.ConcatPermuteScalars(0, 2, 4, 6, y) }

func odd(x, y vec) vec { return x.ConcatPermuteScalars(1, 3, 5, 7, y) }

// rev4 is {x3, x2, x1, x0}.
func rev4(x vec) vec { return x.ConcatPermuteScalars(3, 2, 5, 4, x) }

// ext3 is {x3, y0, y1, y2} (VPALIGNR; ConcatShiftBytesRight shifts its
// argument down and its receiver in).
func ext3(x, y vec) vec {
	xb, yb := x.ToBits().ReshapeToUint8s(), y.ToBits().ReshapeToUint8s()
	return yb.ConcatShiftBytesRight(xb, 12).ReshapeToUint32s().BitsToFloat32()
}

// neg negates every lane. Float32x4.Neg broadcasts its mask with an AVX2
// instruction, so the mask is loaded.
func neg(x vec) vec {
	return x.ToBits().Xor(archsimd.LoadUint32x4Array(&signAll)).BitsToFloat32()
}

var signAll = [4]uint32{0x80000000, 0x80000000, 0x80000000, 0x80000000}

// soaLane0..3 are the transposition of r0..r3: soaLanek is {r0[k], r1[k],
// r2[k], r3[k]}.
func soaLane0(r0, r1, r2, r3 vec) vec {
	return r0.ConcatPermuteScalars(0, 1, 4, 5, r1).ConcatPermuteScalars(0, 2, 4, 6, r2.ConcatPermuteScalars(0, 1, 4, 5, r3))
}

func soaLane1(r0, r1, r2, r3 vec) vec {
	return r0.ConcatPermuteScalars(0, 1, 4, 5, r1).ConcatPermuteScalars(1, 3, 5, 7, r2.ConcatPermuteScalars(0, 1, 4, 5, r3))
}

func soaLane2(r0, r1, r2, r3 vec) vec {
	return r0.ConcatPermuteScalars(2, 3, 6, 7, r1).ConcatPermuteScalars(0, 2, 4, 6, r2.ConcatPermuteScalars(2, 3, 6, 7, r3))
}

func soaLane3(r0, r1, r2, r3 vec) vec {
	return r0.ConcatPermuteScalars(2, 3, 6, 7, r1).ConcatPermuteScalars(1, 3, 5, 7, r2.ConcatPermuteScalars(2, 3, 6, 7, r3))
}
