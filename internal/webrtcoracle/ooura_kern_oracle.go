//go:build webrtc_oracle || (ooura_sse2_oracle && amd64)

package webrtcoracle

/*
#include "ooura_kern_oracle.h"
*/
import "C"

import "unsafe"

func oouraCFloats(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

// OouraKernSpectrum runs FftData::Spectrum(opt) on 65-bin re/im arrays.
func OouraKernSpectrum(opt int, re, im, out *[65]float32) {
	C.oracle_kern_spectrum(C.int(opt), (*C.float)(unsafe.Pointer(&re[0])),
		(*C.float)(unsafe.Pointer(&im[0])), (*C.float)(unsafe.Pointer(&out[0])))
}

// OouraKernSqrt runs VectorMath(opt).Sqrt in place.
func OouraKernSqrt(opt int, x []float32) {
	C.oracle_kern_sqrt(C.int(opt), oouraCFloats(x), C.int(len(x)))
}

// OouraKernMultiply runs VectorMath(opt).Multiply(x, y, z).
func OouraKernMultiply(opt int, x, y, z []float32) {
	C.oracle_kern_multiply(C.int(opt), oouraCFloats(x), oouraCFloats(y), oouraCFloats(z), C.int(len(x)))
}

// OouraKernAccumulate runs VectorMath(opt).Accumulate(x, z).
func OouraKernAccumulate(opt int, x, z []float32) {
	C.oracle_kern_accumulate(C.int(opt), oouraCFloats(x), oouraCFloats(z), C.int(len(x)))
}
