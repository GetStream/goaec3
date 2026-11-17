//go:build webrtc_oracle || (ooura_sse2_oracle && amd64)

package webrtcoracle

/*
#include "ooura_kern_oracle.h"
*/
import "C"

import "unsafe"

func cfloats(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

// KernSpectrum runs FftData::Spectrum(opt) on 65-bin re/im arrays.
func KernSpectrum(opt int, re, im, out *[65]float32) {
	C.oracle_kern_spectrum(C.int(opt), (*C.float)(unsafe.Pointer(&re[0])),
		(*C.float)(unsafe.Pointer(&im[0])), (*C.float)(unsafe.Pointer(&out[0])))
}

// KernSqrt runs VectorMath(opt).Sqrt in place.
func KernSqrt(opt int, x []float32) { C.oracle_kern_sqrt(C.int(opt), cfloats(x), C.int(len(x))) }

// KernMultiply runs VectorMath(opt).Multiply(x, y, z).
func KernMultiply(opt int, x, y, z []float32) {
	C.oracle_kern_multiply(C.int(opt), cfloats(x), cfloats(y), cfloats(z), C.int(len(x)))
}

// KernAccumulate runs VectorMath(opt).Accumulate(x, z).
func KernAccumulate(opt int, x, z []float32) {
	C.oracle_kern_accumulate(C.int(opt), cfloats(x), cfloats(z), C.int(len(x)))
}
