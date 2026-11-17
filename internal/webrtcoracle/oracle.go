//go:build webrtc_oracle

// Package webrtcoracle links the pinned prebuilt libwebrtc (m141) and exposes
// AEC3 entry points to parity tests. It builds only with the webrtc_oracle tag.
package webrtcoracle

/*
#cgo CXXFLAGS: -std=c++20 -DWEBRTC_POSIX -DWEBRTC_MAC -DWEBRTC_HAS_NEON -I${SRCDIR}/../../tmp_check/libwebrtc/include -I${SRCDIR}/../../tmp_check/libwebrtc/include/third_party/abseil-cpp
#cgo LDFLAGS: ${SRCDIR}/../../tmp_check/libwebrtc/lib/libwebrtc.a -lc++ -framework Foundation -framework CoreAudio -framework AudioToolbox -framework CoreMedia -framework CoreVideo -framework AVFoundation -framework AppKit -framework IOSurface -framework Metal -framework VideoToolbox -framework ApplicationServices
#include "oracle.h"
*/
import "C"

import "unsafe"

// OouraFft runs webrtc::OouraFft in place on a 128-sample buffer.
func OouraFft(a *[128]float32, inverse bool) {
	inv := C.int(0)
	if inverse {
		inv = 1
	}
	C.oracle_ooura_fft((*C.float)(unsafe.Pointer(&a[0])), inv, 0)
}

// OouraFftScalar runs the scalar C path of ooura_fft.cc compiled from source.
func OouraFftScalar(a *[128]float32, inverse bool) {
	inv := C.int(0)
	if inverse {
		inv = 1
	}
	C.oracle_ooura_fft_scalar((*C.float)(unsafe.Pointer(&a[0])), inv)
}
