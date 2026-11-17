//go:build ooura_sse2_oracle && amd64

package webrtcoracle

// The x86_64 oracle builds on darwin/arm64 under Rosetta:
//
//	CGO_ENABLED=1 GOARCH=amd64 CC="clang -arch x86_64" CXX="clang++ -arch x86_64" \
//	  GOEXPERIMENT=simd go test -tags ooura_sse2_oracle ./internal/webrtcoracle/
//
// It links nothing from libwebrtc.a (an arm64 archive).

/*
#cgo CXXFLAGS: -std=c++20 -DWEBRTC_POSIX -DWEBRTC_MAC -DNDEBUG -I${SRCDIR}/../../tmp_check/libwebrtc/include -I${SRCDIR}/../../tmp_check/libwebrtc/include/third_party/abseil-cpp
#cgo LDFLAGS: -lc++
#include "ooura_kern_oracle.h"
*/
import "C"

import "unsafe"

// OouraFftX86 runs webrtc::OouraFft(sse2) compiled for x86_64 in place.
func OouraFftX86(a *[128]float32, inverse, sse2 bool) {
	inv, s := C.int(0), C.int(0)
	if inverse {
		inv = 1
	}
	if sse2 {
		s = 1
	}
	C.oracle_ooura_fft_x86((*C.float)(unsafe.Pointer(&a[0])), inv, s)
}
