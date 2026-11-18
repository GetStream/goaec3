//go:build ooura_sse2_oracle && amd64

package webrtcoracle

// The x86_64 oracle compiles its kernels from the WebRTC sources in
// tmp_check/webrtc-m141 (scripts/fetch-webrtc.sh) and links nothing from
// libwebrtc. On linux/amd64:
//
//	CC=clang CXX=clang++ GOEXPERIMENT=simd go test -tags ooura_sse2_oracle ./internal/webrtcoracle/
//
// On darwin/arm64, under Rosetta 2:
//
//	CGO_ENABLED=1 GOARCH=amd64 CC="clang -arch x86_64" CXX="clang++ -arch x86_64" \
//	  GOEXPERIMENT=simd go test -tags ooura_sse2_oracle ./internal/webrtcoracle/

/*
#cgo CXXFLAGS: -std=c++20 -DWEBRTC_POSIX -DNDEBUG -I${SRCDIR}/../../tmp_check/webrtc-m141 -I${SRCDIR}/../../tmp_check/webrtc-m141/third_party/abseil-cpp
#cgo darwin CXXFLAGS: -DWEBRTC_MAC
#cgo darwin LDFLAGS: -lc++
#cgo linux CXXFLAGS: -DWEBRTC_LINUX
#cgo linux LDFLAGS: -lstdc++ -lm
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
