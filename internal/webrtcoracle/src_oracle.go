//go:build webrtc_src_oracle && !webrtc_oracle

// Package webrtcoracle is the cgo test oracle of goaec3: it exposes the C++
// AEC3 of libwebrtc m141 to the parity tests. The webrtc_oracle tag links the
// prebuilt libwebrtc.a (oracle.go); the webrtc_src_oracle tag compiles the
// reference from the WebRTC sources (src_oracle.go). No shipped package
// imports it.
package webrtcoracle

// The source build compiles AEC3, AudioBuffer's band splitting, the high-pass
// filter and their common_audio and rtc_base dependencies from
// tmp_check/webrtc-m141 (scripts/fetch-webrtc.sh) on darwin and linux, arm64
// and amd64. Logging, metrics, field trials (all disabled), the Environment,
// ApmDataDumper (WEBRTC_APM_DEBUG_DUMP=0) and cpu_info are the only replaced
// pieces (src_stubs.cc).
//
// The C++ lane follows the build as in libwebrtc's GN build: NEON on arm64
// (WEBRTC_HAS_NEON), and on amd64 the run-time cpu_info selection of AVX2 or
// SSE2, compiled with the AVX2/FMA kernels of aec3. cpu_info answers with the
// lane aec3.DetectOptimization reports, so GODEBUG=cpu.avx2=off moves both
// sides to SSE2. The purego tag builds the scalar reference
// (Aec3Optimization::kNone, scalar Ooura). TestMain (src_main_test.go) checks
// that both sides run the same lane.
//
// The ec3 parity tests and their Go wrappers (ec3_*.go) are shared with the
// webrtc_oracle build; the translation units and the instances of the C++
// shim are generated (src_gen_*, scripts/gen-src-oracle.sh).
//
// Every translation unit starts with src_prelude.h, which requires clang and
// turns floating-point contraction off for the whole unit, as libwebrtc's
// -ffp-contract=off does (cgo does not accept that flag without
// CGO_CFLAGS_ALLOW).

/*
#cgo CFLAGS: -DNDEBUG -DWEBRTC_POSIX -DWEBRTC_ENABLE_AVX2 -I${SRCDIR}/../../tmp_check/webrtc-m141
#cgo CXXFLAGS: -std=c++20 -DNDEBUG -DWEBRTC_POSIX -DWEBRTC_ENABLE_AVX2 -DWEBRTC_APM_DEBUG_DUMP=0 -DRTC_DISABLE_LOGGING -DRTC_DISABLE_METRICS -DRTC_DISABLE_TRACE_EVENTS -Wno-nullability-completeness -I${SRCDIR}/../../tmp_check/webrtc-m141 -I${SRCDIR}/../../tmp_check/webrtc-m141/third_party/abseil-cpp
#cgo darwin CFLAGS: -DWEBRTC_MAC
#cgo darwin CXXFLAGS: -DWEBRTC_MAC
#cgo darwin LDFLAGS: -lc++
#cgo linux CFLAGS: -DWEBRTC_LINUX
#cgo linux CXXFLAGS: -DWEBRTC_LINUX
#cgo linux LDFLAGS: -lstdc++ -lm
#include "src_oracle.h"
*/
import "C"

import "github.com/GetStream/goaec3/internal/aec3"

func init() { SrcSyncLane() }

// SrcSyncLane makes cpu_info::Supports in the C++ build answer from the
// lane aec3.DetectOptimization() returns at the time of the call; call it
// again after aec3.SetForcedLaneForTesting.
func SrcSyncLane() {
	C.goaec3_src_set_lane(C.int(aec3.DetectOptimization()))
}

// SrcLane returns the lane the C++ build runs: webrtc::DetectOptimization().
func SrcLane() aec3.Aec3Optimization {
	return aec3.Aec3Optimization(C.goaec3_src_detect_optimization())
}

// SrcCPUInfoLane returns the lane libwebrtc's own CPU detection
// (rtc_base/cpu_info.cc, compiled unmodified) selects on this machine,
// following aec3_common.cc's DetectOptimization.
func SrcCPUInfoLane() aec3.Aec3Optimization {
	return aec3.Aec3Optimization(C.goaec3_src_cpu_info_lane())
}

// SrcCPUHasFMA reports whether CPUID advertises FMA (always false off x86).
func SrcCPUHasFMA() bool { return C.goaec3_src_cpu_has_fma() != 0 }
