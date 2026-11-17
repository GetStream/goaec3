//go:build webrtc_oracle

// Source build of echo_remover.cc on the prebuilt AEC3 dependencies, with
// DetectOptimization() replaced by the lane forced through erem_new. The
// prebuilt EchoRemover always runs DetectOptimization() (kNeon on arm64);
// this build is the kNone reference for the default lane. Symbols are renamed
// so they do not clash with libwebrtc.a.
#pragma clang fp contract(off)

#include "erem_shim.h"

#define DetectOptimization EremSrcForcedOptimization
#define EchoRemover EremSrcEchoRemover

namespace webrtc {
Aec3Optimization EremSrcForcedOptimization() {
  return erem_forced_lane == 1 ? Aec3Optimization::kNeon : Aec3Optimization::kNone;
}
}  // namespace webrtc

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/echo_remover.cc"

erem_shim::Base* erem_new_src(const float* f, const int* i, int rate, int nr, int nc, int size) {
  return new erem_shim::Impl<webrtc::EremSrcEchoRemover>(f, i, rate, nr, nc, size);
}
