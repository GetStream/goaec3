//go:build webrtc_oracle

// FftData::Spectrum and VectorMath are header-only; compiled here with the
// same flags as libwebrtc (NEON, contraction off).
//
// libwebrtc's GN build defines WEBRTC_ARCH_ARM64 on arm64 (rtc_base/system/
// arch.h does not); without it VectorMath::Sqrt takes the armv7
// vrsqrteq_f32 path instead of vsqrtq_f32. The prebuilt objects use fsqrt.
#pragma clang fp contract(off)
#if defined(__aarch64__) && !defined(WEBRTC_ARCH_ARM64)
#define WEBRTC_ARCH_ARM64
#endif
#include "ooura_kern_impl.h"
