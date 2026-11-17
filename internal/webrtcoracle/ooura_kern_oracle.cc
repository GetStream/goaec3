//go:build webrtc_oracle

// FftData::Spectrum and VectorMath are header-only; compiled here with the
// same flags as libwebrtc (NEON, contraction off). VectorMath::Sqrt needs
// WEBRTC_ARCH_ARM64 (set package-wide in oracle.go, as libwebrtc's GN build
// does) to take the vsqrtq_f32 path the prebuilt objects use.
#pragma clang fp contract(off)
#include "ooura_kern_impl.h"
