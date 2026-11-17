//go:build webrtc_oracle

// Scalar (non-NEON) reference build of ooura_fft.cc, renamed to avoid
// clashing with the NEON build linked from libwebrtc.a.
#pragma clang fp contract(off)
#undef WEBRTC_HAS_NEON
#define OouraFft OouraFftScalarRef
#include "../../tmp_check/webrtc-m141/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc"
#undef OouraFft
extern "C" void oracle_ooura_fft_scalar(float* a, int inverse) {
  webrtc::OouraFftScalarRef fft(false);
  if (inverse) fft.InverseFft(a); else fft.Fft(a);
}
