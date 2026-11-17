//go:build ooura_sse2_oracle && amd64

// x86_64 oracle: there is no prebuilt amd64 libwebrtc, so the Ooura FFT
// (scalar + SSE2 paths) and the AEC3 FftData/VectorMath kernels (SSE2 +
// AVX2) are compiled from the m141 sources with contraction off, as
// libwebrtc builds them. The *_avx2.cc files are compiled with the
// avx2,fma target like their BUILD.gn target.
#pragma clang fp contract(off)

#include "rtc_base/cpu_info.h"

namespace webrtc {
namespace cpu_info {
bool Supports(ISA) { return true; }
}  // namespace cpu_info
}  // namespace webrtc

#include "../../tmp_check/webrtc-m141/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc"
#include "../../tmp_check/webrtc-m141/common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc"

extern "C" void oracle_ooura_fft_x86(float* a, int inverse, int sse2) {
  webrtc::OouraFft fft(sse2 != 0);
  if (inverse) fft.InverseFft(a); else fft.Fft(a);
}

#include "ooura_kern_impl.h"

#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/fft_data_avx2.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/vector_math_avx2.cc"
#pragma clang attribute pop
