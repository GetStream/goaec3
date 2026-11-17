// Shared body of the FftData::Spectrum / VectorMath shims, included by
// ooura_kern_oracle.cc (arm64, NEON) and ooura_sse2_oracle.cc (x86_64).
#ifndef GOAEC3_OOURA_KERN_IMPL_H
#define GOAEC3_OOURA_KERN_IMPL_H
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include <algorithm>

#include "ooura_kern_oracle.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/vector_math.h"

extern "C" void oracle_kern_spectrum(int opt, const float* re, const float* im, float* out) {
  webrtc::FftData d;
  std::copy(re, re + webrtc::kFftLengthBy2Plus1, d.re.begin());
  std::copy(im, im + webrtc::kFftLengthBy2Plus1, d.im.begin());
  d.Spectrum(static_cast<webrtc::Aec3Optimization>(opt),
             webrtc::ArrayView<float>(out, webrtc::kFftLengthBy2Plus1));
}

extern "C" void oracle_kern_sqrt(int opt, float* x, int n) {
  webrtc::aec3::VectorMath(static_cast<webrtc::Aec3Optimization>(opt))
      .Sqrt(webrtc::ArrayView<float>(x, n));
}

extern "C" void oracle_kern_multiply(int opt, const float* x, const float* y, float* z, int n) {
  webrtc::aec3::VectorMath(static_cast<webrtc::Aec3Optimization>(opt))
      .Multiply(webrtc::ArrayView<const float>(x, n), webrtc::ArrayView<const float>(y, n),
                webrtc::ArrayView<float>(z, n));
}

extern "C" void oracle_kern_accumulate(int opt, const float* x, float* z, int n) {
  webrtc::aec3::VectorMath(static_cast<webrtc::Aec3Optimization>(opt))
      .Accumulate(webrtc::ArrayView<const float>(x, n), webrtc::ArrayView<float>(z, n));
}
#endif
