//go:build webrtc_oracle

// From-source build of suppression_filter.cc and aec3_fft.cc on top of the
// scalar Ooura FFT (OouraFftScalarRef, defined in ooura_scalar.cc), under
// renamed symbols. This is the scalar reference for the purego lane.
#pragma clang fp contract(off)
// libwebrtc's GN build defines WEBRTC_ARCH_ARM64 on arm64 (it selects
// vsqrtq_f32 in VectorMath::Sqrt); the oracle cgo flags do not.
#if defined(__aarch64__) && !defined(WEBRTC_ARCH_ARM64)
#define WEBRTC_ARCH_ARM64
#endif
#define OouraFft OouraFftScalarRef
#define Aec3Fft Aec3FftSupScalar
#define SuppressionFilter SuppressionFilterSupScalar
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_fft.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/suppression_filter.cc"

#include "sup_oracle.h"

extern "C" void* sup_sf_scalar_new(int optimization, int rate, int num_ch) {
  return new webrtc::SuppressionFilter(
      static_cast<webrtc::Aec3Optimization>(optimization), rate, num_ch);
}

extern "C" void sup_sf_scalar_apply(void* p, const float* cn,
                                    const float* cn_high, const float* gain,
                                    float high_bands_gain, const float* E,
                                    float* e, int num_bands, int num_ch) {
  using Spectrum = std::array<float, webrtc::kFftLengthBy2Plus1>;
  webrtc::Block block(num_bands, num_ch);
  for (int b = 0; b < num_bands; ++b)
    for (int ch = 0; ch < num_ch; ++ch)
      std::copy(e + (b * num_ch + ch) * 64, e + (b * num_ch + ch + 1) * 64,
                block.begin(b, ch));
  static_cast<webrtc::SuppressionFilter*>(p)->ApplyGain(
      webrtc::ArrayView<const webrtc::FftData>(
          reinterpret_cast<const webrtc::FftData*>(cn), num_ch),
      webrtc::ArrayView<const webrtc::FftData>(
          reinterpret_cast<const webrtc::FftData*>(cn_high), num_ch),
      *reinterpret_cast<const Spectrum*>(gain), high_bands_gain,
      webrtc::ArrayView<const webrtc::FftData>(
          reinterpret_cast<const webrtc::FftData*>(E), num_ch),
      &block);
  for (int b = 0; b < num_bands; ++b)
    for (int ch = 0; ch < num_ch; ++ch)
      std::copy(block.begin(b, ch), block.end(b, ch),
                e + (b * num_ch + ch) * 64);
}

extern "C" void sup_sf_scalar_free(void* p) {
  delete static_cast<webrtc::SuppressionFilter*>(p);
}
