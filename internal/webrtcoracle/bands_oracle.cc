//go:build webrtc_oracle

#include "bands_oracle.h"

#include <algorithm>
#include <memory>

#include "api/array_view.h"
#include "api/audio/audio_processing.h"
#include "common_audio/include/audio_util.h"
#include "common_audio/signal_processing/include/signal_processing_library.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/three_band_filter_bank.h"

using webrtc::ThreeBandFilterBank;

extern "C" void* bands_tbfb_new(void) { return new ThreeBandFilterBank(); }

extern "C" void bands_tbfb_analysis(void* p, const float* in480, float* out) {
  webrtc::ArrayView<float> views[3] = {
      webrtc::ArrayView<float>(out, 160),
      webrtc::ArrayView<float>(out + 160, 160),
      webrtc::ArrayView<float>(out + 320, 160)};
  static_cast<ThreeBandFilterBank*>(p)->Analysis(
      webrtc::ArrayView<const float, ThreeBandFilterBank::kFullBandSize>(
          in480, ThreeBandFilterBank::kFullBandSize),
      webrtc::ArrayView<const webrtc::ArrayView<float>,
                        ThreeBandFilterBank::kNumBands>(views, 3));
}

extern "C" void bands_tbfb_synthesis(void* p, const float* in, float* out480) {
  float* mut = const_cast<float*>(in);
  webrtc::ArrayView<float> views[3] = {
      webrtc::ArrayView<float>(mut, 160),
      webrtc::ArrayView<float>(mut + 160, 160),
      webrtc::ArrayView<float>(mut + 320, 160)};
  static_cast<ThreeBandFilterBank*>(p)->Synthesis(
      webrtc::ArrayView<const webrtc::ArrayView<float>,
                        ThreeBandFilterBank::kNumBands>(views, 3),
      webrtc::ArrayView<float, ThreeBandFilterBank::kFullBandSize>(
          out480, ThreeBandFilterBank::kFullBandSize));
}

extern "C" void bands_tbfb_free(void* p) {
  delete static_cast<ThreeBandFilterBank*>(p);
}

extern "C" void bands_qmf_analysis(const int16_t* in, size_t in_len,
                                   int16_t* low, int16_t* high,
                                   int32_t* state1, int32_t* state2) {
  WebRtcSpl_AnalysisQMF(in, in_len, low, high, state1, state2);
}

extern "C" void bands_qmf_synthesis(const int16_t* low, const int16_t* high,
                                    size_t band_len, int16_t* out,
                                    int32_t* state1, int32_t* state2) {
  WebRtcSpl_SynthesisQMF(low, high, band_len, out, state1, state2);
}

extern "C" void bands_float_s16_to_s16(const float* src, size_t n,
                                       int16_t* dst) {
  webrtc::FloatS16ToS16(src, n, dst);
}

extern "C" void bands_s16_to_float_s16(const int16_t* src, size_t n,
                                       float* dst) {
  webrtc::S16ToFloatS16(src, n, dst);
}

extern "C" void bands_float_to_float_s16(const float* src, size_t n,
                                         float* dst) {
  webrtc::FloatToFloatS16(src, n, dst);
}

extern "C" void bands_float_s16_to_float(const float* src, size_t n,
                                         float* dst) {
  webrtc::FloatS16ToFloat(src, n, dst);
}

namespace {
struct Ab {
  std::unique_ptr<webrtc::AudioBuffer> buf;
  int n;
  int num_ch;
};
}  // namespace

extern "C" void* bands_ab_new(int rate, int num_ch) {
  return new Ab{std::make_unique<webrtc::AudioBuffer>(rate, num_ch, rate,
                                                      num_ch, rate, num_ch),
                rate / 100, num_ch};
}

extern "C" void bands_ab_copy_from(void* p, const float* data) {
  auto* a = static_cast<Ab*>(p);
  const float* chans[8];
  for (int ch = 0; ch < a->num_ch; ++ch) chans[ch] = data + ch * a->n;
  a->buf->CopyFrom(chans, webrtc::StreamConfig(a->n * 100, a->num_ch));
}

extern "C" void bands_ab_split(void* p) {
  static_cast<Ab*>(p)->buf->SplitIntoFrequencyBands();
}

extern "C" void bands_ab_merge(void* p) {
  static_cast<Ab*>(p)->buf->MergeFrequencyBands();
}

extern "C" void bands_ab_read_bands(void* p, float* bands) {
  auto* a = static_cast<Ab*>(p);
  const size_t nb = a->buf->num_bands();
  const size_t fpb = a->buf->num_frames_per_band();
  for (size_t ch = 0; ch < static_cast<size_t>(a->num_ch); ++ch) {
    const float* const* b = a->buf->split_bands_const(ch);
    for (size_t k = 0; k < nb; ++k)
      std::copy(b[k], b[k] + fpb, bands + (k * a->num_ch + ch) * fpb);
  }
}

extern "C" void bands_ab_write_bands(void* p, const float* bands) {
  auto* a = static_cast<Ab*>(p);
  const size_t nb = a->buf->num_bands();
  const size_t fpb = a->buf->num_frames_per_band();
  for (size_t ch = 0; ch < static_cast<size_t>(a->num_ch); ++ch) {
    float* const* b = a->buf->split_bands(ch);
    for (size_t k = 0; k < nb; ++k)
      std::copy(bands + (k * a->num_ch + ch) * fpb,
                bands + (k * a->num_ch + ch + 1) * fpb, b[k]);
  }
}

extern "C" void bands_ab_read_data(void* p, float* data) {
  auto* a = static_cast<Ab*>(p);
  for (int ch = 0; ch < a->num_ch; ++ch) {
    const float* c = a->buf->channels_const()[ch];
    std::copy(c, c + a->n, data + ch * a->n);
  }
}

extern "C" void bands_ab_free(void* p) { delete static_cast<Ab*>(p); }
