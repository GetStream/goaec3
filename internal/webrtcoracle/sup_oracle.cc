//go:build webrtc_oracle

#include "sup_oracle.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/comfort_noise_generator.h"
#include "modules/audio_processing/aec3/config_selector.h"
#include "modules/audio_processing/aec3/dominant_nearend_detector.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/multi_channel_content_detector.h"
#include "modules/audio_processing/aec3/nearend_detector.h"
#include "modules/audio_processing/aec3/subband_nearend_detector.h"
#include "modules/audio_processing/aec3/suppression_filter.h"

namespace {
using Spectrum = std::array<float, webrtc::kFftLengthBy2Plus1>;
static_assert(sizeof(webrtc::FftData) == 2 * 65 * sizeof(float));
static_assert(sizeof(Spectrum) == 65 * sizeof(float));

struct Cng {
  webrtc::ComfortNoiseGenerator g;
  int num_ch;
};
}  // namespace

extern "C" void* sup_cng_new(float noise_floor_dbfs, int optimization,
                             int num_ch) {
  webrtc::EchoCanceller3Config config;
  config.comfort_noise.noise_floor_dbfs = noise_floor_dbfs;
  return new Cng{webrtc::ComfortNoiseGenerator(
                     config, static_cast<webrtc::Aec3Optimization>(optimization),
                     num_ch),
                 num_ch};
}

extern "C" void sup_cng_compute(void* p, int saturated, const float* Y2,
                                float* low, float* high, float* N2_out) {
  auto* c = static_cast<Cng*>(p);
  c->g.Compute(saturated != 0,
               webrtc::ArrayView<const Spectrum>(
                   reinterpret_cast<const Spectrum*>(Y2), c->num_ch),
               webrtc::ArrayView<webrtc::FftData>(
                   reinterpret_cast<webrtc::FftData*>(low), c->num_ch),
               webrtc::ArrayView<webrtc::FftData>(
                   reinterpret_cast<webrtc::FftData*>(high), c->num_ch));
  auto n2 = c->g.NoiseSpectrum();
  for (int ch = 0; ch < c->num_ch; ++ch) {
    std::copy(n2[ch].begin(), n2[ch].end(), N2_out + ch * 65);
  }
}

extern "C" void sup_cng_free(void* p) { delete static_cast<Cng*>(p); }

extern "C" void* sup_sf_new(int optimization, int rate, int num_ch,
                            int scalar) {
  if (scalar) return sup_sf_scalar_new(optimization, rate, num_ch);
  return new webrtc::SuppressionFilter(
      static_cast<webrtc::Aec3Optimization>(optimization), rate, num_ch);
}

extern "C" void sup_sf_apply(void* p, const float* cn, const float* cn_high,
                             const float* gain, float high_bands_gain,
                             const float* E, float* e, int num_bands,
                             int num_ch, int scalar) {
  if (scalar) {
    sup_sf_scalar_apply(p, cn, cn_high, gain, high_bands_gain, E, e, num_bands,
                        num_ch);
    return;
  }
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

extern "C" void sup_sf_free(void* p, int scalar) {
  if (scalar) {
    sup_sf_scalar_free(p);
    return;
  }
  delete static_cast<webrtc::SuppressionFilter*>(p);
}

extern "C" void* sup_dne_new(float enr_threshold, float enr_exit_threshold,
                             float snr_threshold, int hold_duration,
                             int trigger_threshold,
                             int use_during_initial_phase, int num_ch) {
  webrtc::EchoCanceller3Config::Suppressor::DominantNearendDetection c;
  c.enr_threshold = enr_threshold;
  c.enr_exit_threshold = enr_exit_threshold;
  c.snr_threshold = snr_threshold;
  c.hold_duration = hold_duration;
  c.trigger_threshold = trigger_threshold;
  c.use_during_initial_phase = use_during_initial_phase != 0;
  webrtc::NearendDetector* d = new webrtc::DominantNearendDetector(c, num_ch);
  return d;
}

extern "C" void* sup_sne_new(int nearend_average_blocks, int s1_low,
                             int s1_high, int s2_low, int s2_high,
                             float nearend_threshold, float snr_threshold,
                             int num_ch) {
  webrtc::EchoCanceller3Config::Suppressor::SubbandNearendDetection c;
  c.nearend_average_blocks = nearend_average_blocks;
  c.subband1 = {static_cast<size_t>(s1_low), static_cast<size_t>(s1_high)};
  c.subband2 = {static_cast<size_t>(s2_low), static_cast<size_t>(s2_high)};
  c.nearend_threshold = nearend_threshold;
  c.snr_threshold = snr_threshold;
  webrtc::NearendDetector* d = new webrtc::SubbandNearendDetector(c, num_ch);
  return d;
}

extern "C" int sup_ne_update(void* p, const float* nearend, const float* echo,
                             const float* noise, int num_ch,
                             int initial_state) {
  auto* d = static_cast<webrtc::NearendDetector*>(p);
  d->Update(webrtc::ArrayView<const Spectrum>(
                reinterpret_cast<const Spectrum*>(nearend), num_ch),
            webrtc::ArrayView<const Spectrum>(
                reinterpret_cast<const Spectrum*>(echo), num_ch),
            webrtc::ArrayView<const Spectrum>(
                reinterpret_cast<const Spectrum*>(noise), num_ch),
            initial_state != 0);
  return d->IsNearendState() ? 1 : 0;
}

extern "C" void sup_ne_free(void* p) {
  delete static_cast<webrtc::NearendDetector*>(p);
}

extern "C" void* sup_mccd_new(int detect_stereo_content, int num_render_ch,
                              float detection_threshold, int timeout_seconds,
                              float hysteresis_seconds) {
  return new webrtc::MultiChannelContentDetector(
      detect_stereo_content != 0, num_render_ch, detection_threshold,
      timeout_seconds, hysteresis_seconds);
}

extern "C" int sup_mccd_update(void* p, const float* frame, int num_bands,
                               int num_ch, int n, int* proper,
                               int* temporary) {
  auto* d = static_cast<webrtc::MultiChannelContentDetector*>(p);
  std::vector<std::vector<std::vector<float>>> f(
      num_bands, std::vector<std::vector<float>>(num_ch));
  for (int b = 0; b < num_bands; ++b)
    for (int ch = 0; ch < num_ch; ++ch)
      f[b][ch].assign(frame + (b * num_ch + ch) * n,
                      frame + (b * num_ch + ch + 1) * n);
  int changed = d->UpdateDetection(f) ? 1 : 0;
  *proper = d->IsProperMultiChannelContentDetected() ? 1 : 0;
  *temporary = d->IsTemporaryMultiChannelContentDetected() ? 1 : 0;
  return changed;
}

extern "C" void sup_mccd_free(void* p) {
  delete static_cast<webrtc::MultiChannelContentDetector*>(p);
}

extern "C" void* sup_cs_new(int detect_stereo_content, int has_multichannel,
                            int mc_marker, int num_render_ch) {
  webrtc::EchoCanceller3Config mono;
  mono.multi_channel.detect_stereo_content = detect_stereo_content != 0;
  std::optional<webrtc::EchoCanceller3Config> mc;
  if (has_multichannel) {
    mc = mono;
    mc->filter.refined.length_blocks = mc_marker;
  }
  return new webrtc::ConfigSelector(mono, mc, num_render_ch);
}

extern "C" void sup_cs_update(void* p, int multichannel_content) {
  static_cast<webrtc::ConfigSelector*>(p)->Update(multichannel_content != 0);
}

extern "C" int sup_cs_active_marker(void* p) {
  return static_cast<int>(static_cast<webrtc::ConfigSelector*>(p)
                              ->active_config()
                              .filter.refined.length_blocks);
}

extern "C" void sup_cs_free(void* p) {
  delete static_cast<webrtc::ConfigSelector*>(p);
}
