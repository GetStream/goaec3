// Shared Subtractor simulation, instantiated for the prebuilt class
// (subt_oracle.cc) and for the scalar-FFT source build (subt_scalar.cc).
#ifndef GOAEC3_SUBT_SHIM_H
#define GOAEC3_SUBT_SHIM_H

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "subt_oracle.h"

namespace subt_shim {

inline webrtc::EchoCanceller3Config::Filter::RefinedConfiguration SubtRefined(const float* c) {
  return {static_cast<size_t>(c[0]), c[1], c[2], c[3], c[4], c[5]};
}
inline webrtc::EchoCanceller3Config::Filter::CoarseConfiguration SubtCoarse(const float* c) {
  return {static_cast<size_t>(c[0]), c[1], c[2]};
}

inline webrtc::EchoCanceller3Config SubtConfig(const subt_filter_config* c) {
  webrtc::EchoCanceller3Config cfg;
  cfg.filter.refined = SubtRefined(c->refined);
  cfg.filter.refined_initial = SubtRefined(c->refined_initial);
  cfg.filter.coarse = SubtCoarse(c->coarse);
  cfg.filter.coarse_initial = SubtCoarse(c->coarse_initial);
  cfg.filter.config_change_duration_blocks = c->config_change_duration_blocks;
  cfg.filter.initial_state_seconds = c->initial_state_seconds;
  cfg.filter.coarse_reset_hangover_blocks = c->coarse_reset_hangover_blocks;
  cfg.filter.conservative_initial_phase = c->conservative_initial_phase != 0;
  cfg.filter.enable_coarse_filter_output_usage = c->enable_coarse_filter_output_usage != 0;
  cfg.filter.use_linear_filter = c->use_linear_filter != 0;
  return cfg;
}

struct Base {
  virtual ~Base() = default;
  virtual void SetRender(int slot, int ch, const float* block64, const float* fft130, const float* spectrum65) = 0;
  virtual void SetIndices(int write, int read) = 0;
  virtual void HandleEchoPathChange(int gain_change, int delay_change) = 0;
  virtual void ExitInitialState() = 0;
  virtual void Process(const float* capture, const float* Y2, int saturated, int follow_transition,
                       subt_output* out) = 0;
  virtual void AecStateGet(subt_aec_state* out) = 0;
  virtual int FreqResp(int ch, float* H2) = 0;
  virtual int ImpulseResp(int ch, float* h) = 0;
};

template <class S>
struct Impl : Base {
  webrtc::EchoCanceller3Config cfg;
  webrtc::Environment env;
  webrtc::ApmDataDumper dumper;
  int nch;
  webrtc::BlockBuffer bb;
  webrtc::SpectrumBuffer sb;
  webrtc::FftBuffer fb;
  webrtc::RenderBuffer rb;
  webrtc::RenderSignalAnalyzer rsa;
  webrtc::AecState aec;
  S sub;
  webrtc::Block capture;
  std::vector<webrtc::SubtractorOutput> outputs;
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> E2, Y2;

  Impl(const subt_filter_config* c, int size, int num_render, int num_capture, webrtc::Aec3Optimization opt)
      : cfg(SubtConfig(c)),
        env(webrtc::CreateEnvironment()),
        dumper(0),
        nch(num_capture),
        bb(size, 1, num_render),
        sb(size, num_render),
        fb(size, num_render),
        rb(&bb, &sb, &fb),
        rsa(cfg),
        aec(env, cfg, num_capture),
        sub(env, cfg, num_render, num_capture, &dumper, opt),
        capture(1, num_capture),
        outputs(num_capture),
        E2(num_capture),
        Y2(num_capture) {
    for (auto& o : outputs) o.Reset();
  }

  void SetRender(int slot, int ch, const float* block64, const float* fft130, const float* spectrum65) override {
    std::copy(block64, block64 + 64, bb.buffer[slot].begin(0, ch));
    auto& X = fb.buffer[slot][ch];
    std::copy(fft130, fft130 + 65, X.re.begin());
    std::copy(fft130 + 65, fft130 + 130, X.im.begin());
    std::copy(spectrum65, spectrum65 + 65, sb.buffer[slot][ch].begin());
  }
  void SetIndices(int write, int read) override {
    bb.write = sb.write = fb.write = write;
    bb.read = sb.read = fb.read = read;
  }
  void HandleEchoPathChange(int gain_change, int delay_change) override {
    webrtc::EchoPathVariability v(gain_change != 0,
                                  static_cast<webrtc::EchoPathVariability::DelayAdjustment>(delay_change), false);
    sub.HandleEchoPathChange(v);
    aec.HandleEchoPathChange(v);
  }
  void ExitInitialState() override { sub.ExitInitialState(); }
  void Process(const float* y, const float* y2, int saturated, int follow_transition, subt_output* out) override {
    for (int ch = 0; ch < nch; ++ch) {
      std::copy(y + ch * 64, y + ch * 64 + 64, capture.begin(0, ch));
      std::copy(y2 + ch * 65, y2 + ch * 65 + 65, Y2[ch].begin());
    }
    aec.UpdateCaptureSaturation(saturated != 0);
    rsa.Update(rb, aec.MinDirectPathFilterDelay());
    if (follow_transition && aec.TransitionTriggered()) sub.ExitInitialState();
    sub.Process(rb, capture, rsa, aec, outputs);
    for (int ch = 0; ch < nch; ++ch) E2[ch] = outputs[ch].E2_refined;
    aec.Update(std::nullopt, sub.FilterFrequencyResponses(), sub.FilterImpulseResponses(), rb, E2, Y2, outputs);
    for (int ch = 0; ch < nch; ++ch) {
      const webrtc::SubtractorOutput& o = outputs[ch];
      subt_output& d = out[ch];
      std::copy(o.s_refined.begin(), o.s_refined.end(), d.s_refined);
      std::copy(o.s_coarse.begin(), o.s_coarse.end(), d.s_coarse);
      std::copy(o.e_refined.begin(), o.e_refined.end(), d.e_refined);
      std::copy(o.e_coarse.begin(), o.e_coarse.end(), d.e_coarse);
      std::copy(o.E_refined.re.begin(), o.E_refined.re.end(), d.E_refined_re);
      std::copy(o.E_refined.im.begin(), o.E_refined.im.end(), d.E_refined_im);
      std::copy(o.E2_refined.begin(), o.E2_refined.end(), d.E2_refined);
      std::copy(o.E2_coarse.begin(), o.E2_coarse.end(), d.E2_coarse);
      d.s2_refined = o.s2_refined;
      d.s2_coarse = o.s2_coarse;
      d.e2_refined = o.e2_refined;
      d.e2_coarse = o.e2_coarse;
      d.y2 = o.y2;
      d.s_refined_max_abs = o.s_refined_max_abs;
      d.s_coarse_max_abs = o.s_coarse_max_abs;
    }
  }
  void AecStateGet(subt_aec_state* out) override {
    out->min_direct_path_filter_delay = aec.MinDirectPathFilterDelay();
    out->transition_triggered = aec.TransitionTriggered();
    out->saturated_capture = aec.SaturatedCapture();
    out->usable_linear_estimate = aec.UsableLinearEstimate();
    out->fullband_erle_log2 = aec.FullBandErleLog2();
  }
  int FreqResp(int ch, float* H2) override {
    const auto& r = sub.FilterFrequencyResponses()[ch];
    for (size_t k = 0; k < r.size(); ++k) std::copy(r[k].begin(), r[k].end(), H2 + k * 65);
    return static_cast<int>(r.size());
  }
  int ImpulseResp(int ch, float* h) override {
    const auto& r = sub.FilterImpulseResponses()[ch];
    std::copy(r.begin(), r.end(), h);
    return static_cast<int>(r.size());
  }
};

}  // namespace subt_shim

#endif
