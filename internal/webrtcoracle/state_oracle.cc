//go:build webrtc_oracle

#pragma clang fp contract(off)
#pragma clang diagnostic ignored "-Wnullability-completeness"
#define WEBRTC_APM_DEBUG_DUMP 0

#include "state_oracle.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/field_trials_view.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/echo_audibility.h"
#include "modules/audio_processing/aec3/erl_estimator.h"
#include "modules/audio_processing/aec3/erle_estimator.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/filter_analyzer.h"
#include "modules/audio_processing/aec3/fullband_erle_estimator.h"
#include "modules/audio_processing/aec3/moving_average.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/residual_echo_estimator.h"
#include "modules/audio_processing/aec3/reverb_decay_estimator.h"
#include "modules/audio_processing/aec3/reverb_frequency_response.h"
#include "modules/audio_processing/aec3/reverb_model.h"
#include "modules/audio_processing/aec3/signal_dependent_erle_estimator.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "modules/audio_processing/aec3/stationarity_estimator.h"
#include "modules/audio_processing/aec3/subband_erle_estimator.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/aec3/subtractor_output_analyzer.h"
#include "modules/audio_processing/aec3/transparent_mode.h"

using namespace webrtc;

namespace {

constexpr int kN = kFftLengthBy2Plus1;
using Spectrum = std::array<float, kFftLengthBy2Plus1>;

class StTrials : public FieldTrialsView {
 public:
  explicit StTrials(bool hmm) : hmm_(hmm) {}
  std::string Lookup(absl::string_view key) const override {
    if (hmm_ && key == "WebRTC-Aec3TransparentModeHmm") return "Enabled";
    return "";
  }

 private:
  bool hmm_;
};

Environment StEnv(bool hmm) {
  return CreateEnvironment(std::unique_ptr<FieldTrialsView>(new StTrials(hmm)));
}

EchoCanceller3Config StConfig(const st_config* c) {
  EchoCanceller3Config cfg;
  cfg.filter.refined.length_blocks = c->refined_length_blocks;
  cfg.filter.refined_initial.length_blocks = c->refined_initial_length_blocks;
  cfg.filter.use_linear_filter = c->use_linear_filter;
  cfg.filter.conservative_initial_phase = c->conservative_initial_phase;
  cfg.filter.initial_state_seconds = c->initial_state_seconds;
  cfg.erle.min = c->erle_min;
  cfg.erle.max_l = c->erle_max_l;
  cfg.erle.max_h = c->erle_max_h;
  cfg.erle.onset_detection = c->erle_onset_detection;
  cfg.erle.num_sections = c->erle_num_sections;
  cfg.erle.clamp_quality_estimate_to_zero = c->erle_clamp_quality_estimate_to_zero;
  cfg.erle.clamp_quality_estimate_to_one = c->erle_clamp_quality_estimate_to_one;
  cfg.ep_strength.default_gain = c->default_gain;
  cfg.ep_strength.default_len = c->default_len;
  cfg.ep_strength.nearend_len = c->nearend_len;
  cfg.ep_strength.echo_can_saturate = c->echo_can_saturate;
  cfg.ep_strength.bounded_erl = c->bounded_erl;
  cfg.ep_strength.erle_onset_compensation_in_dominant_nearend = c->erle_onset_compensation_in_dominant_nearend;
  cfg.ep_strength.use_conservative_tail_frequency_response = c->use_conservative_tail_frequency_response;
  cfg.echo_audibility.use_stationarity_properties = c->use_stationarity_properties;
  cfg.echo_audibility.use_stationarity_properties_at_init = c->use_stationarity_properties_at_init;
  cfg.render_levels.active_render_limit = c->active_render_limit;
  cfg.delay.delay_headroom_samples = c->delay_headroom_samples;
  cfg.echo_removal_control.linear_and_stable_echo_path = c->linear_and_stable_echo_path;
  cfg.echo_model.noise_floor_hold = c->noise_floor_hold;
  cfg.echo_model.min_noise_floor_power = c->min_noise_floor_power;
  cfg.echo_model.stationary_gate_slope = c->stationary_gate_slope;
  cfg.echo_model.noise_gate_power = c->noise_gate_power;
  cfg.echo_model.noise_gate_slope = c->noise_gate_slope;
  cfg.echo_model.render_pre_window_size = c->render_pre_window_size;
  cfg.echo_model.render_post_window_size = c->render_post_window_size;
  cfg.echo_model.model_reverb_in_nonlinear_mode = c->model_reverb_in_nonlinear_mode;
  return cfg;
}

struct StRender {
  StRender(int size, int nch)
      : blocks(size, 1, nch), spectra(size, nch), ffts(size, nch), rb(&blocks, &spectra, &ffts) {}
  BlockBuffer blocks;
  SpectrumBuffer spectra;
  FftBuffer ffts;
  RenderBuffer rb;
};

std::vector<Spectrum> Spectra(const float* x, int n) {
  std::vector<Spectrum> v(n);
  for (int i = 0; i < n; ++i) std::copy(x + i * kN, x + (i + 1) * kN, v[i].begin());
  return v;
}

std::vector<std::vector<Spectrum>> FreqResp(const float* fr, int nch, int num_blocks) {
  std::vector<std::vector<Spectrum>> v(nch, std::vector<Spectrum>(num_blocks));
  for (int ch = 0; ch < nch; ++ch)
    for (int b = 0; b < num_blocks; ++b)
      std::copy(fr + (ch * num_blocks + b) * kN, fr + (ch * num_blocks + b + 1) * kN, v[ch][b].begin());
  return v;
}

std::vector<bool> Bools(const int* x, int n) {
  std::vector<bool> v(n);
  for (int i = 0; i < n; ++i) v[i] = x[i] != 0;
  return v;
}

void PutSpectra(ArrayView<const Spectrum> s, float* out) {
  for (size_t i = 0; i < s.size(); ++i) std::copy(s[i].begin(), s[i].end(), out + i * kN);
}

std::optional<float> Opt(int ok, float v) { return ok ? std::optional<float>(v) : std::nullopt; }

}  // namespace

extern "C" {

void* st_render_new(int size, int nch) { return new StRender(size, nch); }
void st_render_free(void* r) { delete static_cast<StRender*>(r); }
void st_render_set_block(void* r, int slot, int ch, const float* x) {
  auto v = static_cast<StRender*>(r)->blocks.buffer[slot].View(0, ch);
  std::copy(x, x + kBlockSize, v.begin());
}
void st_render_set_spectrum(void* r, int slot, int ch, const float* x2) {
  auto& s = static_cast<StRender*>(r)->spectra.buffer[slot][ch];
  std::copy(x2, x2 + kN, s.begin());
}
void st_render_set_indices(void* r, int write, int read) {
  auto* p = static_cast<StRender*>(r);
  p->blocks.write = p->spectra.write = p->ffts.write = write;
  p->blocks.read = p->spectra.read = p->ffts.read = read;
}

void* st_moving_average_new(int num_elem, int mem_len) { return new aec3::MovingAverage(num_elem, mem_len); }
void st_moving_average_free(void* p) { delete static_cast<aec3::MovingAverage*>(p); }
void st_moving_average_average(void* p, const float* in, float* out, int n) {
  static_cast<aec3::MovingAverage*>(p)->Average(ArrayView<const float>(in, n), ArrayView<float>(out, n));
}

void* st_reverb_model_new(void) { return new ReverbModel(); }
void st_reverb_model_free(void* p) { delete static_cast<ReverbModel*>(p); }
void st_reverb_model_update_no_freq_shaping(void* p, const float* ps, int n, float scaling, float decay) {
  static_cast<ReverbModel*>(p)->UpdateReverbNoFreqShaping(ArrayView<const float>(ps, n), scaling, decay);
}
void st_reverb_model_update(void* p, const float* ps, const float* scaling, int n, float decay) {
  static_cast<ReverbModel*>(p)->UpdateReverb(ArrayView<const float>(ps, n), ArrayView<const float>(scaling, n),
                                             decay);
}
void st_reverb_model_reverb(void* p, float* out) {
  auto r = static_cast<ReverbModel*>(p)->reverb();
  std::copy(r.begin(), r.end(), out);
}

void* st_reverb_frequency_response_new(int conservative) { return new ReverbFrequencyResponse(conservative); }
void st_reverb_frequency_response_free(void* p) { delete static_cast<ReverbFrequencyResponse*>(p); }
void st_reverb_frequency_response_update(void* p, const float* fr, int num_blocks, int delay, int q_ok, float q,
                                         int stationary) {
  auto v = FreqResp(fr, 1, num_blocks);
  static_cast<ReverbFrequencyResponse*>(p)->Update(v[0], delay, Opt(q_ok, q), stationary);
}
void st_reverb_frequency_response_get(void* p, float* out) {
  auto r = static_cast<ReverbFrequencyResponse*>(p)->FrequencyResponse();
  std::copy(r.begin(), r.end(), out);
}

void* st_reverb_decay_new(const st_config* c) { return new ReverbDecayEstimator(StConfig(c)); }
void st_reverb_decay_free(void* p) { delete static_cast<ReverbDecayEstimator*>(p); }
void st_reverb_decay_update(void* p, const float* filter, int n, int q_ok, float q, int delay, int usable,
                            int stationary) {
  static_cast<ReverbDecayEstimator*>(p)->Update(ArrayView<const float>(filter, n), Opt(q_ok, q), delay, usable,
                                                stationary);
}
float st_reverb_decay_decay(void* p, int mild) { return static_cast<ReverbDecayEstimator*>(p)->Decay(mild); }

void* st_erl_new(int startup) { return new ErlEstimator(startup); }
void st_erl_free(void* p) { delete static_cast<ErlEstimator*>(p); }
void st_erl_reset(void* p) { static_cast<ErlEstimator*>(p)->Reset(); }
void st_erl_update(void* p, const int* converged, int num_capture, const float* render, int num_render,
                   const float* capture) {
  auto x = Spectra(render, num_render);
  auto y = Spectra(capture, num_capture);
  static_cast<ErlEstimator*>(p)->Update(Bools(converged, num_capture), x, y);
}
void st_erl_get(void* p, float* erl, float* erl_td) {
  auto* e = static_cast<ErlEstimator*>(p);
  std::copy(e->Erl().begin(), e->Erl().end(), erl);
  *erl_td = e->ErlTimeDomain();
}

struct StFullband {
  StFullband(const EchoCanceller3Config& cfg, int nch) : nch(nch), e(cfg.erle, nch) {}
  int nch;
  FullBandErleEstimator e;
};
void* st_fullband_erle_new(const st_config* c, int nch) { return new StFullband(StConfig(c), nch); }
void st_fullband_erle_free(void* p) { delete static_cast<StFullband*>(p); }
void st_fullband_erle_reset(void* p) { static_cast<StFullband*>(p)->e.Reset(); }
void st_fullband_erle_update(void* p, const float* X2, const float* Y2, const float* E2, const int* converged) {
  auto* f = static_cast<StFullband*>(p);
  f->e.Update(ArrayView<const float>(X2, kN), Spectra(Y2, f->nch), Spectra(E2, f->nch), Bools(converged, f->nch));
}
float st_fullband_erle_log2(void* p) { return static_cast<StFullband*>(p)->e.FullbandErleLog2(); }
void st_fullband_erle_qualities(void* p, float* q, int* ok) {
  auto v = static_cast<StFullband*>(p)->e.GetInstLinearQualityEstimates();
  for (size_t i = 0; i < v.size(); ++i) {
    ok[i] = v[i].has_value();
    q[i] = v[i].value_or(0.f);
  }
}

struct StSubband {
  StSubband(const EchoCanceller3Config& cfg, int nch) : nch(nch), e(StEnv(false), cfg, nch) {}
  int nch;
  SubbandErleEstimator e;
};
void* st_subband_erle_new(const st_config* c, int nch) { return new StSubband(StConfig(c), nch); }
void st_subband_erle_free(void* p) { delete static_cast<StSubband*>(p); }
void st_subband_erle_reset(void* p) { static_cast<StSubband*>(p)->e.Reset(); }
void st_subband_erle_update(void* p, const float* X2, const float* Y2, const float* E2, const int* converged) {
  auto* s = static_cast<StSubband*>(p);
  s->e.Update(ArrayView<const float, kFftLengthBy2Plus1>(X2, kN), Spectra(Y2, s->nch), Spectra(E2, s->nch),
              Bools(converged, s->nch));
}
void st_subband_erle_get(void* p, int which, float* out) {
  auto& e = static_cast<StSubband*>(p)->e;
  switch (which) {
    case 0: PutSpectra(e.Erle(false), out); break;
    case 1: PutSpectra(e.Erle(true), out); break;
    case 2: PutSpectra(e.ErleUnbounded(), out); break;
    default: PutSpectra(e.ErleDuringOnsets(), out); break;
  }
}

struct StSd {
  StSd(const EchoCanceller3Config& cfg, int nch) : nch(nch), e(cfg, nch) {}
  int nch;
  SignalDependentErleEstimator e;
};
void* st_sd_erle_new(const st_config* c, int nch) { return new StSd(StConfig(c), nch); }
void st_sd_erle_free(void* p) { delete static_cast<StSd*>(p); }
void st_sd_erle_reset(void* p) { static_cast<StSd*>(p)->e.Reset(); }
void st_sd_erle_update(void* p, void* render, const float* fr, int num_blocks, const float* X2, const float* Y2,
                       const float* E2, const float* avg_erle, const float* avg_erle_oc, const int* converged) {
  auto* s = static_cast<StSd*>(p);
  auto frv = FreqResp(fr, s->nch, num_blocks);
  s->e.Update(static_cast<StRender*>(render)->rb, frv, ArrayView<const float, kFftLengthBy2Plus1>(X2, kN),
              Spectra(Y2, s->nch), Spectra(E2, s->nch), Spectra(avg_erle, s->nch), Spectra(avg_erle_oc, s->nch),
              Bools(converged, s->nch));
}
void st_sd_erle_get(void* p, int onset_compensated, float* out) {
  PutSpectra(static_cast<StSd*>(p)->e.Erle(onset_compensated), out);
}

struct StErle {
  StErle(const EchoCanceller3Config& cfg, int startup, int nch) : nch(nch), e(StEnv(false), startup, cfg, nch) {}
  int nch;
  ErleEstimator e;
};
void* st_erle_new(const st_config* c, int startup, int nch) { return new StErle(StConfig(c), startup, nch); }
void st_erle_free(void* p) { delete static_cast<StErle*>(p); }
void st_erle_reset(void* p, int delay_change) { static_cast<StErle*>(p)->e.Reset(delay_change); }
void st_erle_update(void* p, void* render, const float* fr, int num_blocks, const float* X2, const float* Y2,
                    const float* E2, const int* converged) {
  auto* s = static_cast<StErle*>(p);
  auto frv = FreqResp(fr, s->nch, num_blocks);
  s->e.Update(static_cast<StRender*>(render)->rb, frv, ArrayView<const float, kFftLengthBy2Plus1>(X2, kN),
              Spectra(Y2, s->nch), Spectra(E2, s->nch), Bools(converged, s->nch));
}
void st_erle_get(void* p, int which, float* out) {
  auto& e = static_cast<StErle*>(p)->e;
  switch (which) {
    case 0: PutSpectra(e.Erle(false), out); break;
    case 1: PutSpectra(e.Erle(true), out); break;
    case 2: PutSpectra(e.ErleUnbounded(), out); break;
    default: PutSpectra(e.ErleDuringOnsets(), out); break;
  }
}
float st_erle_fullband_log2(void* p) { return static_cast<StErle*>(p)->e.FullbandErleLog2(); }
void st_erle_qualities(void* p, float* q, int* ok) {
  auto v = static_cast<StErle*>(p)->e.GetInstLinearQualityEstimates();
  for (size_t i = 0; i < v.size(); ++i) {
    ok[i] = v[i].has_value();
    q[i] = v[i].value_or(0.f);
  }
}

void* st_stationarity_new(void) { return new StationarityEstimator(); }
void st_stationarity_free(void* p) { delete static_cast<StationarityEstimator*>(p); }
void st_stationarity_update_noise(void* p, const float* spectrum, int nch) {
  static_cast<StationarityEstimator*>(p)->UpdateNoiseEstimator(Spectra(spectrum, nch));
}
void st_stationarity_update_flags(void* p, void* render, const float* reverb, int idx_current, int num_lookahead) {
  static_cast<StationarityEstimator*>(p)->UpdateStationarityFlags(
      static_cast<StRender*>(render)->spectra, ArrayView<const float>(reverb, kN), idx_current, num_lookahead);
}
void st_stationarity_flags(void* p, int* band_stationary, int* block_stationary) {
  auto* s = static_cast<StationarityEstimator*>(p);
  for (int k = 0; k < kN; ++k) band_stationary[k] = s->IsBandStationary(k);
  *block_stationary = s->IsBlockStationary();
}

void* st_echo_audibility_new(int at_init) { return new EchoAudibility(at_init); }
void st_echo_audibility_free(void* p) { delete static_cast<EchoAudibility*>(p); }
void st_echo_audibility_update(void* p, void* render, const float* reverb, int delay, int external_delay_seen) {
  static_cast<EchoAudibility*>(p)->Update(static_cast<StRender*>(render)->rb, ArrayView<const float>(reverb, kN),
                                          delay, external_delay_seen);
}
void st_echo_audibility_get(void* p, int converged, float* scaling, int* block_stationary) {
  auto* e = static_cast<EchoAudibility*>(p);
  e->GetResidualEchoScaling(converged, ArrayView<float>(scaling, kN));
  *block_stationary = e->IsBlockStationary();
}

struct StFilterAnalyzer {
  StFilterAnalyzer(const EchoCanceller3Config& cfg, int nch) : nch(nch), f(cfg, nch) {}
  int nch;
  FilterAnalyzer f;
};
void* st_filter_analyzer_new(const st_config* c, int nch) { return new StFilterAnalyzer(StConfig(c), nch); }
void st_filter_analyzer_free(void* p) { delete static_cast<StFilterAnalyzer*>(p); }
void st_filter_analyzer_reset(void* p) { static_cast<StFilterAnalyzer*>(p)->f.Reset(); }
void st_filter_analyzer_update(void* p, const float* filters, int len, void* render, int* consistent, float* gain) {
  auto* s = static_cast<StFilterAnalyzer*>(p);
  std::vector<std::vector<float>> h(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) h[ch].assign(filters + ch * len, filters + (ch + 1) * len);
  bool any = false;
  s->f.Update(h, static_cast<StRender*>(render)->rb, &any, gain);
  *consistent = any;
}
void st_filter_analyzer_get(void* p, int* delays, int* min_delay, int* length_blocks, float* adjusted) {
  auto* s = static_cast<StFilterAnalyzer*>(p);
  auto d = s->f.FilterDelaysBlocks();
  for (size_t i = 0; i < d.size(); ++i) delays[i] = d[i];
  *min_delay = s->f.MinFilterDelayBlocks();
  *length_blocks = s->f.FilterLengthBlocks();
  auto h = s->f.GetAdjustedFilters();
  size_t off = 0;
  for (size_t ch = 0; ch < h.size(); ++ch) {
    std::copy(h[ch].begin(), h[ch].end(), adjusted + off);
    off += h[ch].size();
  }
}

struct StSubAnalyzer {
  explicit StSubAnalyzer(int nch) : nch(nch), a(nch) {}
  int nch;
  SubtractorOutputAnalyzer a;
};
void* st_subtractor_analyzer_new(int nch) { return new StSubAnalyzer(nch); }
void st_subtractor_analyzer_free(void* p) { delete static_cast<StSubAnalyzer*>(p); }
void st_subtractor_analyzer_update(void* p, const float* y2, const float* e2r, const float* e2c, int* out3,
                                   int* converged) {
  auto* s = static_cast<StSubAnalyzer*>(p);
  std::vector<SubtractorOutput> so(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) {
    so[ch].y2 = y2[ch];
    so[ch].e2_refined = e2r[ch];
    so[ch].e2_coarse = e2c[ch];
  }
  bool a = false, b = false, c = false;
  s->a.Update(so, &a, &b, &c);
  out3[0] = a;
  out3[1] = b;
  out3[2] = c;
  const auto& conv = s->a.ConvergedFilters();
  for (int ch = 0; ch < s->nch; ++ch) converged[ch] = conv[ch];
}
void st_subtractor_analyzer_handle_echo_path_change(void* p) {
  static_cast<StSubAnalyzer*>(p)->a.HandleEchoPathChange();
}

void* st_transparent_mode_new(const st_config* c, int hmm) {
  return TransparentMode::Create(StEnv(hmm), StConfig(c)).release();
}
void st_transparent_mode_free(void* p) { delete static_cast<TransparentMode*>(p); }
void st_transparent_mode_reset(void* p) { static_cast<TransparentMode*>(p)->Reset(); }
int st_transparent_mode_update(void* p, int delay, int consistent, int converged, int coarse_converged, int diverged,
                               int active_render, int saturated_capture) {
  auto* t = static_cast<TransparentMode*>(p);
  t->Update(delay, consistent, converged, coarse_converged, diverged, active_render, saturated_capture);
  return t->Active();
}

struct StAec {
  StAec(const EchoCanceller3Config& cfg, int nch, bool hmm) : nch(nch), a(StEnv(hmm), cfg, nch) {}
  int nch;
  AecState a;
};
void* st_aec_state_new(const st_config* c, int nch, int hmm) { return new StAec(StConfig(c), nch, hmm); }
void st_aec_state_free(void* p) { delete static_cast<StAec*>(p); }
void st_aec_state_handle_echo_path_change(void* p, int gain_change, int delay_change) {
  EchoPathVariability v(gain_change, static_cast<EchoPathVariability::DelayAdjustment>(delay_change), false);
  static_cast<StAec*>(p)->a.HandleEchoPathChange(v);
}
void st_aec_state_update_capture_saturation(void* p, int saturated) {
  static_cast<StAec*>(p)->a.UpdateCaptureSaturation(saturated);
}
void st_aec_state_update(void* p, int ext_valid, int ext_quality, int ext_delay, const float* fr, int num_blocks,
                         const float* ir, int ir_len, void* render, const float* E2, const float* Y2,
                         const float* subtractor) {
  auto* s = static_cast<StAec*>(p);
  std::optional<DelayEstimate> ext;
  if (ext_valid) ext = DelayEstimate(static_cast<DelayEstimate::Quality>(ext_quality), ext_delay);
  auto frv = FreqResp(fr, s->nch, num_blocks);
  std::vector<std::vector<float>> h(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) h[ch].assign(ir + ch * ir_len, ir + (ch + 1) * ir_len);
  std::vector<SubtractorOutput> so(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) {
    const float* v = subtractor + 5 * ch;
    so[ch].y2 = v[0];
    so[ch].e2_refined = v[1];
    so[ch].e2_coarse = v[2];
    so[ch].s_refined_max_abs = v[3];
    so[ch].s_coarse_max_abs = v[4];
  }
  auto e2 = Spectra(E2, s->nch);
  auto y2 = Spectra(Y2, s->nch);
  s->a.Update(ext, frv, h, static_cast<StRender*>(render)->rb, e2, y2, so);
}
void st_aec_state_get(void* p, st_aec_state_out* out) {
  auto* s = static_cast<StAec*>(p);
  const AecState& a = s->a;
  PutSpectra(a.Erle(false), &out->erle[0][0]);
  PutSpectra(a.Erle(true), &out->erle_oc[0][0]);
  PutSpectra(a.ErleUnbounded(), &out->erle_unbounded[0][0]);
  std::copy(a.Erl().begin(), a.Erl().end(), out->erl);
  a.GetResidualEchoScaling(ArrayView<float>(out->residual_scaling, kN));
  auto fr = a.GetReverbFrequencyResponse();
  std::copy(fr.begin(), fr.end(), out->reverb_freq_resp);
  out->erl_time_domain = a.ErlTimeDomain();
  out->fullband_erle_log2 = a.FullBandErleLog2();
  out->reverb_decay = a.ReverbDecay(false);
  out->reverb_decay_mild = a.ReverbDecay(true);
  out->usable_linear_estimate = a.UsableLinearEstimate();
  out->use_linear_filter_output = a.UseLinearFilterOutput();
  out->active_render = a.ActiveRender();
  out->use_stationarity_properties = a.UseStationarityProperties();
  out->min_direct_path_filter_delay = a.MinDirectPathFilterDelay();
  out->saturated_capture = a.SaturatedCapture();
  out->saturated_echo = a.SaturatedEcho();
  out->transparent_mode_active = a.TransparentModeActive();
  out->transition_triggered = a.TransitionTriggered();
  out->filter_length_blocks = a.FilterLengthBlocks();
}

struct StResidual {
  StResidual(const EchoCanceller3Config& cfg, int nr) : r(StEnv(false), cfg, nr) {}
  ResidualEchoEstimator r;
};
void* st_residual_echo_new(const st_config* c, int num_render) { return new StResidual(StConfig(c), num_render); }
void st_residual_echo_free(void* p) { delete static_cast<StResidual*>(p); }
void st_residual_echo_estimate(void* p, void* aec, void* render, const float* S2, const float* Y2, int nch,
                               int dominant_nearend, float* R2, float* R2u) {
  std::vector<Spectrum> r2(nch), r2u(nch);
  static_cast<StResidual*>(p)->r.Estimate(static_cast<StAec*>(aec)->a, static_cast<StRender*>(render)->rb,
                                          Spectra(S2, nch), Spectra(Y2, nch), dominant_nearend, r2, r2u);
  PutSpectra(r2, R2);
  PutSpectra(r2u, R2u);
}

}  // extern "C"
