//go:build webrtc_oracle

#pragma clang fp contract(off)
#pragma clang diagnostic ignored "-Wnullability-completeness"

#include "sgain_oracle.h"

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
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/aec3/suppression_gain.h"

using namespace webrtc;

namespace {

constexpr int kN = kFftLengthBy2Plus1;
using Spectrum = std::array<float, kFftLengthBy2Plus1>;

void SgTuning(const float* t, EchoCanceller3Config::Suppressor::Tuning* out) {
  out->mask_lf = EchoCanceller3Config::Suppressor::MaskingThresholds(t[0], t[1], t[2]);
  out->mask_hf = EchoCanceller3Config::Suppressor::MaskingThresholds(t[3], t[4], t[5]);
  out->max_inc_factor = t[6];
  out->max_dec_factor_lf = t[7];
}

EchoCanceller3Config SgConfig(const sg_config* s) {
  const st_config* c = &s->st;
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

  cfg.filter.config_change_duration_blocks = s->config_change_duration_blocks;
  cfg.echo_audibility.low_render_limit = s->low_render_limit;
  cfg.echo_audibility.normal_render_limit = s->normal_render_limit;
  cfg.echo_audibility.floor_power = s->floor_power;
  cfg.echo_audibility.audibility_threshold_lf = s->audibility_threshold_lf;
  cfg.echo_audibility.audibility_threshold_mf = s->audibility_threshold_mf;
  cfg.echo_audibility.audibility_threshold_hf = s->audibility_threshold_hf;
  auto& sup = cfg.suppressor;
  sup.nearend_average_blocks = s->nearend_average_blocks;
  SgTuning(s->normal_tuning, &sup.normal_tuning);
  SgTuning(s->nearend_tuning, &sup.nearend_tuning);
  sup.lf_smoothing_during_initial_phase = s->lf_smoothing_during_initial_phase;
  sup.last_permanent_lf_smoothing_band = s->last_permanent_lf_smoothing_band;
  sup.last_lf_smoothing_band = s->last_lf_smoothing_band;
  sup.last_lf_band = s->last_lf_band;
  sup.first_hf_band = s->first_hf_band;
  auto& dn = sup.dominant_nearend_detection;
  dn.enr_threshold = s->dn_enr_threshold;
  dn.enr_exit_threshold = s->dn_enr_exit_threshold;
  dn.snr_threshold = s->dn_snr_threshold;
  dn.hold_duration = s->dn_hold_duration;
  dn.trigger_threshold = s->dn_trigger_threshold;
  dn.use_during_initial_phase = s->dn_use_during_initial_phase;
  dn.use_unbounded_echo_spectrum = s->dn_use_unbounded_echo_spectrum;
  auto& sn = sup.subband_nearend_detection;
  sn.nearend_average_blocks = s->sn_nearend_average_blocks;
  sn.subband1.low = s->sn_subband1_low;
  sn.subband1.high = s->sn_subband1_high;
  sn.subband2.low = s->sn_subband2_low;
  sn.subband2.high = s->sn_subband2_high;
  sn.nearend_threshold = s->sn_nearend_threshold;
  sn.snr_threshold = s->sn_snr_threshold;
  sup.use_subband_nearend_detection = s->use_subband_nearend_detection;
  sup.high_bands_suppression.enr_threshold = s->hb_enr_threshold;
  sup.high_bands_suppression.max_gain_during_echo = s->hb_max_gain_during_echo;
  sup.high_bands_suppression.anti_howling_activation_threshold = s->hb_anti_howling_activation_threshold;
  sup.high_bands_suppression.anti_howling_gain = s->hb_anti_howling_gain;
  sup.floor_first_increase = s->floor_first_increase;
  sup.conservative_hf_suppression = s->conservative_hf_suppression;
  return cfg;
}

std::vector<Spectrum> SgSpectra(const float* x, int n) {
  std::vector<Spectrum> v(n);
  for (int i = 0; i < n; ++i) std::copy(x + i * kN, x + (i + 1) * kN, v[i].begin());
  return v;
}

struct SgHarness {
  SgHarness(const EchoCanceller3Config& cfg, int opt, int rate, int num_render, int nch, int size)
      : num_render(num_render),
        nch(nch),
        blocks(size, 1, num_render),
        spectra(size, num_render),
        ffts(size, num_render),
        rb(&blocks, &spectra, &ffts),
        aec(CreateEnvironment(), cfg, nch),
        rsa(cfg),
        gain(cfg, static_cast<Aec3Optimization>(opt), rate, nch) {}
  int num_render;
  int nch;
  BlockBuffer blocks;
  SpectrumBuffer spectra;
  FftBuffer ffts;
  RenderBuffer rb;
  AecState aec;
  RenderSignalAnalyzer rsa;
  SuppressionGain gain;
};

SgHarness* SgH(void* h) { return static_cast<SgHarness*>(h); }

}  // namespace

extern "C" {

void* sgain_new(const sg_config* c, int optimization, int sample_rate_hz, int num_render, int num_capture,
                int render_buffer_size) {
  return new SgHarness(SgConfig(c), optimization, sample_rate_hz, num_render, num_capture, render_buffer_size);
}
void sgain_free(void* h) { delete SgH(h); }
void sgain_set_render_block(void* h, int slot, int ch, const float* x) {
  auto v = SgH(h)->blocks.buffer[slot].View(0, ch);
  std::copy(x, x + kBlockSize, v.begin());
}
void sgain_set_render_spectrum(void* h, int slot, int ch, const float* x2) {
  auto& s = SgH(h)->spectra.buffer[slot][ch];
  std::copy(x2, x2 + kN, s.begin());
}
void sgain_set_indices(void* h, int write, int read) {
  auto* p = SgH(h);
  p->blocks.write = p->spectra.write = p->ffts.write = write;
  p->blocks.read = p->spectra.read = p->ffts.read = read;
}
void sgain_aec_handle_echo_path_change(void* h, int gain_change, int delay_change) {
  EchoPathVariability v(gain_change, static_cast<EchoPathVariability::DelayAdjustment>(delay_change), false);
  SgH(h)->aec.HandleEchoPathChange(v);
}
void sgain_aec_update_capture_saturation(void* h, int saturated) { SgH(h)->aec.UpdateCaptureSaturation(saturated); }
void sgain_aec_update(void* h, int ext_valid, int ext_quality, int ext_delay, const float* fr, int num_blocks,
                      const float* ir, int ir_len, const float* E2, const float* Y2, const float* subtractor) {
  auto* s = SgH(h);
  std::optional<DelayEstimate> ext;
  if (ext_valid) ext = DelayEstimate(static_cast<DelayEstimate::Quality>(ext_quality), ext_delay);
  std::vector<std::vector<Spectrum>> frv(s->nch, std::vector<Spectrum>(num_blocks));
  for (int ch = 0; ch < s->nch; ++ch)
    for (int b = 0; b < num_blocks; ++b)
      std::copy(fr + (ch * num_blocks + b) * kN, fr + (ch * num_blocks + b + 1) * kN, frv[ch][b].begin());
  std::vector<std::vector<float>> ir_v(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) ir_v[ch].assign(ir + ch * ir_len, ir + (ch + 1) * ir_len);
  std::vector<SubtractorOutput> so(s->nch);
  for (int ch = 0; ch < s->nch; ++ch) {
    const float* v = subtractor + 5 * ch;
    so[ch].y2 = v[0];
    so[ch].e2_refined = v[1];
    so[ch].e2_coarse = v[2];
    so[ch].s_refined_max_abs = v[3];
    so[ch].s_coarse_max_abs = v[4];
  }
  auto e2 = SgSpectra(E2, s->nch);
  auto y2 = SgSpectra(Y2, s->nch);
  s->aec.Update(ext, frv, ir_v, s->rb, e2, y2, so);
}
int sgain_aec_saturated_echo(void* h) { return SgH(h)->aec.SaturatedEcho(); }
int sgain_aec_min_direct_path_filter_delay(void* h) { return SgH(h)->aec.MinDirectPathFilterDelay(); }
void sgain_rsa_update(void* h, int delay, int has_delay) {
  std::optional<size_t> d;
  if (has_delay) d = static_cast<size_t>(delay);
  SgH(h)->rsa.Update(SgH(h)->rb, d);
}
int sgain_rsa_narrow_peak_band(void* h, int* band) {
  auto b = SgH(h)->rsa.NarrowPeakBand();
  *band = b ? *b : -1;
  return b.has_value();
}
void sgain_set_initial_state(void* h, int state) { SgH(h)->gain.SetInitialState(state); }
int sgain_get_gain(void* h, const float* nearend, const float* echo, const float* residual,
                   const float* residual_unbounded, const float* comfort_noise, const float* render, int num_bands,
                   int clock_drift, float* high_bands_gain, float* low_band_gain) {
  auto* s = SgH(h);
  auto ne = SgSpectra(nearend, s->nch);
  auto ec = SgSpectra(echo, s->nch);
  auto re = SgSpectra(residual, s->nch);
  auto reu = SgSpectra(residual_unbounded, s->nch);
  auto cn = SgSpectra(comfort_noise, s->nch);
  Block block(num_bands, s->num_render);
  for (int b = 0; b < num_bands; ++b) {
    for (int ch = 0; ch < s->num_render; ++ch) {
      auto v = block.View(b, ch);
      const float* src = render + (b * s->num_render + ch) * kBlockSize;
      std::copy(src, src + kBlockSize, v.begin());
    }
  }
  Spectrum low;
  s->gain.GetGain(ne, ec, re, reu, cn, s->rsa, s->aec, block, clock_drift, high_bands_gain, &low);
  std::copy(low.begin(), low.end(), low_band_gain);
  return s->gain.IsDominantNearend();
}

}  // extern "C"
