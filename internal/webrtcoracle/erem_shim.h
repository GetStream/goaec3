// Shared EchoRemover simulation, instantiated for the prebuilt class
// (erem_oracle.cc) and for the source builds (erem_src.cc, erem_scalar.cc).
#ifndef GOAEC3_EREM_SHIM_H
#define GOAEC3_EREM_SHIM_H

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"

// Forced lane read by the source builds' DetectOptimization replacement.
extern int erem_forced_lane;

// EREM_CONFIG(F, I) lists every EchoCanceller3Config field; F for float
// fields, I for integer and bool fields. erem_oracle.go flattens in the same
// order.
#define EREM_CONFIG(F, I)                                      \
  I(buffering.excess_render_detection_interval_blocks)         \
  I(buffering.max_allowed_excess_render_blocks)                \
  I(delay.default_delay)                                       \
  I(delay.down_sampling_factor)                                \
  I(delay.num_filters)                                         \
  I(delay.delay_headroom_samples)                              \
  I(delay.hysteresis_limit_blocks)                             \
  I(delay.fixed_capture_delay_samples)                         \
  F(delay.delay_estimate_smoothing)                            \
  F(delay.delay_estimate_smoothing_delay_found)                \
  F(delay.delay_candidate_detection_threshold)                 \
  I(delay.delay_selection_thresholds.initial)                  \
  I(delay.delay_selection_thresholds.converged)                \
  I(delay.use_external_delay_estimator)                        \
  I(delay.log_warning_on_delay_changes)                        \
  I(delay.render_alignment_mixing.downmix)                     \
  I(delay.render_alignment_mixing.adaptive_selection)          \
  F(delay.render_alignment_mixing.activity_power_threshold)    \
  I(delay.render_alignment_mixing.prefer_first_two_channels)   \
  I(delay.capture_alignment_mixing.downmix)                    \
  I(delay.capture_alignment_mixing.adaptive_selection)         \
  F(delay.capture_alignment_mixing.activity_power_threshold)   \
  I(delay.capture_alignment_mixing.prefer_first_two_channels)  \
  I(delay.detect_pre_echo)                                     \
  I(filter.refined.length_blocks)                              \
  F(filter.refined.leakage_converged)                          \
  F(filter.refined.leakage_diverged)                           \
  F(filter.refined.error_floor)                                \
  F(filter.refined.error_ceil)                                 \
  F(filter.refined.noise_gate)                                 \
  I(filter.coarse.length_blocks)                               \
  F(filter.coarse.rate)                                        \
  F(filter.coarse.noise_gate)                                  \
  I(filter.refined_initial.length_blocks)                      \
  F(filter.refined_initial.leakage_converged)                  \
  F(filter.refined_initial.leakage_diverged)                   \
  F(filter.refined_initial.error_floor)                        \
  F(filter.refined_initial.error_ceil)                         \
  F(filter.refined_initial.noise_gate)                         \
  I(filter.coarse_initial.length_blocks)                       \
  F(filter.coarse_initial.rate)                                \
  F(filter.coarse_initial.noise_gate)                          \
  I(filter.config_change_duration_blocks)                      \
  F(filter.initial_state_seconds)                              \
  I(filter.coarse_reset_hangover_blocks)                       \
  I(filter.conservative_initial_phase)                         \
  I(filter.enable_coarse_filter_output_usage)                  \
  I(filter.use_linear_filter)                                  \
  I(filter.high_pass_filter_echo_reference)                    \
  I(filter.export_linear_aec_output)                           \
  F(erle.min)                                                  \
  F(erle.max_l)                                                \
  F(erle.max_h)                                                \
  I(erle.onset_detection)                                      \
  I(erle.num_sections)                                         \
  I(erle.clamp_quality_estimate_to_zero)                       \
  I(erle.clamp_quality_estimate_to_one)                        \
  F(ep_strength.default_gain)                                  \
  F(ep_strength.default_len)                                   \
  F(ep_strength.nearend_len)                                   \
  I(ep_strength.echo_can_saturate)                             \
  I(ep_strength.bounded_erl)                                   \
  I(ep_strength.erle_onset_compensation_in_dominant_nearend)   \
  I(ep_strength.use_conservative_tail_frequency_response)      \
  F(echo_audibility.low_render_limit)                          \
  F(echo_audibility.normal_render_limit)                       \
  F(echo_audibility.floor_power)                               \
  F(echo_audibility.audibility_threshold_lf)                   \
  F(echo_audibility.audibility_threshold_mf)                   \
  F(echo_audibility.audibility_threshold_hf)                   \
  I(echo_audibility.use_stationarity_properties)               \
  I(echo_audibility.use_stationarity_properties_at_init)       \
  F(render_levels.active_render_limit)                         \
  F(render_levels.poor_excitation_render_limit)                \
  F(render_levels.poor_excitation_render_limit_ds8)            \
  F(render_levels.render_power_gain_db)                        \
  I(echo_removal_control.has_clock_drift)                      \
  I(echo_removal_control.linear_and_stable_echo_path)          \
  I(echo_model.noise_floor_hold)                               \
  F(echo_model.min_noise_floor_power)                          \
  F(echo_model.stationary_gate_slope)                          \
  F(echo_model.noise_gate_power)                               \
  F(echo_model.noise_gate_slope)                               \
  I(echo_model.render_pre_window_size)                         \
  I(echo_model.render_post_window_size)                        \
  I(echo_model.model_reverb_in_nonlinear_mode)                 \
  F(comfort_noise.noise_floor_dbfs)                            \
  I(suppressor.nearend_average_blocks)                         \
  F(suppressor.normal_tuning.mask_lf.enr_transparent)          \
  F(suppressor.normal_tuning.mask_lf.enr_suppress)             \
  F(suppressor.normal_tuning.mask_lf.emr_transparent)          \
  F(suppressor.normal_tuning.mask_hf.enr_transparent)          \
  F(suppressor.normal_tuning.mask_hf.enr_suppress)             \
  F(suppressor.normal_tuning.mask_hf.emr_transparent)          \
  F(suppressor.normal_tuning.max_inc_factor)                   \
  F(suppressor.normal_tuning.max_dec_factor_lf)                \
  F(suppressor.nearend_tuning.mask_lf.enr_transparent)         \
  F(suppressor.nearend_tuning.mask_lf.enr_suppress)            \
  F(suppressor.nearend_tuning.mask_lf.emr_transparent)         \
  F(suppressor.nearend_tuning.mask_hf.enr_transparent)         \
  F(suppressor.nearend_tuning.mask_hf.enr_suppress)            \
  F(suppressor.nearend_tuning.mask_hf.emr_transparent)         \
  F(suppressor.nearend_tuning.max_inc_factor)                  \
  F(suppressor.nearend_tuning.max_dec_factor_lf)               \
  I(suppressor.lf_smoothing_during_initial_phase)              \
  I(suppressor.last_permanent_lf_smoothing_band)               \
  I(suppressor.last_lf_smoothing_band)                         \
  I(suppressor.last_lf_band)                                   \
  I(suppressor.first_hf_band)                                  \
  F(suppressor.dominant_nearend_detection.enr_threshold)       \
  F(suppressor.dominant_nearend_detection.enr_exit_threshold)  \
  F(suppressor.dominant_nearend_detection.snr_threshold)       \
  I(suppressor.dominant_nearend_detection.hold_duration)       \
  I(suppressor.dominant_nearend_detection.trigger_threshold)   \
  I(suppressor.dominant_nearend_detection.use_during_initial_phase) \
  I(suppressor.dominant_nearend_detection.use_unbounded_echo_spectrum) \
  I(suppressor.subband_nearend_detection.nearend_average_blocks) \
  I(suppressor.subband_nearend_detection.subband1.low)         \
  I(suppressor.subband_nearend_detection.subband1.high)        \
  I(suppressor.subband_nearend_detection.subband2.low)         \
  I(suppressor.subband_nearend_detection.subband2.high)        \
  F(suppressor.subband_nearend_detection.nearend_threshold)    \
  F(suppressor.subband_nearend_detection.snr_threshold)        \
  I(suppressor.use_subband_nearend_detection)                  \
  F(suppressor.high_bands_suppression.enr_threshold)           \
  F(suppressor.high_bands_suppression.max_gain_during_echo)    \
  F(suppressor.high_bands_suppression.anti_howling_activation_threshold) \
  F(suppressor.high_bands_suppression.anti_howling_gain)       \
  F(suppressor.floor_first_increase)                           \
  I(suppressor.conservative_hf_suppression)                    \
  I(multi_channel.detect_stereo_content)                       \
  F(multi_channel.stereo_detection_threshold)                  \
  I(multi_channel.stereo_detection_timeout_threshold_seconds)  \
  F(multi_channel.stereo_detection_hysteresis_seconds)

namespace erem_shim {

inline webrtc::EchoCanceller3Config Config(const float* f, const int* i) {
  webrtc::EchoCanceller3Config cfg;
  int fi = 0, ii = 0;
#define EREM_F(x) cfg.x = f[fi++];
#define EREM_I(x) cfg.x = static_cast<decltype(cfg.x)>(i[ii++]);
  EREM_CONFIG(EREM_F, EREM_I)
#undef EREM_F
#undef EREM_I
  return cfg;
}

struct Base {
  virtual ~Base() = default;
  virtual void SetRender(int slot, const float* blocks, const float* fft, const float* spectrum) = 0;
  virtual void SetIndices(int write, int read) = 0;
  virtual void Process(int gain_change, int delay_change, int clock_drift, int saturation, const int* ext_delay,
                       float* capture, float* linear) = 0;
  virtual void SetLeakage(int leakage) = 0;
  virtual void SetOutputUsage(int used) = 0;
  virtual void GetMetrics(double* out) = 0;
};

// R is webrtc::EchoRemover or a renamed source build of it.
template <class R>
struct Impl : Base {
  webrtc::EchoCanceller3Config cfg;
  webrtc::Environment env;
  int nb, nr, nc;
  webrtc::BlockBuffer bb;
  webrtc::SpectrumBuffer sb;
  webrtc::FftBuffer fb;
  webrtc::RenderBuffer rb;
  std::unique_ptr<R> er;
  webrtc::Block capture;
  webrtc::Block linear;

  Impl(const float* f, const int* i, int rate, int num_render, int num_capture, int size)
      : cfg(Config(f, i)),
        env(webrtc::CreateEnvironment()),
        nb(webrtc::NumBandsForRate(rate)),
        nr(num_render),
        nc(num_capture),
        bb(size, nb, num_render),
        sb(size, num_render),
        fb(size, num_render),
        rb(&bb, &sb, &fb),
        er(R::Create(env, cfg, rate, num_render, num_capture)),
        capture(nb, num_capture),
        linear(1, num_capture) {}

  void SetRender(int slot, const float* blocks, const float* fft, const float* spectrum) override {
    for (int b = 0; b < nb; ++b)
      for (int ch = 0; ch < nr; ++ch) {
        const float* src = blocks + (b * nr + ch) * 64;
        std::copy(src, src + 64, bb.buffer[slot].begin(b, ch));
      }
    for (int ch = 0; ch < nr; ++ch) {
      auto& X = fb.buffer[slot][ch];
      std::copy(fft + ch * 130, fft + ch * 130 + 65, X.re.begin());
      std::copy(fft + ch * 130 + 65, fft + ch * 130 + 130, X.im.begin());
      std::copy(spectrum + ch * 65, spectrum + ch * 65 + 65, sb.buffer[slot][ch].begin());
    }
  }
  void SetIndices(int write, int read) override {
    bb.write = sb.write = fb.write = write;
    bb.read = sb.read = fb.read = read;
  }
  void Process(int gain_change, int delay_change, int clock_drift, int saturation, const int* ext_delay,
               float* y, float* lin) override {
    for (int b = 0; b < nb; ++b)
      for (int ch = 0; ch < nc; ++ch) {
        const float* src = y + (b * nc + ch) * 64;
        std::copy(src, src + 64, capture.begin(b, ch));
      }
    webrtc::EchoPathVariability v(gain_change != 0,
                                  static_cast<webrtc::EchoPathVariability::DelayAdjustment>(delay_change),
                                  clock_drift != 0);
    std::optional<webrtc::DelayEstimate> d;
    if (ext_delay) {
      d = webrtc::DelayEstimate(static_cast<webrtc::DelayEstimate::Quality>(ext_delay[0]),
                                static_cast<size_t>(ext_delay[1]));
      d->blocks_since_last_change = static_cast<size_t>(ext_delay[2]);
      d->blocks_since_last_update = static_cast<size_t>(ext_delay[3]);
    }
    er->ProcessCapture(v, saturation != 0, d, &rb, lin ? &linear : nullptr, &capture);
    for (int b = 0; b < nb; ++b)
      for (int ch = 0; ch < nc; ++ch) std::copy(capture.begin(b, ch), capture.end(b, ch), y + (b * nc + ch) * 64);
    if (lin)
      for (int ch = 0; ch < nc; ++ch) std::copy(linear.begin(0, ch), linear.end(0, ch), lin + ch * 64);
  }
  void SetLeakage(int leakage) override { er->UpdateEchoLeakageStatus(leakage != 0); }
  void SetOutputUsage(int used) override { er->SetCaptureOutputUsage(used != 0); }
  void GetMetrics(double* out) override {
    webrtc::EchoControl::Metrics m{};
    er->GetMetrics(&m);
    out[0] = m.echo_return_loss;
    out[1] = m.echo_return_loss_enhancement;
  }
};

}  // namespace erem_shim

#endif
