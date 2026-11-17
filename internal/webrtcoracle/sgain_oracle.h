#ifndef GOAEC3_SGAIN_ORACLE_H
#define GOAEC3_SGAIN_ORACLE_H

#include "state_oracle.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tuning arrays: mask_lf {enr_transparent, enr_suppress, emr_transparent},
// mask_hf {same}, max_inc_factor, max_dec_factor_lf.
typedef struct {
  st_config st;
  int config_change_duration_blocks;
  float low_render_limit;
  float normal_render_limit;
  float floor_power;
  float audibility_threshold_lf;
  float audibility_threshold_mf;
  float audibility_threshold_hf;
  int nearend_average_blocks;
  float normal_tuning[8];
  float nearend_tuning[8];
  int lf_smoothing_during_initial_phase;
  int last_permanent_lf_smoothing_band;
  int last_lf_smoothing_band;
  int last_lf_band;
  int first_hf_band;
  float dn_enr_threshold;
  float dn_enr_exit_threshold;
  float dn_snr_threshold;
  int dn_hold_duration;
  int dn_trigger_threshold;
  int dn_use_during_initial_phase;
  int dn_use_unbounded_echo_spectrum;
  int sn_nearend_average_blocks;
  int sn_subband1_low;
  int sn_subband1_high;
  int sn_subband2_low;
  int sn_subband2_high;
  float sn_nearend_threshold;
  float sn_snr_threshold;
  int use_subband_nearend_detection;
  float hb_enr_threshold;
  float hb_max_gain_during_echo;
  float hb_anti_howling_activation_threshold;
  float hb_anti_howling_gain;
  float floor_first_increase;
  int conservative_hf_suppression;
} sg_config;

// Harness owning a one-band render buffer, AecState, RenderSignalAnalyzer
// and SuppressionGain.
void* sgain_new(const sg_config* c, int optimization, int sample_rate_hz, int num_render, int num_capture,
                int render_buffer_size);
void sgain_free(void* h);
void sgain_set_render_block(void* h, int slot, int ch, const float* x);
void sgain_set_render_spectrum(void* h, int slot, int ch, const float* x2);
void sgain_set_indices(void* h, int write, int read);
void sgain_aec_handle_echo_path_change(void* h, int gain_change, int delay_change);
void sgain_aec_update_capture_saturation(void* h, int saturated);
// subtractor: per channel {y2, e2_refined, e2_coarse, s_refined_max_abs, s_coarse_max_abs}.
void sgain_aec_update(void* h, int ext_valid, int ext_quality, int ext_delay, const float* fr, int num_blocks,
                      const float* ir, int ir_len, const float* E2, const float* Y2, const float* subtractor);
int sgain_aec_saturated_echo(void* h);
int sgain_aec_min_direct_path_filter_delay(void* h);
void sgain_rsa_update(void* h, int delay, int has_delay);
int sgain_rsa_narrow_peak_band(void* h, int* band);
void sgain_set_initial_state(void* h, int state);
// Spectra are num_capture x 65. render is [band][channel][64] with num_render
// channels. Returns IsDominantNearend() after the call.
int sgain_get_gain(void* h, const float* nearend, const float* echo, const float* residual,
                   const float* residual_unbounded, const float* comfort_noise, const float* render, int num_bands,
                   int clock_drift, float* high_bands_gain, float* low_band_gain);

#ifdef __cplusplus
}
#endif
#endif
