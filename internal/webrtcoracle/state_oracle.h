#ifndef GOAEC3_STATE_ORACLE_H
#define GOAEC3_STATE_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// Subset of EchoCanceller3Config read by AecState and its estimators.
typedef struct {
  int refined_length_blocks;
  int refined_initial_length_blocks;
  int use_linear_filter;
  int conservative_initial_phase;
  float initial_state_seconds;
  float erle_min;
  float erle_max_l;
  float erle_max_h;
  int erle_onset_detection;
  int erle_num_sections;
  int erle_clamp_quality_estimate_to_zero;
  int erle_clamp_quality_estimate_to_one;
  float default_gain;
  float default_len;
  float nearend_len;
  int echo_can_saturate;
  int bounded_erl;
  int erle_onset_compensation_in_dominant_nearend;
  int use_conservative_tail_frequency_response;
  int use_stationarity_properties;
  int use_stationarity_properties_at_init;
  float active_render_limit;
  int delay_headroom_samples;
  int linear_and_stable_echo_path;
  int noise_floor_hold;
  float min_noise_floor_power;
  float stationary_gate_slope;
  float noise_gate_power;
  float noise_gate_slope;
  int render_pre_window_size;
  int render_post_window_size;
  int model_reverb_in_nonlinear_mode;
} st_config;

// Render buffer (block, spectrum and FFT rings) behind a RenderBuffer view.
void* st_render_new(int size, int num_channels);
void st_render_free(void* r);
void st_render_set_block(void* r, int slot, int ch, const float* x);
void st_render_set_spectrum(void* r, int slot, int ch, const float* x2);
void st_render_set_indices(void* r, int write, int read);

void* st_moving_average_new(int num_elem, int mem_len);
void st_moving_average_free(void* p);
void st_moving_average_average(void* p, const float* in, float* out, int n);

void* st_reverb_model_new(void);
void st_reverb_model_free(void* p);
void st_reverb_model_update_no_freq_shaping(void* p, const float* ps, int n, float scaling, float decay);
void st_reverb_model_update(void* p, const float* ps, const float* scaling, int n, float decay);
void st_reverb_model_reverb(void* p, float* out);

void* st_reverb_frequency_response_new(int conservative);
void st_reverb_frequency_response_free(void* p);
void st_reverb_frequency_response_update(void* p, const float* fr, int num_blocks, int delay, int q_ok, float q,
                                         int stationary);
void st_reverb_frequency_response_get(void* p, float* out);

void* st_reverb_decay_new(const st_config* c);
void st_reverb_decay_free(void* p);
void st_reverb_decay_update(void* p, const float* filter, int n, int q_ok, float q, int delay, int usable,
                            int stationary);
float st_reverb_decay_decay(void* p, int mild);

void* st_erl_new(int startup);
void st_erl_free(void* p);
void st_erl_reset(void* p);
void st_erl_update(void* p, const int* converged, int num_capture, const float* render, int num_render,
                   const float* capture);
void st_erl_get(void* p, float* erl, float* erl_td);

void* st_fullband_erle_new(const st_config* c, int nch);
void st_fullband_erle_free(void* p);
void st_fullband_erle_reset(void* p);
void st_fullband_erle_update(void* p, const float* X2, const float* Y2, const float* E2, const int* converged);
float st_fullband_erle_log2(void* p);
void st_fullband_erle_qualities(void* p, float* q, int* ok);

void* st_subband_erle_new(const st_config* c, int nch);
void st_subband_erle_free(void* p);
void st_subband_erle_reset(void* p);
void st_subband_erle_update(void* p, const float* X2, const float* Y2, const float* E2, const int* converged);
// which: 0 erle, 1 onset compensated, 2 unbounded, 3 during onsets.
void st_subband_erle_get(void* p, int which, float* out);

void* st_sd_erle_new(const st_config* c, int nch);
void st_sd_erle_free(void* p);
void st_sd_erle_reset(void* p);
void st_sd_erle_update(void* p, void* render, const float* fr, int num_blocks, const float* X2, const float* Y2,
                       const float* E2, const float* avg_erle, const float* avg_erle_oc, const int* converged);
void st_sd_erle_get(void* p, int onset_compensated, float* out);

void* st_erle_new(const st_config* c, int startup, int nch);
void st_erle_free(void* p);
void st_erle_reset(void* p, int delay_change);
void st_erle_update(void* p, void* render, const float* fr, int num_blocks, const float* X2, const float* Y2,
                    const float* E2, const int* converged);
// which: 0 erle, 1 onset compensated, 2 unbounded, 3 during onsets.
void st_erle_get(void* p, int which, float* out);
float st_erle_fullband_log2(void* p);
void st_erle_qualities(void* p, float* q, int* ok);

void* st_stationarity_new(void);
void st_stationarity_free(void* p);
void st_stationarity_update_noise(void* p, const float* spectrum, int nch);
void st_stationarity_update_flags(void* p, void* render, const float* reverb, int idx_current, int num_lookahead);
void st_stationarity_flags(void* p, int* band_stationary, int* block_stationary);

void* st_echo_audibility_new(int at_init);
void st_echo_audibility_free(void* p);
void st_echo_audibility_update(void* p, void* render, const float* reverb, int delay, int external_delay_seen);
void st_echo_audibility_get(void* p, int converged, float* scaling, int* block_stationary);

void* st_filter_analyzer_new(const st_config* c, int nch);
void st_filter_analyzer_free(void* p);
void st_filter_analyzer_reset(void* p);
void st_filter_analyzer_update(void* p, const float* filters, int len, void* render, int* consistent, float* gain);
void st_filter_analyzer_get(void* p, int* delays, int* min_delay, int* length_blocks, float* adjusted);

void* st_subtractor_analyzer_new(int nch);
void st_subtractor_analyzer_free(void* p);
void st_subtractor_analyzer_update(void* p, const float* y2, const float* e2r, const float* e2c, int* out3,
                                   int* converged);
void st_subtractor_analyzer_handle_echo_path_change(void* p);

// hmm selects the WebRTC-Aec3TransparentModeHmm field trial. Returns NULL
// when TransparentMode::Create returns no classifier.
void* st_transparent_mode_new(const st_config* c, int hmm);
void st_transparent_mode_free(void* p);
void st_transparent_mode_reset(void* p);
int st_transparent_mode_update(void* p, int delay, int consistent, int converged, int coarse_converged, int diverged,
                               int active_render, int saturated_capture);

typedef struct {
  float erle[8][65];
  float erle_oc[8][65];
  float erle_unbounded[8][65];
  float erl[65];
  float residual_scaling[65];
  float reverb_freq_resp[65];
  float erl_time_domain;
  float fullband_erle_log2;
  float reverb_decay;
  float reverb_decay_mild;
  int usable_linear_estimate;
  int use_linear_filter_output;
  int active_render;
  int use_stationarity_properties;
  int min_direct_path_filter_delay;
  int saturated_capture;
  int saturated_echo;
  int transparent_mode_active;
  int transition_triggered;
  int filter_length_blocks;
} st_aec_state_out;

void* st_aec_state_new(const st_config* c, int nch, int hmm);
void st_aec_state_free(void* p);
void st_aec_state_handle_echo_path_change(void* p, int gain_change, int delay_change);
void st_aec_state_update_capture_saturation(void* p, int saturated);
// subtractor: per channel {y2, e2_refined, e2_coarse, s_refined_max_abs, s_coarse_max_abs}.
void st_aec_state_update(void* p, int ext_valid, int ext_quality, int ext_delay, const float* fr, int num_blocks,
                         const float* ir, int ir_len, void* render, const float* E2, const float* Y2,
                         const float* subtractor);
void st_aec_state_get(void* p, st_aec_state_out* out);

void* st_residual_echo_new(const st_config* c, int num_render);
void st_residual_echo_free(void* p);
void st_residual_echo_estimate(void* p, void* aec, void* render, const float* S2, const float* Y2, int nch,
                               int dominant_nearend, float* R2, float* R2u);

#ifdef __cplusplus
}
#endif
#endif
