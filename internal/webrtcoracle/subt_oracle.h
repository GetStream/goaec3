#ifndef GOAEC3_SUBT_ORACLE_H
#define GOAEC3_SUBT_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// EchoCanceller3Config::Filter. Refined configurations are {length_blocks,
// leakage_converged, leakage_diverged, error_floor, error_ceil, noise_gate},
// coarse configurations {length_blocks, rate, noise_gate}.
typedef struct {
  float refined[6];
  float refined_initial[6];
  float coarse[3];
  float coarse_initial[3];
  int config_change_duration_blocks;
  float initial_state_seconds;
  int coarse_reset_hangover_blocks;
  int conservative_initial_phase;
  int enable_coarse_filter_output_usage;
  int use_linear_filter;
} subt_filter_config;

// Same layout as aec3.SubtractorOutput.
typedef struct {
  float s_refined[64];
  float s_coarse[64];
  float e_refined[64];
  float e_coarse[64];
  float E_refined_re[65];
  float E_refined_im[65];
  float E2_refined[65];
  float E2_coarse[65];
  float s2_refined;
  float s2_coarse;
  float e2_refined;
  float e2_coarse;
  float y2;
  float s_refined_max_abs;
  float s_coarse_max_abs;
} subt_output;

typedef struct {
  int min_direct_path_filter_delay;
  int transition_triggered;
  int saturated_capture;
  int usable_linear_estimate;
  float fullband_erle_log2;
} subt_aec_state;

// A webrtc::Subtractor together with the RenderBuffer (1 band), the
// RenderSignalAnalyzer and the AecState it is driven with. lane: 0 = kNone,
// 1 = kNeon. scalar_fft selects a source build of subtractor.cc whose
// Aec3Fft (and the AdaptiveFirFilters') runs the scalar Ooura FFT.
typedef struct subt_sim subt_sim;
subt_sim* subt_new(const subt_filter_config* c, int render_size, int num_render, int num_capture, int lane,
                   int scalar_fft);
void subt_free(subt_sim* s);
void subt_set_render(subt_sim* s, int slot, int ch, const float* block64, const float* fft130,
                     const float* spectrum65);
void subt_set_indices(subt_sim* s, int write, int read);
void subt_handle_echo_path_change(subt_sim* s, int gain_change, int delay_change);
void subt_exit_initial_state(subt_sim* s);
// One EchoRemover-like step: AecState::UpdateCaptureSaturation,
// RenderSignalAnalyzer::Update, ExitInitialState on a triggered transition
// (when follow_transition), Subtractor::Process, then AecState::Update with
// E2 = the outputs' E2_refined and the given Y2. capture and Y2 are
// [num_capture][64] and [num_capture][65].
void subt_process(subt_sim* s, const float* capture, const float* Y2, int saturated, int follow_transition,
                  subt_output* out);
void subt_aec_state_get(subt_sim* s, subt_aec_state* out);
// Return the number of partitions / samples written.
int subt_freq_resp(subt_sim* s, int ch, float* H2);
int subt_impulse_resp(subt_sim* s, int ch, float* h);

#ifdef __cplusplus
}
#endif
#endif
