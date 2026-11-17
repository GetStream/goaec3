#ifndef GOAEC3_DLY_ORACLE_H
#define GOAEC3_DLY_ORACLE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// The EchoCanceller3Config fields read by the render/delay path.
typedef struct {
  int default_delay;
  int down_sampling_factor;
  int num_filters;
  int delay_headroom_samples;
  int hysteresis_limit_blocks;
  float delay_estimate_smoothing;
  float delay_estimate_smoothing_delay_found;
  float delay_candidate_detection_threshold;
  int thresholds_initial;
  int thresholds_converged;
  int use_external_delay_estimator;
  int render_mix_downmix;
  int render_mix_adaptive;
  float render_mix_threshold;
  int render_mix_prefer_first_two;
  int capture_mix_downmix;
  int capture_mix_adaptive;
  float capture_mix_threshold;
  int capture_mix_prefer_first_two;
  int detect_pre_echo;
  int refined_length_blocks;
  float active_render_limit;
  float poor_excitation_render_limit;
  float poor_excitation_render_limit_ds8;
  float render_power_gain_db;
  int excess_render_detection_interval_blocks;
  int max_allowed_excess_render_blocks;
} dly_config;

// lane: 0 = aec3::MatchedFilterCore, 1 = aec3::MatchedFilterCore_NEON,
// 2 = aec3::MatchedFilterCoreWithAccumulatedError_NEON.
void dly_mf_core(int lane, size_t x_start_index, float x2_sum_threshold,
                 float smoothing, const float* x, int x_size, const float* y,
                 int y_size, float* h, int h_size, int* filters_updated,
                 float* error_sum, int compute_accumulated_error,
                 float* accumulated_error, int accumulated_error_size,
                 float* scratch, int scratch_size);

void* dly_mf_new(int optimization, int sub_block_size,
                 int window_size_sub_blocks, int num_matched_filters,
                 int alignment_shift_sub_blocks, float excitation_limit,
                 float smoothing_fast, float smoothing_slow,
                 float matching_filter_threshold, int detect_pre_echo,
                 int render_buffer_size);
void dly_mf_free(void* p);
void dly_mf_reset(void* p, int full_reset);
int dly_mf_max_filter_lag(void* p);
// Returns 1 when a lag estimate is available.
int dly_mf_update(void* p, const float* render, int read, int write,
                  const float* capture, int capture_size,
                  int use_slow_smoothing, int* lag, int* pre_echo_lag);

void* dly_decimator_new(int down_sampling_factor);
void dly_decimator_free(void* p);
void dly_decimator_decimate(void* p, const float* in, float* out, int out_size);

void* dly_frame_blocker_new(int num_bands, int num_channels);
void dly_frame_blocker_free(void* p);
void dly_frame_blocker_insert(void* p, const float* sub_frame, float* block);
int dly_frame_blocker_available(void* p);
void dly_frame_blocker_extract(void* p, float* block);

void* dly_block_framer_new(int num_bands, int num_channels);
void dly_block_framer_free(void* p);
void dly_block_framer_insert(void* p, const float* block);
void dly_block_framer_insert_extract(void* p, const float* block,
                                     float* sub_frame);

void* dly_block_delay_buffer_new(int num_channels, int sample_rate_hz,
                                 int delay_samples);
void dly_block_delay_buffer_free(void* p);
// frame is [band][channel][160].
void dly_block_delay_buffer_delay(void* p, float* frame);

void* dly_alignment_mixer_new(int num_channels, int downmix,
                              int adaptive_selection,
                              float activity_power_threshold,
                              int prefer_first_two_channels);
void dly_alignment_mixer_free(void* p);
// block is [channel][64] (one band); y receives 64 samples.
void dly_alignment_mixer_produce(void* p, const float* block,
                                 int num_channels, float* y);

void* dly_clockdrift_new(void);
void dly_clockdrift_free(void* p);
int dly_clockdrift_update(void* p, int delay_estimate);

// Class wrappers built twice: dly_* against the prebuilt library (its
// DetectOptimization() lane), dly_scalar_* against the scalar sources.
#define DLY_DECLARE_CLASS_WRAPPERS(P)                                         \
  void* P##epde_new(const dly_config* c, int num_capture_channels,            \
                    int render_buffer_size);                                  \
  void P##epde_free(void* p);                                                 \
  void P##epde_reset(void* p, int reset_delay_confidence);                    \
  int P##epde_estimate(void* p, const float* render, int read, int write,     \
                       const float* capture, int num_bands, int num_channels, \
                       int* quality, int* delay, int* clockdrift);            \
  void* P##rdb_new(const dly_config* c, int sample_rate_hz,                   \
                   int num_render_channels);                                  \
  void P##rdb_free(void* p);                                                  \
  void P##rdb_reset(void* p);                                                 \
  int P##rdb_insert(void* p, const float* block, int num_bands,               \
                    int num_channels);                                        \
  int P##rdb_prepare(void* p);                                                \
  void P##rdb_skipped(void* p);                                               \
  int P##rdb_align_from_delay(void* p, int delay);                            \
  void P##rdb_align_from_external_delay(void* p);                             \
  long long P##rdb_delay(void* p);                                            \
  int P##rdb_max_delay(void* p);                                              \
  void P##rdb_set_audio_buffer_delay(void* p, int delay_ms);                  \
  int P##rdb_has_received_buffer_delay(void* p);                              \
  void P##rdb_sizes(void* p, int* sizes);                                     \
  void P##rdb_state(void* p, float* blocks, float* spectra, float* fft_re,    \
                    float* fft_im, float* low_rate, int* idx);                \
  void* P##rdc_new(const dly_config* c, int sample_rate_hz,                   \
                   int num_capture_channels);                                 \
  void P##rdc_free(void* p);                                                  \
  void P##rdc_reset(void* p, int reset_delay_confidence);                     \
  int P##rdc_get_delay(void* p, void* rdb, const float* capture,              \
                       int num_bands, int num_channels, int* out);            \
  int P##rdc_has_clockdrift(void* p);

DLY_DECLARE_CLASS_WRAPPERS(dly_)
DLY_DECLARE_CLASS_WRAPPERS(dly_scalar_)

#ifdef __cplusplus
}
#endif
#endif
