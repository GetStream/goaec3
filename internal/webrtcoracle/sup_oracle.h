#ifndef GOAEC3_SUP_ORACLE_H
#define GOAEC3_SUP_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// ComfortNoiseGenerator. Spectra are [ch][65]; FftData is {re[65], im[65]}.
void* sup_cng_new(float noise_floor_dbfs, int optimization, int num_ch);
void sup_cng_compute(void* p, int saturated, const float* Y2, float* low,
                     float* high, float* N2_out);
void sup_cng_free(void* p);

// SuppressionFilter. e is [band][ch][64]; read and written. scalar selects
// the from-source build linked against the scalar Ooura FFT.
void* sup_sf_new(int optimization, int rate, int num_ch, int scalar);
void sup_sf_apply(void* p, const float* cn, const float* cn_high,
                  const float* gain, float high_bands_gain, const float* E,
                  float* e, int num_bands, int num_ch, int scalar);
void sup_sf_free(void* p, int scalar);
void* sup_sf_scalar_new(int optimization, int rate, int num_ch);
void sup_sf_scalar_apply(void* p, const float* cn, const float* cn_high,
                         const float* gain, float high_bands_gain,
                         const float* E, float* e, int num_bands, int num_ch);
void sup_sf_scalar_free(void* p);

// Nearend detectors (NearendDetector*).
void* sup_dne_new(float enr_threshold, float enr_exit_threshold,
                  float snr_threshold, int hold_duration, int trigger_threshold,
                  int use_during_initial_phase, int num_ch);
void* sup_sne_new(int nearend_average_blocks, int s1_low, int s1_high,
                  int s2_low, int s2_high, float nearend_threshold,
                  float snr_threshold, int num_ch);
int sup_ne_update(void* p, const float* nearend, const float* echo,
                  const float* noise, int num_ch, int initial_state);
void sup_ne_free(void* p);

// MultiChannelContentDetector. frame is [band][ch][n].
void* sup_mccd_new(int detect_stereo_content, int num_render_ch,
                   float detection_threshold, int timeout_seconds,
                   float hysteresis_seconds);
int sup_mccd_update(void* p, const float* frame, int num_bands, int num_ch,
                    int n, int* proper, int* temporary);
void sup_mccd_free(void* p);

// ConfigSelector. The mono config keeps the default
// filter.refined.length_blocks; the multichannel one uses mc_marker.
void* sup_cs_new(int detect_stereo_content, int has_multichannel,
                 int mc_marker, int num_render_ch);
void sup_cs_update(void* p, int multichannel_content);
int sup_cs_active_marker(void* p);
void sup_cs_free(void* p);

#ifdef __cplusplus
}
#endif
#endif
