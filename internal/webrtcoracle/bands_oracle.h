#ifndef GOAEC3_BANDS_ORACLE_H
#define GOAEC3_BANDS_ORACLE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// webrtc::ThreeBandFilterBank. Bands are [3][160].
void* bands_tbfb_new(void);
void bands_tbfb_analysis(void* p, const float* in480, float* out);
void bands_tbfb_synthesis(void* p, const float* in, float* out480);
void bands_tbfb_free(void* p);

// WebRtcSpl_AnalysisQMF / WebRtcSpl_SynthesisQMF.
void bands_qmf_analysis(const int16_t* in, size_t in_len, int16_t* low,
                        int16_t* high, int32_t* state1, int32_t* state2);
void bands_qmf_synthesis(const int16_t* low, const int16_t* high,
                         size_t band_len, int16_t* out, int32_t* state1,
                         int32_t* state2);

// audio_util conversions.
void bands_float_s16_to_s16(const float* src, size_t n, int16_t* dst);
void bands_s16_to_float_s16(const int16_t* src, size_t n, float* dst);
void bands_float_to_float_s16(const float* src, size_t n, float* dst);
void bands_float_s16_to_float(const float* src, size_t n, float* dst);

// webrtc::AudioBuffer at rate/channels (no resampling). data is [ch][n],
// bands is [band][ch][160].
void* bands_ab_new(int rate, int num_ch);
void bands_ab_copy_from(void* p, const float* data);
void bands_ab_split(void* p);
void bands_ab_merge(void* p);
void bands_ab_read_bands(void* p, float* bands);
void bands_ab_write_bands(void* p, const float* bands);
void bands_ab_read_data(void* p, float* data);
void bands_ab_free(void* p);

#ifdef __cplusplus
}
#endif
#endif
