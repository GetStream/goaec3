#ifndef GOAEC3_EREM_ORACLE_H
#define GOAEC3_EREM_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// An EchoRemover together with the multi-band RenderBuffer it reads.
// kind: 0 = the prebuilt webrtc::EchoRemover::Create (DetectOptimization(),
// i.e. kNeon on arm64); 1 = echo_remover.cc built from source on the prebuilt
// dependencies with the lane forced to `lane`; 2 = echo_remover.cc,
// subtractor.cc, adaptive_fir_filter.cc, suppression_filter.cc and aec3_fft.cc
// built from source on the scalar Ooura FFT with the lane forced to `lane`.
// lane: 0 = kNone, 1 = kNeon. The config is the flattened
// EchoCanceller3Config (erem_shim.h lists the order).
typedef struct erem_sim erem_sim;
erem_sim* erem_new(const float* cfg_f, const int* cfg_i, int rate, int num_render, int num_capture, int render_size,
                   int kind, int lane);
void erem_free(erem_sim* s);
// blocks: [num_bands][num_render][64]; fft: [num_render][re 65, im 65];
// spectrum: [num_render][65].
void erem_set_render(erem_sim* s, int slot, const float* blocks, const float* fft, const float* spectrum);
void erem_set_indices(erem_sim* s, int write, int read);
// ext_delay: NULL or {quality, delay, blocks_since_last_change,
// blocks_since_last_update}. capture: [num_bands][num_capture][64], in/out.
// linear: NULL or [num_capture][64] out.
void erem_process(erem_sim* s, int gain_change, int delay_change, int clock_drift, int saturation,
                  const int* ext_delay, float* capture, float* linear);
void erem_set_leakage(erem_sim* s, int leakage);
void erem_set_output_usage(erem_sim* s, int used);
// out: {echo_return_loss, echo_return_loss_enhancement}.
void erem_get_metrics(erem_sim* s, double* out);

float erem_log10f(float x);
int erem_transform_db_metric(int negate, float min_value, float max_value, float offset, float scaling, float value);

#ifdef __cplusplus
}
#endif
#endif
