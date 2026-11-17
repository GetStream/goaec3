#ifndef GOAEC3_EC3_ORACLE_H
#define GOAEC3_EC3_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

#define EC3_MAX_BANDS 3
#define EC3_MAX_CHANNELS 8

// One BlockProcessor -> EchoRemover::ProcessCapture call as seen by the
// recording echo remover.
typedef struct {
  int called;
  int gain_change;
  int delay_change;
  int clock_drift;
  int saturation;
  int has_delay;
  int delay;
  int quality;
  int blocks_since_last_change;
  int blocks_since_last_update;
  int position;
  int render_activity;
  // RenderBuffer::GetBlock(0), [band][channel][64].
  float block[EC3_MAX_BANDS * EC3_MAX_CHANNELS * 64];
} ec3_bp_record;

// The config is passed as its leaves in declaration order (see
// ec3_config_fields.inc), each as a double.
int ec3_config_num_fields(void);

// Prebuilt libwebrtc.
void* ec3_bp_new(const double* cfg, int rate, int num_render, int num_capture);
void ec3_bp_free(void* p);
void ec3_bp_buffer_render(void* p, const float* block);
void ec3_bp_process_capture(void* p, int gain_change, int saturation, float* capture, float* linear,
                            ec3_bp_record* rec);
void ec3_bp_set_audio_buffer_delay(void* p, int delay_ms);
void ec3_bp_metrics(void* p, double* erl, double* erle, int* delay_ms);

void* ec3_canceller_new(const double* cfg, const double* mc_cfg, int rate, int num_render, int num_capture);
void ec3_canceller_free(void* p);
// FloatS16 full-band frames, [channel][rate/100].
void ec3_canceller_analyze_render(void* p, const float* render);
void ec3_canceller_process_capture(void* p, float* capture, float* linear, int level_change);
// [-1, 1] full-band frames through AudioBuffer::CopyFrom / CopyTo.
void ec3_canceller_analyze_render_float(void* p, const float* render);
void ec3_canceller_process_capture_float(void* p, float* capture, float* linear, int level_change);
void ec3_canceller_metrics(void* p, double* erl, double* erle, int* delay_ms);
void ec3_canceller_set_audio_buffer_delay(void* p, int delay_ms);
void ec3_canceller_set_capture_output_usage(void* p, int used);

// Scalar (Aec3Optimization::kNone, scalar Ooura) source build, same API.
void* ec3s_bp_new(const double* cfg, int rate, int num_render, int num_capture);
void ec3s_bp_free(void* p);
void ec3s_bp_buffer_render(void* p, const float* block);
void ec3s_bp_process_capture(void* p, int gain_change, int saturation, float* capture, float* linear,
                             ec3_bp_record* rec);
void ec3s_bp_set_audio_buffer_delay(void* p, int delay_ms);
void ec3s_bp_metrics(void* p, double* erl, double* erle, int* delay_ms);

void* ec3s_canceller_new(const double* cfg, const double* mc_cfg, int rate, int num_render, int num_capture);
void ec3s_canceller_free(void* p);
void ec3s_canceller_analyze_render(void* p, const float* render);
void ec3s_canceller_process_capture(void* p, float* capture, float* linear, int level_change);
void ec3s_canceller_analyze_render_float(void* p, const float* render);
void ec3s_canceller_process_capture_float(void* p, float* capture, float* linear, int level_change);
void ec3s_canceller_metrics(void* p, double* erl, double* erle, int* delay_ms);
void ec3s_canceller_set_audio_buffer_delay(void* p, int delay_ms);
void ec3s_canceller_set_capture_output_usage(void* p, int used);

#ifdef __cplusplus
}
#endif
#endif
