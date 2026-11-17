//go:build webrtc_oracle

#include "erem_oracle.h"

#include <cmath>

#include "erem_shim.h"
#include "modules/audio_processing/aec3/echo_remover.h"
#include "modules/audio_processing/aec3/echo_remover_metrics.h"

int erem_forced_lane = 0;

erem_shim::Base* erem_new_src(const float* f, const int* i, int rate, int nr, int nc, int size);
erem_shim::Base* erem_new_scalar(const float* f, const int* i, int rate, int nr, int nc, int size);

struct erem_sim {
  erem_shim::Base* p;
};

extern "C" {

erem_sim* erem_new(const float* cfg_f, const int* cfg_i, int rate, int num_render, int num_capture, int render_size,
                   int kind, int lane) {
  erem_forced_lane = lane;
  erem_shim::Base* p = nullptr;
  switch (kind) {
    case 0:
      p = new erem_shim::Impl<webrtc::EchoRemover>(cfg_f, cfg_i, rate, num_render, num_capture, render_size);
      break;
    case 1:
      p = erem_new_src(cfg_f, cfg_i, rate, num_render, num_capture, render_size);
      break;
    default:
      p = erem_new_scalar(cfg_f, cfg_i, rate, num_render, num_capture, render_size);
      break;
  }
  return new erem_sim{p};
}
void erem_free(erem_sim* s) {
  delete s->p;
  delete s;
}
void erem_set_render(erem_sim* s, int slot, const float* blocks, const float* fft, const float* spectrum) {
  s->p->SetRender(slot, blocks, fft, spectrum);
}
void erem_set_indices(erem_sim* s, int write, int read) { s->p->SetIndices(write, read); }
void erem_process(erem_sim* s, int gain_change, int delay_change, int clock_drift, int saturation,
                  const int* ext_delay, float* capture, float* linear) {
  s->p->Process(gain_change, delay_change, clock_drift, saturation, ext_delay, capture, linear);
}
void erem_set_leakage(erem_sim* s, int leakage) { s->p->SetLeakage(leakage); }
void erem_set_output_usage(erem_sim* s, int used) { s->p->SetOutputUsage(used); }
void erem_get_metrics(erem_sim* s, double* out) { s->p->GetMetrics(out); }

float erem_log10f(float x) { return std::log10(x); }
int erem_transform_db_metric(int negate, float min_value, float max_value, float offset, float scaling, float value) {
  return webrtc::aec3::TransformDbMetricForReporting(negate != 0, min_value, max_value, offset, scaling, value);
}

}  // extern "C"
