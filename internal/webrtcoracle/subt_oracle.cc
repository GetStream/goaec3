//go:build webrtc_oracle

#include "subt_oracle.h"

#include "modules/audio_processing/aec3/subtractor.h"
#include "subt_shim.h"

subt_shim::Base* subt_new_scalar_fft(const subt_filter_config* c, int size, int num_render, int num_capture,
                                     webrtc::Aec3Optimization opt);

struct subt_sim {
  subt_shim::Base* p;
};

extern "C" {

subt_sim* subt_new(const subt_filter_config* c, int render_size, int num_render, int num_capture, int lane,
                   int scalar_fft) {
  auto opt = lane == 1 ? webrtc::Aec3Optimization::kNeon : webrtc::Aec3Optimization::kNone;
  subt_shim::Base* p =
      scalar_fft ? subt_new_scalar_fft(c, render_size, num_render, num_capture, opt)
                 : new subt_shim::Impl<webrtc::Subtractor>(c, render_size, num_render, num_capture, opt);
  return new subt_sim{p};
}
void subt_free(subt_sim* s) {
  delete s->p;
  delete s;
}
void subt_set_render(subt_sim* s, int slot, int ch, const float* block64, const float* fft130,
                     const float* spectrum65) {
  s->p->SetRender(slot, ch, block64, fft130, spectrum65);
}
void subt_set_indices(subt_sim* s, int write, int read) { s->p->SetIndices(write, read); }
void subt_handle_echo_path_change(subt_sim* s, int gain_change, int delay_change) {
  s->p->HandleEchoPathChange(gain_change, delay_change);
}
void subt_exit_initial_state(subt_sim* s) { s->p->ExitInitialState(); }
void subt_process(subt_sim* s, const float* capture, const float* Y2, int saturated, int follow_transition,
                  subt_output* out) {
  s->p->Process(capture, Y2, saturated, follow_transition, out);
}
void subt_aec_state_get(subt_sim* s, subt_aec_state* out) { s->p->AecStateGet(out); }
int subt_freq_resp(subt_sim* s, int ch, float* H2) { return s->p->FreqResp(ch, H2); }
int subt_impulse_resp(subt_sim* s, int ch, float* h) { return s->p->ImpulseResp(ch, h); }

}  // extern "C"
