//go:build webrtc_oracle

#include "afir_oracle.h"

#include <optional>

#include "afir_filter_shim.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter_erl.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/coarse_filter_update_gain.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/refined_filter_update_gain.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "modules/audio_processing/aec3/subtractor_output.h"

using afir_shim::LoadFft;
using afir_shim::LoadH;
using afir_shim::StoreFft;
using afir_shim::StoreH;

const afir_shim::FilterVTable* afir_scalar_fft_vtable();

struct afir_rb {
  webrtc::BlockBuffer bb;
  webrtc::SpectrumBuffer sb;
  webrtc::FftBuffer fb;
  webrtc::RenderBuffer rb;
  afir_rb(int size, int bands, int channels)
      : bb(size, bands, channels),
        sb(size, channels),
        fb(size, channels),
        rb(&bb, &sb, &fb) {}
};

extern "C" {

afir_rb* afir_rb_new(int size, int num_bands, int num_channels) {
  return new afir_rb(size, num_bands, num_channels);
}
void afir_rb_free(afir_rb* rb) { delete rb; }
void afir_rb_set_fft(afir_rb* rb, int index, int ch, const float* fft130) {
  LoadFft(fft130, &rb->fb.buffer[index][ch]);
}
void afir_rb_set_spectrum(afir_rb* rb, int index, int ch, const float* x65) {
  std::copy(x65, x65 + 65, rb->sb.buffer[index][ch].begin());
}
void afir_rb_set_block(afir_rb* rb, int index, int band, int ch,
                       const float* x64) {
  std::copy(x64, x64 + 64, rb->bb.buffer[index].begin(band, ch));
}
void afir_rb_set_read(afir_rb* rb, int fft_read, int spectrum_read,
                      int block_read) {
  rb->fb.read = fft_read;
  rb->sb.read = spectrum_read;
  rb->bb.read = block_read;
}

void afir_apply_filter(afir_rb* rb, int lane, int num_partitions,
                       const float* H, int h_partitions, int num_channels,
                       float* S) {
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  webrtc::FftData s;
  if (lane == 1)
    webrtc::aec3::ApplyFilter_Neon(rb->rb, num_partitions, h, &s);
  else
    webrtc::aec3::ApplyFilter(rb->rb, num_partitions, h, &s);
  StoreFft(s, S);
}

void afir_adapt_partitions(afir_rb* rb, int lane, const float* G,
                           int num_partitions, float* H, int h_partitions,
                           int num_channels) {
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  webrtc::FftData g;
  LoadFft(G, &g);
  if (lane == 1)
    webrtc::aec3::AdaptPartitions_Neon(rb->rb, g, num_partitions, &h);
  else
    webrtc::aec3::AdaptPartitions(rb->rb, g, num_partitions, &h);
  StoreH(h, H);
}

void afir_freq_resp(int lane, int num_partitions, const float* H,
                    int h_partitions, int num_channels, float* H2) {
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> h2(h_partitions);
  for (int p = 0; p < h_partitions; ++p)
    std::copy(H2 + p * 65, H2 + p * 65 + 65, h2[p].begin());
  if (lane == 1)
    webrtc::aec3::ComputeFrequencyResponse_Neon(num_partitions, h, &h2);
  else
    webrtc::aec3::ComputeFrequencyResponse(num_partitions, h, &h2);
  for (int p = 0; p < h_partitions; ++p)
    std::copy(h2[p].begin(), h2[p].end(), H2 + p * 65);
}

void afir_erl(int lane, const float* H2, int n, float* erl) {
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> h2(n);
  for (int p = 0; p < n; ++p)
    std::copy(H2 + p * 65, H2 + p * 65 + 65, h2[p].begin());
  webrtc::ArrayView<float> e(erl, 65);
  if (lane == 1)
    webrtc::aec3::ErlComputer_NEON(h2, e);
  else
    webrtc::aec3::ErlComputer(h2, e);
}

struct afir_filter {
  const afir_shim::FilterVTable* v;
  void* impl;
};

afir_filter* afir_filter_new(int scalar_fft, int max_size, int initial_size,
                             int change_duration, int num_channels, int lane) {
  static const afir_shim::FilterVTable prebuilt =
      afir_shim::MakeVTable<webrtc::AdaptiveFirFilter>();
  const afir_shim::FilterVTable* v =
      scalar_fft ? afir_scalar_fft_vtable() : &prebuilt;
  auto opt = lane == 1 ? webrtc::Aec3Optimization::kNeon
                       : webrtc::Aec3Optimization::kNone;
  return new afir_filter{
      v, v->create(max_size, initial_size, change_duration, num_channels, opt)};
}
void afir_filter_free(afir_filter* f) {
  f->v->destroy(f->impl);
  delete f;
}
void afir_filter_filter(afir_filter* f, afir_rb* rb, float* S) {
  f->v->filter(f->impl, rb->rb, S);
}
void afir_filter_adapt(afir_filter* f, afir_rb* rb, const float* G) {
  f->v->adapt(f->impl, rb->rb, G);
}
int afir_filter_adapt_ir(afir_filter* f, afir_rb* rb, const float* G,
                         float* ir, int ir_len) {
  return f->v->adapt_ir(f->impl, rb->rb, G, ir, ir_len);
}
int afir_filter_freq_resp(afir_filter* f, float* H2) {
  return f->v->freq_resp(f->impl, H2);
}
void afir_filter_set_size(afir_filter* f, int size, int immediate) {
  f->v->set_size(f->impl, size, immediate);
}
void afir_filter_handle_echo_path_change(afir_filter* f) {
  f->v->handle_echo_path_change(f->impl);
}
void afir_filter_scale(afir_filter* f, float factor) {
  f->v->scale(f->impl, factor);
}
void afir_filter_set_filter(afir_filter* f, int num_partitions,
                            const float* H) {
  f->v->set_filter(f->impl, num_partitions, H);
}
void afir_filter_get_filter(afir_filter* f, float* H) {
  f->v->get_filter(f->impl, H);
}
int afir_filter_size(afir_filter* f) { return f->v->size(f->impl); }

struct afir_rsa {
  webrtc::RenderSignalAnalyzer a;
  explicit afir_rsa(const webrtc::EchoCanceller3Config& c) : a(c) {}
};

afir_rsa* afir_rsa_new(int refined_length_blocks) {
  webrtc::EchoCanceller3Config c;
  c.filter.refined.length_blocks = refined_length_blocks;
  return new afir_rsa(c);
}
void afir_rsa_free(afir_rsa* a) { delete a; }
void afir_rsa_update(afir_rsa* a, afir_rb* rb, int delay, int has_delay) {
  std::optional<size_t> d;
  if (has_delay) d = delay;
  a->a.Update(rb->rb, d);
}
int afir_rsa_poor_signal_excitation(afir_rsa* a) {
  return a->a.PoorSignalExcitation();
}
void afir_rsa_mask(afir_rsa* a, float* v65) {
  std::array<float, 65> v;
  std::copy(v65, v65 + 65, v.begin());
  a->a.MaskRegionsAroundNarrowBands(&v);
  std::copy(v.begin(), v.end(), v65);
}
int afir_rsa_narrow_peak_band(afir_rsa* a, int* band) {
  auto b = a->a.NarrowPeakBand();
  if (!b) return 0;
  *band = *b;
  return 1;
}

struct afir_coarse {
  webrtc::CoarseFilterUpdateGain g;
  afir_coarse(const webrtc::EchoCanceller3Config::Filter::CoarseConfiguration& c,
              int d)
      : g(c, d) {}
};

afir_coarse* afir_coarse_new(int length_blocks, float rate, float noise_gate,
                             int change_duration) {
  webrtc::EchoCanceller3Config::Filter::CoarseConfiguration c{
      static_cast<size_t>(length_blocks), rate, noise_gate};
  return new afir_coarse(c, change_duration);
}
void afir_coarse_free(afir_coarse* g) { delete g; }
void afir_coarse_handle_echo_path_change(afir_coarse* g) {
  g->g.HandleEchoPathChange();
}
void afir_coarse_set_config(afir_coarse* g, int length_blocks, float rate,
                            float noise_gate, int immediate) {
  webrtc::EchoCanceller3Config::Filter::CoarseConfiguration c{
      static_cast<size_t>(length_blocks), rate, noise_gate};
  g->g.SetConfig(c, immediate != 0);
}
void afir_coarse_compute(afir_coarse* g, const float* X2, afir_rsa* a,
                         const float* E130, int size_partitions, int saturated,
                         float* G130) {
  std::array<float, 65> x2;
  std::copy(X2, X2 + 65, x2.begin());
  webrtc::FftData e, out;
  LoadFft(E130, &e);
  LoadFft(G130, &out);
  g->g.Compute(x2, a->a, e, size_partitions, saturated != 0, &out);
  StoreFft(out, G130);
}

static webrtc::EchoCanceller3Config::Filter::RefinedConfiguration
AfirRefinedConfig(const float* c) {
  return {static_cast<size_t>(c[0]), c[1], c[2], c[3], c[4], c[5]};
}

struct afir_refined {
  webrtc::RefinedFilterUpdateGain g;
  afir_refined(
      const webrtc::EchoCanceller3Config::Filter::RefinedConfiguration& c,
      int d)
      : g(c, d) {}
};

afir_refined* afir_refined_new(const float* cfg6, int change_duration) {
  return new afir_refined(AfirRefinedConfig(cfg6), change_duration);
}
void afir_refined_free(afir_refined* g) { delete g; }
void afir_refined_handle_echo_path_change(afir_refined* g, int gain_change,
                                          int delay_change, int clock_drift) {
  webrtc::EchoPathVariability v(
      gain_change != 0,
      static_cast<webrtc::EchoPathVariability::DelayAdjustment>(delay_change),
      clock_drift != 0);
  g->g.HandleEchoPathChange(v);
}
void afir_refined_set_config(afir_refined* g, const float* cfg6,
                             int immediate) {
  g->g.SetConfig(AfirRefinedConfig(cfg6), immediate != 0);
}
void afir_refined_compute(afir_refined* g, const float* X2, afir_rsa* a,
                          const float* E130, const float* E2_refined,
                          const float* E2_coarse, const float* erl,
                          int size_partitions, int saturated,
                          int disallow_leakage_diverged, float* G130) {
  std::array<float, 65> x2;
  std::copy(X2, X2 + 65, x2.begin());
  webrtc::SubtractorOutput so;
  so.Reset();
  LoadFft(E130, &so.E_refined);
  std::copy(E2_refined, E2_refined + 65, so.E2_refined.begin());
  std::copy(E2_coarse, E2_coarse + 65, so.E2_coarse.begin());
  webrtc::FftData out;
  LoadFft(G130, &out);
  g->g.Compute(x2, a->a, so, webrtc::ArrayView<const float>(erl, 65),
               size_partitions, saturated != 0, disallow_leakage_diverged != 0,
               &out);
  StoreFft(out, G130);
}

}  // extern "C"
