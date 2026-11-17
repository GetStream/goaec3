// Shared AdaptiveFirFilter shim, instantiated for the prebuilt class
// (afir_oracle.cc) and for the scalar-FFT source build (afir_scalar.cc).
#ifndef GOAEC3_AFIR_FILTER_SHIM_H
#define GOAEC3_AFIR_FILTER_SHIM_H

#include <algorithm>
#include <array>
#include <vector>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace afir_shim {

inline void LoadFft(const float* src, webrtc::FftData* d) {
  std::copy(src, src + 65, d->re.begin());
  std::copy(src + 65, src + 130, d->im.begin());
}
inline void StoreFft(const webrtc::FftData& d, float* dst) {
  std::copy(d.re.begin(), d.re.end(), dst);
  std::copy(d.im.begin(), d.im.end(), dst + 65);
}
inline void LoadH(const float* H, int partitions, int channels,
                  std::vector<std::vector<webrtc::FftData>>* out) {
  out->assign(partitions, std::vector<webrtc::FftData>(channels));
  for (int p = 0; p < partitions; ++p)
    for (int ch = 0; ch < channels; ++ch)
      LoadFft(H + (p * channels + ch) * 130, &(*out)[p][ch]);
}
inline void StoreH(const std::vector<std::vector<webrtc::FftData>>& H,
                   float* out) {
  int channels = H.empty() ? 0 : static_cast<int>(H[0].size());
  for (size_t p = 0; p < H.size(); ++p)
    for (int ch = 0; ch < channels; ++ch)
      StoreFft(H[p][ch], out + (p * channels + ch) * 130);
}

struct FilterVTable {
  void* (*create)(int max_size, int initial_size, int change_duration,
                  int num_channels, webrtc::Aec3Optimization opt);
  void (*destroy)(void*);
  void (*filter)(void*, const webrtc::RenderBuffer&, float* S);
  void (*adapt)(void*, const webrtc::RenderBuffer&, const float* G);
  int (*adapt_ir)(void*, const webrtc::RenderBuffer&, const float* G,
                  float* ir, int ir_len);
  int (*freq_resp)(void*, float* H2);
  void (*set_size)(void*, int size, int immediate);
  void (*handle_echo_path_change)(void*);
  void (*scale)(void*, float);
  void (*set_filter)(void*, int num_partitions, const float* H);
  void (*get_filter)(void*, float* H);
  int (*size)(void*);
};

template <class F>
struct FilterImpl {
  F filter;
  int max_size;
  int num_channels;
  std::vector<float> ir;
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> H2;
  FilterImpl(int max_size, int initial_size, int change_duration,
             int num_channels, webrtc::Aec3Optimization opt,
             webrtc::ApmDataDumper* dumper)
      : filter(max_size, initial_size, change_duration, num_channels, opt,
               dumper),
        max_size(max_size),
        num_channels(num_channels) {
    ir.reserve(webrtc::GetTimeDomainLength(max_size));
    H2.reserve(max_size);
  }
};

template <class F>
FilterVTable MakeVTable() {
  using I = FilterImpl<F>;
  static webrtc::ApmDataDumper dumper(0);
  FilterVTable v;
  v.create = [](int m, int i, int d, int c,
                webrtc::Aec3Optimization o) -> void* {
    return new I(m, i, d, c, o, &dumper);
  };
  v.destroy = [](void* p) { delete static_cast<I*>(p); };
  v.filter = [](void* p, const webrtc::RenderBuffer& rb, float* S) {
    webrtc::FftData s;
    static_cast<I*>(p)->filter.Filter(rb, &s);
    StoreFft(s, S);
  };
  v.adapt = [](void* p, const webrtc::RenderBuffer& rb, const float* G) {
    webrtc::FftData g;
    LoadFft(G, &g);
    static_cast<I*>(p)->filter.Adapt(rb, g);
  };
  v.adapt_ir = [](void* p, const webrtc::RenderBuffer& rb, const float* G,
                  float* ir, int ir_len) -> int {
    I* f = static_cast<I*>(p);
    webrtc::FftData g;
    LoadFft(G, &g);
    f->ir.assign(ir, ir + ir_len);
    f->filter.Adapt(rb, g, &f->ir);
    std::copy(f->ir.begin(), f->ir.end(), ir);
    return static_cast<int>(f->ir.size());
  };
  v.freq_resp = [](void* p, float* H2) -> int {
    I* f = static_cast<I*>(p);
    f->filter.ComputeFrequencyResponse(&f->H2);
    for (size_t k = 0; k < f->H2.size(); ++k)
      std::copy(f->H2[k].begin(), f->H2[k].end(), H2 + k * 65);
    return static_cast<int>(f->H2.size());
  };
  v.set_size = [](void* p, int size, int immediate) {
    static_cast<I*>(p)->filter.SetSizePartitions(size, immediate != 0);
  };
  v.handle_echo_path_change = [](void* p) {
    static_cast<I*>(p)->filter.HandleEchoPathChange();
  };
  v.scale = [](void* p, float factor) {
    static_cast<I*>(p)->filter.ScaleFilter(factor);
  };
  v.set_filter = [](void* p, int n, const float* H) {
    I* f = static_cast<I*>(p);
    std::vector<std::vector<webrtc::FftData>> h;
    LoadH(H, f->max_size, f->num_channels, &h);
    f->filter.SetFilter(n, h);
  };
  v.get_filter = [](void* p, float* H) {
    StoreH(static_cast<I*>(p)->filter.GetFilter(), H);
  };
  v.size = [](void* p) -> int {
    return static_cast<int>(static_cast<I*>(p)->filter.SizePartitions());
  };
  return v;
}

}  // namespace afir_shim

#endif
