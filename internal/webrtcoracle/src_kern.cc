//go:build webrtc_src_oracle && !webrtc_oracle

// Per-kernel entry points into the source build: the matched filter core and
// the adaptive filter kernels of every lane this architecture compiles
// (scalar everywhere, SSE2 and AVX2 on x86, NEON on arm64).
#include "src_prelude.h"

#include <algorithm>
#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter_erl.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block_buffer.h"
#include "modules/audio_processing/aec3/fft_buffer.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/matched_filter.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "src_kern.h"

namespace {

constexpr int kNone = 0, kSse2 = 1, kAvx2 = 2, kNeon = 3;

bool HasLane(int lane) {
  switch (lane) {
    case kNone:
      return true;
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
    case kAvx2:
      return true;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      return true;
#endif
  }
  return false;
}

void LoadFft(const float* p, webrtc::FftData* d) {
  std::copy(p, p + 65, d->re.begin());
  std::copy(p + 65, p + 130, d->im.begin());
}

void StoreFft(const webrtc::FftData& d, float* p) {
  std::copy(d.re.begin(), d.re.end(), p);
  std::copy(d.im.begin(), d.im.end(), p + 65);
}

void LoadH(const float* p, int partitions, int channels, std::vector<std::vector<webrtc::FftData>>* h) {
  h->assign(partitions, std::vector<webrtc::FftData>(channels));
  for (int i = 0; i < partitions; ++i) {
    for (int ch = 0; ch < channels; ++ch) {
      LoadFft(p + (i * channels + ch) * 130, &(*h)[i][ch]);
    }
  }
}

void StoreH(const std::vector<std::vector<webrtc::FftData>>& h, float* p) {
  const int channels = h.empty() ? 0 : static_cast<int>(h[0].size());
  for (size_t i = 0; i < h.size(); ++i) {
    for (int ch = 0; ch < channels; ++ch) {
      StoreFft(h[i][ch], p + (i * channels + ch) * 130);
    }
  }
}

}  // namespace

struct src_kern_rb {
  webrtc::BlockBuffer bb;
  webrtc::SpectrumBuffer sb;
  webrtc::FftBuffer fb;
  webrtc::RenderBuffer rb;
  src_kern_rb(int size, int channels) : bb(size, 1, channels), sb(size, channels), fb(size, channels), rb(&bb, &sb, &fb) {}
};

extern "C" {

int src_kern_has_lane(int lane) { return HasLane(lane) ? 1 : 0; }

int src_kern_mf_core(int lane, int x_start_index, float x2_sum_threshold, float smoothing, const float* x,
                     int x_size, const float* y, int y_size, float* h, int h_size, int* filters_updated,
                     float* error_sum, int compute_accumulated_error, float* accumulated_error,
                     int accumulated_error_size, float* scratch, int scratch_size) {
  bool updated = *filters_updated != 0;
  webrtc::ArrayView<const float> xv(x, x_size);
  webrtc::ArrayView<const float> yv(y, y_size);
  webrtc::ArrayView<float> hv(h, h_size);
  webrtc::ArrayView<float> av(accumulated_error, accumulated_error_size);
  webrtc::ArrayView<float> sv(scratch, scratch_size);
  const bool acc = compute_accumulated_error != 0;
  const size_t xs = static_cast<size_t>(x_start_index);
  switch (lane) {
    case kNone:
      webrtc::aec3::MatchedFilterCore(xs, x2_sum_threshold, smoothing, xv, yv, hv, &updated, error_sum, acc, av);
      break;
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
      webrtc::aec3::MatchedFilterCore_SSE2(xs, x2_sum_threshold, smoothing, xv, yv, hv, &updated, error_sum, acc,
                                           av, sv);
      break;
    case kAvx2:
      webrtc::aec3::MatchedFilterCore_AVX2(xs, x2_sum_threshold, smoothing, xv, yv, hv, &updated, error_sum, acc,
                                           av, sv);
      break;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      webrtc::aec3::MatchedFilterCore_NEON(xs, x2_sum_threshold, smoothing, xv, yv, hv, &updated, error_sum, acc,
                                           av, sv);
      break;
#endif
    default:
      return -1;
  }
  *filters_updated = updated ? 1 : 0;
  return 0;
}

src_kern_rb* src_kern_rb_new(int size, int num_channels) { return new src_kern_rb(size, num_channels); }
void src_kern_rb_free(src_kern_rb* rb) { delete rb; }
void src_kern_rb_set_fft(src_kern_rb* rb, int index, int ch, const float* fft130) {
  LoadFft(fft130, &rb->fb.buffer[index][ch]);
}
void src_kern_rb_set_read(src_kern_rb* rb, int read) { rb->fb.read = read; }

int src_kern_apply_filter(src_kern_rb* rb, int lane, int num_partitions, const float* H, int h_partitions,
                          int num_channels, float* S) {
  if (!HasLane(lane)) return -1;
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  webrtc::FftData s;
  const size_t np = static_cast<size_t>(num_partitions);
  switch (lane) {
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
      webrtc::aec3::ApplyFilter_Sse2(rb->rb, np, h, &s);
      break;
    case kAvx2:
      webrtc::aec3::ApplyFilter_Avx2(rb->rb, np, h, &s);
      break;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      webrtc::aec3::ApplyFilter_Neon(rb->rb, np, h, &s);
      break;
#endif
    default:
      webrtc::aec3::ApplyFilter(rb->rb, np, h, &s);
  }
  StoreFft(s, S);
  return 0;
}

int src_kern_adapt_partitions(src_kern_rb* rb, int lane, const float* G, int num_partitions, float* H,
                              int h_partitions, int num_channels) {
  if (!HasLane(lane)) return -1;
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  webrtc::FftData g;
  LoadFft(G, &g);
  const size_t np = static_cast<size_t>(num_partitions);
  switch (lane) {
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
      webrtc::aec3::AdaptPartitions_Sse2(rb->rb, g, np, &h);
      break;
    case kAvx2:
      webrtc::aec3::AdaptPartitions_Avx2(rb->rb, g, np, &h);
      break;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      webrtc::aec3::AdaptPartitions_Neon(rb->rb, g, np, &h);
      break;
#endif
    default:
      webrtc::aec3::AdaptPartitions(rb->rb, g, np, &h);
  }
  StoreH(h, H);
  return 0;
}

int src_kern_freq_resp(int lane, int num_partitions, const float* H, int h_partitions, int num_channels,
                       float* H2) {
  if (!HasLane(lane)) return -1;
  std::vector<std::vector<webrtc::FftData>> h;
  LoadH(H, h_partitions, num_channels, &h);
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> h2(h_partitions);
  for (int p = 0; p < h_partitions; ++p) std::copy(H2 + p * 65, H2 + p * 65 + 65, h2[p].begin());
  const size_t np = static_cast<size_t>(num_partitions);
  switch (lane) {
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
      webrtc::aec3::ComputeFrequencyResponse_Sse2(np, h, &h2);
      break;
    case kAvx2:
      webrtc::aec3::ComputeFrequencyResponse_Avx2(np, h, &h2);
      break;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      webrtc::aec3::ComputeFrequencyResponse_Neon(np, h, &h2);
      break;
#endif
    default:
      webrtc::aec3::ComputeFrequencyResponse(np, h, &h2);
  }
  for (int p = 0; p < h_partitions; ++p) std::copy(h2[p].begin(), h2[p].end(), H2 + p * 65);
  return 0;
}

int src_kern_erl(int lane, const float* H2, int n, float* erl) {
  if (!HasLane(lane)) return -1;
  std::vector<std::array<float, webrtc::kFftLengthBy2Plus1>> h2(n);
  for (int p = 0; p < n; ++p) std::copy(H2 + p * 65, H2 + p * 65 + 65, h2[p].begin());
  webrtc::ArrayView<float> e(erl, 65);
  switch (lane) {
#if defined(WEBRTC_ARCH_X86_FAMILY)
    case kSse2:
      webrtc::aec3::ErlComputer_SSE2(h2, e);
      break;
    case kAvx2:
      webrtc::aec3::ErlComputer_AVX2(h2, e);
      break;
#endif
#if defined(WEBRTC_HAS_NEON)
    case kNeon:
      webrtc::aec3::ErlComputer_NEON(h2, e);
      break;
#endif
    default:
      webrtc::aec3::ErlComputer(h2, e);
  }
  return 0;
}

}  // extern "C"
