//go:build webrtc_oracle

// Render/delay path entry points against the prebuilt libwebrtc classes.
#ifndef WEBRTC_APM_DEBUG_DUMP
#define WEBRTC_APM_DEBUG_DUMP 0
#endif
#pragma clang diagnostic ignored "-Wnullability-completeness"

#include "dly_oracle.h"

#include <algorithm>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/alignment_mixer.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/clockdrift_detector.h"
#include "modules/audio_processing/aec3/block_delay_buffer.h"
#include "modules/audio_processing/aec3/block_framer.h"
#include "modules/audio_processing/aec3/decimator.h"
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"
#include "modules/audio_processing/aec3/echo_path_delay_estimator.h"
#include "modules/audio_processing/aec3/frame_blocker.h"
#include "modules/audio_processing/aec3/matched_filter.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/aec3/render_delay_controller.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace webrtc {
namespace aec3 {
// Defined in matched_filter.cc but not declared in its header.
void MatchedFilterCoreWithAccumulatedError_NEON(
    size_t x_start_index,
    float x2_sum_threshold,
    float smoothing,
    ArrayView<const float> x,
    ArrayView<const float> y,
    ArrayView<float> h,
    bool* filters_updated,
    float* error_sum,
    ArrayView<float> accumulated_error,
    ArrayView<float> scratch_memory);
}  // namespace aec3
}  // namespace webrtc

namespace {

struct DlyMf {
  DlyMf(int optimization, int sub_block_size, int window_size_sub_blocks,
        int num_matched_filters, int alignment_shift_sub_blocks,
        float excitation_limit, float smoothing_fast, float smoothing_slow,
        float matching_filter_threshold, bool detect_pre_echo,
        int render_buffer_size)
      : dumper(0),
        mf(&dumper, static_cast<webrtc::Aec3Optimization>(optimization),
           sub_block_size, window_size_sub_blocks, num_matched_filters,
           alignment_shift_sub_blocks, excitation_limit, smoothing_fast,
           smoothing_slow, matching_filter_threshold, detect_pre_echo),
        buf(render_buffer_size) {}
  webrtc::ApmDataDumper dumper;
  webrtc::MatchedFilter mf;
  webrtc::DownsampledRenderBuffer buf;
};

struct DlyBlocker {
  DlyBlocker(int bands, int channels)
      : blocker(bands, channels), block(bands, channels) {}
  webrtc::FrameBlocker blocker;
  webrtc::Block block;
};

struct DlyFramer {
  DlyFramer(int bands, int channels)
      : framer(bands, channels), block(bands, channels) {}
  webrtc::BlockFramer framer;
  webrtc::Block block;
};

struct DlyBdb {
  DlyBdb(int channels, int rate, int delay)
      : audio(rate, channels, rate, channels, rate, channels),
        bands(rate / 16000),
        bdb(channels, rate / 16000, 160, delay) {}
  webrtc::AudioBuffer audio;
  int bands;
  webrtc::BlockDelayBuffer bdb;
};

}  // namespace

extern "C" {

void dly_mf_core(int lane, size_t x_start_index, float x2_sum_threshold,
                 float smoothing, const float* x, int x_size, const float* y,
                 int y_size, float* h, int h_size, int* filters_updated,
                 float* error_sum, int compute_accumulated_error,
                 float* accumulated_error, int accumulated_error_size,
                 float* scratch, int scratch_size) {
  bool updated = *filters_updated != 0;
  webrtc::ArrayView<const float> xv(x, x_size);
  webrtc::ArrayView<const float> yv(y, y_size);
  webrtc::ArrayView<float> hv(h, h_size);
  webrtc::ArrayView<float> av(accumulated_error, accumulated_error_size);
  webrtc::ArrayView<float> sv(scratch, scratch_size);
  switch (lane) {
    case 0:
      webrtc::aec3::MatchedFilterCore(x_start_index, x2_sum_threshold,
                                      smoothing, xv, yv, hv, &updated,
                                      error_sum, compute_accumulated_error != 0,
                                      av);
      break;
    case 1:
      webrtc::aec3::MatchedFilterCore_NEON(
          x_start_index, x2_sum_threshold, smoothing, xv, yv, hv, &updated,
          error_sum, compute_accumulated_error != 0, av, sv);
      break;
    default:
      webrtc::aec3::MatchedFilterCoreWithAccumulatedError_NEON(
          x_start_index, x2_sum_threshold, smoothing, xv, yv, hv, &updated,
          error_sum, av, sv);
      break;
  }
  *filters_updated = updated ? 1 : 0;
}

void* dly_mf_new(int optimization, int sub_block_size,
                 int window_size_sub_blocks, int num_matched_filters,
                 int alignment_shift_sub_blocks, float excitation_limit,
                 float smoothing_fast, float smoothing_slow,
                 float matching_filter_threshold, int detect_pre_echo,
                 int render_buffer_size) {
  return new DlyMf(optimization, sub_block_size, window_size_sub_blocks,
                   num_matched_filters, alignment_shift_sub_blocks,
                   excitation_limit, smoothing_fast, smoothing_slow,
                   matching_filter_threshold, detect_pre_echo != 0,
                   render_buffer_size);
}

void dly_mf_free(void* p) { delete static_cast<DlyMf*>(p); }

void dly_mf_reset(void* p, int full_reset) {
  static_cast<DlyMf*>(p)->mf.Reset(full_reset != 0);
}

int dly_mf_max_filter_lag(void* p) {
  return static_cast<int>(static_cast<DlyMf*>(p)->mf.GetMaxFilterLag());
}

int dly_mf_update(void* p, const float* render, int read, int write,
                  const float* capture, int capture_size,
                  int use_slow_smoothing, int* lag, int* pre_echo_lag) {
  auto* m = static_cast<DlyMf*>(p);
  std::copy(render, render + m->buf.size, m->buf.buffer.begin());
  m->buf.read = read;
  m->buf.write = write;
  m->mf.Update(m->buf, webrtc::ArrayView<const float>(capture, capture_size),
               use_slow_smoothing != 0);
  auto e = m->mf.GetBestLagEstimate();
  if (!e) return 0;
  *lag = static_cast<int>(e->lag);
  *pre_echo_lag = static_cast<int>(e->pre_echo_lag);
  return 1;
}

void* dly_decimator_new(int down_sampling_factor) {
  return new webrtc::Decimator(down_sampling_factor);
}

void dly_decimator_free(void* p) { delete static_cast<webrtc::Decimator*>(p); }

void dly_decimator_decimate(void* p, const float* in, float* out,
                            int out_size) {
  static_cast<webrtc::Decimator*>(p)->Decimate(
      webrtc::ArrayView<const float>(in, webrtc::kBlockSize),
      webrtc::ArrayView<float>(out, out_size));
}

void* dly_frame_blocker_new(int num_bands, int num_channels) {
  return new DlyBlocker(num_bands, num_channels);
}

void dly_frame_blocker_free(void* p) { delete static_cast<DlyBlocker*>(p); }

static void DlyCopyOutBlock(const webrtc::Block& b, float* out) {
  for (int band = 0; band < b.NumBands(); ++band) {
    for (int ch = 0; ch < b.NumChannels(); ++ch) {
      auto v = b.View(band, ch);
      std::copy(v.begin(), v.end(), out);
      out += webrtc::kBlockSize;
    }
  }
}

static void DlyCopyInBlock(webrtc::Block* b, const float* in) {
  for (int band = 0; band < b->NumBands(); ++band) {
    for (int ch = 0; ch < b->NumChannels(); ++ch) {
      auto v = b->View(band, ch);
      std::copy(in, in + webrtc::kBlockSize, v.begin());
      in += webrtc::kBlockSize;
    }
  }
}

void dly_frame_blocker_insert(void* p, const float* sub_frame, float* block) {
  auto* f = static_cast<DlyBlocker*>(p);
  const int bands = f->block.NumBands();
  const int channels = f->block.NumChannels();
  std::vector<std::vector<float>> storage(bands * channels);
  std::vector<std::vector<webrtc::ArrayView<float>>> views(bands);
  for (int band = 0; band < bands; ++band) {
    for (int ch = 0; ch < channels; ++ch) {
      auto& s = storage[band * channels + ch];
      const float* src = sub_frame + (band * channels + ch) * 80;
      s.assign(src, src + 80);
      views[band].push_back(webrtc::ArrayView<float>(s.data(), 80));
    }
  }
  f->blocker.InsertSubFrameAndExtractBlock(views, &f->block);
  DlyCopyOutBlock(f->block, block);
}

int dly_frame_blocker_available(void* p) {
  return static_cast<DlyBlocker*>(p)->blocker.IsBlockAvailable() ? 1 : 0;
}

void dly_frame_blocker_extract(void* p, float* block) {
  auto* f = static_cast<DlyBlocker*>(p);
  f->blocker.ExtractBlock(&f->block);
  DlyCopyOutBlock(f->block, block);
}

void* dly_block_framer_new(int num_bands, int num_channels) {
  return new DlyFramer(num_bands, num_channels);
}

void dly_block_framer_free(void* p) { delete static_cast<DlyFramer*>(p); }

void dly_block_framer_insert(void* p, const float* block) {
  auto* f = static_cast<DlyFramer*>(p);
  DlyCopyInBlock(&f->block, block);
  f->framer.InsertBlock(f->block);
}

void dly_block_framer_insert_extract(void* p, const float* block,
                                     float* sub_frame) {
  auto* f = static_cast<DlyFramer*>(p);
  DlyCopyInBlock(&f->block, block);
  const int bands = f->block.NumBands();
  const int channels = f->block.NumChannels();
  std::vector<std::vector<webrtc::ArrayView<float>>> views(bands);
  for (int band = 0; band < bands; ++band) {
    for (int ch = 0; ch < channels; ++ch) {
      views[band].push_back(webrtc::ArrayView<float>(
          sub_frame + (band * channels + ch) * 80, 80));
    }
  }
  f->framer.InsertBlockAndExtractSubFrame(f->block, &views);
}

void* dly_block_delay_buffer_new(int num_channels, int sample_rate_hz,
                                 int delay_samples) {
  return new DlyBdb(num_channels, sample_rate_hz, delay_samples);
}

void dly_block_delay_buffer_free(void* p) { delete static_cast<DlyBdb*>(p); }

void dly_block_delay_buffer_delay(void* p, float* frame) {
  auto* d = static_cast<DlyBdb*>(p);
  const int channels = static_cast<int>(d->audio.num_channels());
  for (int band = 0; band < d->bands; ++band) {
    for (int ch = 0; ch < channels; ++ch) {
      float* dst = d->audio.split_bands(ch)[band];
      std::copy(frame + (band * channels + ch) * 160,
                frame + (band * channels + ch + 1) * 160, dst);
    }
  }
  d->bdb.DelaySignal(&d->audio);
  for (int band = 0; band < d->bands; ++band) {
    for (int ch = 0; ch < channels; ++ch) {
      const float* src = d->audio.split_bands(ch)[band];
      std::copy(src, src + 160, frame + (band * channels + ch) * 160);
    }
  }
}

void* dly_alignment_mixer_new(int num_channels, int downmix,
                              int adaptive_selection,
                              float activity_power_threshold,
                              int prefer_first_two_channels) {
  return new webrtc::AlignmentMixer(num_channels, downmix != 0,
                                    adaptive_selection != 0,
                                    activity_power_threshold,
                                    prefer_first_two_channels != 0);
}

void dly_alignment_mixer_free(void* p) {
  delete static_cast<webrtc::AlignmentMixer*>(p);
}

void dly_alignment_mixer_produce(void* p, const float* block,
                                 int num_channels, float* y) {
  webrtc::Block b(1, num_channels);
  DlyCopyInBlock(&b, block);
  static_cast<webrtc::AlignmentMixer*>(p)->ProduceOutput(
      b, webrtc::ArrayView<float, webrtc::kBlockSize>(y, webrtc::kBlockSize));
}

void* dly_clockdrift_new(void) { return new webrtc::ClockdriftDetector(); }

void dly_clockdrift_free(void* p) {
  delete static_cast<webrtc::ClockdriftDetector*>(p);
}

int dly_clockdrift_update(void* p, int delay_estimate) {
  auto* d = static_cast<webrtc::ClockdriftDetector*>(p);
  d->Update(delay_estimate);
  return static_cast<int>(d->ClockdriftLevel());
}

}  // extern "C"

#define DLY_FN(name) dly_##name
#include "dly_oracle_wrappers.inc"
