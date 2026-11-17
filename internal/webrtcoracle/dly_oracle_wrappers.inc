// Class wrappers shared by dly_oracle.cc (prebuilt library classes) and
// dly_scalar_oracle.cc (scalar classes compiled from source under renamed
// symbols). The includer defines DLY_FN(name) and has already included the
// class headers; in the scalar unit the class names are macros resolving to
// the renamed classes.

#include <algorithm>
#include <cstring>
#include <memory>

namespace {

webrtc::EchoCanceller3Config DlyToConfig(const dly_config* c) {
  webrtc::EchoCanceller3Config cfg;
  cfg.delay.default_delay = c->default_delay;
  cfg.delay.down_sampling_factor = c->down_sampling_factor;
  cfg.delay.num_filters = c->num_filters;
  cfg.delay.delay_headroom_samples = c->delay_headroom_samples;
  cfg.delay.hysteresis_limit_blocks = c->hysteresis_limit_blocks;
  cfg.delay.delay_estimate_smoothing = c->delay_estimate_smoothing;
  cfg.delay.delay_estimate_smoothing_delay_found =
      c->delay_estimate_smoothing_delay_found;
  cfg.delay.delay_candidate_detection_threshold =
      c->delay_candidate_detection_threshold;
  cfg.delay.delay_selection_thresholds.initial = c->thresholds_initial;
  cfg.delay.delay_selection_thresholds.converged = c->thresholds_converged;
  cfg.delay.use_external_delay_estimator = c->use_external_delay_estimator != 0;
  cfg.delay.render_alignment_mixing.downmix = c->render_mix_downmix != 0;
  cfg.delay.render_alignment_mixing.adaptive_selection =
      c->render_mix_adaptive != 0;
  cfg.delay.render_alignment_mixing.activity_power_threshold =
      c->render_mix_threshold;
  cfg.delay.render_alignment_mixing.prefer_first_two_channels =
      c->render_mix_prefer_first_two != 0;
  cfg.delay.capture_alignment_mixing.downmix = c->capture_mix_downmix != 0;
  cfg.delay.capture_alignment_mixing.adaptive_selection =
      c->capture_mix_adaptive != 0;
  cfg.delay.capture_alignment_mixing.activity_power_threshold =
      c->capture_mix_threshold;
  cfg.delay.capture_alignment_mixing.prefer_first_two_channels =
      c->capture_mix_prefer_first_two != 0;
  cfg.delay.detect_pre_echo = c->detect_pre_echo != 0;
  cfg.filter.refined.length_blocks = c->refined_length_blocks;
  cfg.render_levels.active_render_limit = c->active_render_limit;
  cfg.render_levels.poor_excitation_render_limit =
      c->poor_excitation_render_limit;
  cfg.render_levels.poor_excitation_render_limit_ds8 =
      c->poor_excitation_render_limit_ds8;
  cfg.render_levels.render_power_gain_db = c->render_power_gain_db;
  cfg.buffering.excess_render_detection_interval_blocks =
      c->excess_render_detection_interval_blocks;
  cfg.buffering.max_allowed_excess_render_blocks =
      c->max_allowed_excess_render_blocks;
  return cfg;
}

void DlyFillBlock(webrtc::Block* b, const float* data) {
  for (int band = 0; band < b->NumBands(); ++band) {
    for (int ch = 0; ch < b->NumChannels(); ++ch) {
      auto v = b->View(band, ch);
      std::copy(data, data + webrtc::kBlockSize, v.begin());
      data += webrtc::kBlockSize;
    }
  }
}

struct DlyEpde {
  DlyEpde(const webrtc::EchoCanceller3Config& cfg, int ch, int size)
      : dumper(0), est(&dumper, cfg, ch), buf(size) {}
  webrtc::ApmDataDumper dumper;
  webrtc::EchoPathDelayEstimator est;
  webrtc::DownsampledRenderBuffer buf;
};

}  // namespace

extern "C" {

void* DLY_FN(epde_new)(const dly_config* c, int num_capture_channels,
                       int render_buffer_size) {
  return new DlyEpde(DlyToConfig(c), num_capture_channels, render_buffer_size);
}

void DLY_FN(epde_free)(void* p) { delete static_cast<DlyEpde*>(p); }

void DLY_FN(epde_reset)(void* p, int reset_delay_confidence) {
  static_cast<DlyEpde*>(p)->est.Reset(reset_delay_confidence != 0);
}

int DLY_FN(epde_estimate)(void* p, const float* render, int read, int write,
                          const float* capture, int num_bands,
                          int num_channels, int* quality, int* delay,
                          int* clockdrift) {
  auto* e = static_cast<DlyEpde*>(p);
  std::copy(render, render + e->buf.size, e->buf.buffer.begin());
  e->buf.read = read;
  e->buf.write = write;
  webrtc::Block block(num_bands, num_channels);
  DlyFillBlock(&block, capture);
  auto d = e->est.EstimateDelay(e->buf, block);
  *clockdrift = static_cast<int>(e->est.Clockdrift());
  if (!d) return 0;
  *quality = static_cast<int>(d->quality);
  *delay = static_cast<int>(d->delay);
  return 1;
}

void* DLY_FN(rdb_new)(const dly_config* c, int sample_rate_hz,
                      int num_render_channels) {
  return webrtc::RenderDelayBuffer::Create(DlyToConfig(c), sample_rate_hz,
                                           num_render_channels);
}

void DLY_FN(rdb_free)(void* p) {
  delete static_cast<webrtc::RenderDelayBuffer*>(p);
}

void DLY_FN(rdb_reset)(void* p) {
  static_cast<webrtc::RenderDelayBuffer*>(p)->Reset();
}

int DLY_FN(rdb_insert)(void* p, const float* block, int num_bands,
                       int num_channels) {
  webrtc::Block b(num_bands, num_channels);
  DlyFillBlock(&b, block);
  return static_cast<int>(
      static_cast<webrtc::RenderDelayBuffer*>(p)->Insert(b));
}

int DLY_FN(rdb_prepare)(void* p) {
  return static_cast<int>(
      static_cast<webrtc::RenderDelayBuffer*>(p)->PrepareCaptureProcessing());
}

void DLY_FN(rdb_skipped)(void* p) {
  static_cast<webrtc::RenderDelayBuffer*>(p)->HandleSkippedCaptureProcessing();
}

int DLY_FN(rdb_align_from_delay)(void* p, int delay) {
  return static_cast<webrtc::RenderDelayBuffer*>(p)->AlignFromDelay(delay) ? 1
                                                                           : 0;
}

void DLY_FN(rdb_align_from_external_delay)(void* p) {
  static_cast<webrtc::RenderDelayBuffer*>(p)->AlignFromExternalDelay();
}

long long DLY_FN(rdb_delay)(void* p) {
  return static_cast<long long>(
      static_cast<webrtc::RenderDelayBuffer*>(p)->Delay());
}

int DLY_FN(rdb_max_delay)(void* p) {
  return static_cast<int>(
      static_cast<webrtc::RenderDelayBuffer*>(p)->MaxDelay());
}

void DLY_FN(rdb_set_audio_buffer_delay)(void* p, int delay_ms) {
  static_cast<webrtc::RenderDelayBuffer*>(p)->SetAudioBufferDelay(delay_ms);
}

int DLY_FN(rdb_has_received_buffer_delay)(void* p) {
  return static_cast<webrtc::RenderDelayBuffer*>(p)->HasReceivedBufferDelay()
             ? 1
             : 0;
}

// sizes: ring size, bands, channels, low-rate size.
void DLY_FN(rdb_sizes)(void* p, int* sizes) {
  auto* r = static_cast<webrtc::RenderDelayBuffer*>(p);
  const webrtc::BlockBuffer& b = r->GetRenderBuffer()->GetBlockBuffer();
  sizes[0] = b.size;
  sizes[1] = b.buffer[0].NumBands();
  sizes[2] = b.buffer[0].NumChannels();
  sizes[3] = r->GetDownsampledRenderBuffer().size;
}

// idx: blocks read/write, spectra read/write, fft position, headroom,
// low-rate read/write, render activity.
void DLY_FN(rdb_state)(void* p, float* blocks, float* spectra, float* fft_re,
                       float* fft_im, float* low_rate, int* idx) {
  auto* r = static_cast<webrtc::RenderDelayBuffer*>(p);
  webrtc::RenderBuffer* rb = r->GetRenderBuffer();
  const webrtc::BlockBuffer& b = rb->GetBlockBuffer();
  const webrtc::SpectrumBuffer& s = rb->GetSpectrumBuffer();
  auto f = rb->GetFftBuffer();
  const webrtc::DownsampledRenderBuffer& l = r->GetDownsampledRenderBuffer();
  for (int i = 0; i < b.size; ++i) {
    const webrtc::Block& blk = b.buffer[i];
    for (int band = 0; band < blk.NumBands(); ++band) {
      for (int ch = 0; ch < blk.NumChannels(); ++ch) {
        auto v = blk.View(band, ch);
        std::copy(v.begin(), v.end(), blocks);
        blocks += webrtc::kBlockSize;
      }
    }
    for (size_t ch = 0; ch < s.buffer[i].size(); ++ch) {
      std::copy(s.buffer[i][ch].begin(), s.buffer[i][ch].end(), spectra);
      spectra += webrtc::kFftLengthBy2Plus1;
      std::copy(f[i][ch].re.begin(), f[i][ch].re.end(), fft_re);
      std::copy(f[i][ch].im.begin(), f[i][ch].im.end(), fft_im);
      fft_re += webrtc::kFftLengthBy2Plus1;
      fft_im += webrtc::kFftLengthBy2Plus1;
    }
  }
  std::copy(l.buffer.begin(), l.buffer.end(), low_rate);
  idx[0] = b.read;
  idx[1] = b.write;
  idx[2] = s.read;
  idx[3] = s.write;
  idx[4] = static_cast<int>(rb->Position());
  idx[5] = rb->Headroom();
  idx[6] = l.read;
  idx[7] = l.write;
  idx[8] = rb->GetRenderActivity() ? 1 : 0;
}

void* DLY_FN(rdc_new)(const dly_config* c, int sample_rate_hz,
                      int num_capture_channels) {
  return webrtc::RenderDelayController::Create(DlyToConfig(c), sample_rate_hz,
                                               num_capture_channels);
}

void DLY_FN(rdc_free)(void* p) {
  delete static_cast<webrtc::RenderDelayController*>(p);
}

void DLY_FN(rdc_reset)(void* p, int reset_delay_confidence) {
  static_cast<webrtc::RenderDelayController*>(p)->Reset(
      reset_delay_confidence != 0);
}

// out: quality, delay, blocks_since_last_change, blocks_since_last_update.
int DLY_FN(rdc_get_delay)(void* p, void* rdb, const float* capture,
                          int num_bands, int num_channels, int* out) {
  auto* c = static_cast<webrtc::RenderDelayController*>(p);
  auto* r = static_cast<webrtc::RenderDelayBuffer*>(rdb);
  webrtc::Block b(num_bands, num_channels);
  DlyFillBlock(&b, capture);
  auto d = c->GetDelay(r->GetDownsampledRenderBuffer(), r->Delay(), b);
  if (!d) return 0;
  out[0] = static_cast<int>(d->quality);
  out[1] = static_cast<int>(d->delay);
  out[2] = static_cast<int>(d->blocks_since_last_change);
  out[3] = static_cast<int>(d->blocks_since_last_update);
  return 1;
}

int DLY_FN(rdc_has_clockdrift)(void* p) {
  return static_cast<webrtc::RenderDelayController*>(p)->HasClockdrift() ? 1
                                                                         : 0;
}

}  // extern "C"
