// Shared BlockProcessor / EchoCanceller3 simulation, included by
// ec3_oracle.cc (prebuilt libwebrtc, EC3_FN(x) = ec3_x) and by
// ec3_scalar_shim.cc (scalar source build, where `webrtc` names the renamed
// namespace and EC3_FN(x) = ec3s_x).

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "api/audio/audio_processing.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "common_audio/include/audio_util.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/block_processor.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/echo_remover.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/aec3/render_delay_controller.h"
#include "modules/audio_processing/audio_buffer.h"
#include "ec3_oracle.h"

namespace {

void Ec3ApplyConfig(const double* v, webrtc::EchoCanceller3Config* cfg) {
  webrtc::EchoCanceller3Config& c = *cfg;
  int i = 0;
#define EC3_F(x) x = static_cast<std::remove_reference_t<decltype(x)>>(v[i++]);
#include "ec3_config_fields.inc"
#undef EC3_F
}

class Ec3RecordingRemover : public webrtc::EchoRemover {
 public:
  void GetMetrics(webrtc::EchoControl::Metrics* metrics) const override {
    metrics->echo_return_loss = 1.0;
    metrics->echo_return_loss_enhancement = 2.0;
  }

  void ProcessCapture(webrtc::EchoPathVariability echo_path_variability, bool capture_signal_saturation,
                      const std::optional<webrtc::DelayEstimate>& external_delay,
                      webrtc::RenderBuffer* render_buffer, webrtc::Block* linear_output,
                      webrtc::Block* capture) override {
    ec3_bp_record& r = rec;
    r.called = 1;
    r.gain_change = echo_path_variability.gain_change;
    r.delay_change = static_cast<int>(echo_path_variability.delay_change);
    r.clock_drift = echo_path_variability.clock_drift;
    r.saturation = capture_signal_saturation;
    r.has_delay = external_delay.has_value();
    if (external_delay) {
      r.delay = static_cast<int>(external_delay->delay);
      r.quality = static_cast<int>(external_delay->quality);
      r.blocks_since_last_change = static_cast<int>(external_delay->blocks_since_last_change);
      r.blocks_since_last_update = static_cast<int>(external_delay->blocks_since_last_update);
    }
    r.position = static_cast<int>(render_buffer->Position());
    r.render_activity = render_buffer->GetRenderActivity();
    const webrtc::Block& b = render_buffer->GetBlock(0);
    int i = 0;
    for (int band = 0; band < b.NumBands(); ++band) {
      for (int ch = 0; ch < b.NumChannels(); ++ch) {
        std::copy(b.begin(band, ch), b.end(band, ch), &r.block[i]);
        i += 64;
      }
    }
    for (int band = 0; band < capture->NumBands(); ++band) {
      for (int ch = 0; ch < capture->NumChannels(); ++ch) {
        for (float& x : capture->View(band, ch)) {
          x *= 0.5f;
        }
      }
    }
    if (linear_output) {
      for (int ch = 0; ch < linear_output->NumChannels(); ++ch) {
        std::copy(capture->begin(0, ch), capture->end(0, ch), linear_output->begin(0, ch));
      }
    }
  }

  void UpdateEchoLeakageStatus(bool) override {}
  void SetCaptureOutputUsage(bool) override {}

  ec3_bp_record rec;
};

struct Ec3Bp {
  int num_bands;
  int num_render;
  int num_capture;
  Ec3RecordingRemover* remover;
  std::unique_ptr<webrtc::BlockProcessor> bp;
  std::unique_ptr<webrtc::Block> linear;
};

void Ec3FillBlock(const float* src, webrtc::Block* b) {
  int i = 0;
  for (int band = 0; band < b->NumBands(); ++band) {
    for (int ch = 0; ch < b->NumChannels(); ++ch) {
      std::copy(&src[i], &src[i + 64], b->begin(band, ch));
      i += 64;
    }
  }
}

void Ec3ReadBlock(const webrtc::Block& b, float* dst) {
  int i = 0;
  for (int band = 0; band < b.NumBands(); ++band) {
    for (int ch = 0; ch < b.NumChannels(); ++ch) {
      std::copy(b.begin(band, ch), b.end(band, ch), &dst[i]);
      i += 64;
    }
  }
}

struct Ec3Canceller {
  int rate;
  int num_render;
  int num_capture;
  bool export_linear;
  webrtc::StreamConfig render_config;
  webrtc::StreamConfig capture_config;
  std::unique_ptr<webrtc::EchoCanceller3> ec;
  std::unique_ptr<webrtc::AudioBuffer> render;
  std::unique_ptr<webrtc::AudioBuffer> capture;
  std::unique_ptr<webrtc::AudioBuffer> linear;
  std::vector<const float*> render_ptrs;
  std::vector<float*> capture_ptrs;
};

// Mirrors AudioProcessingImpl: render is split when multiband, then
// analyzed; capture is analyzed full-band, split, processed and merged.
void Ec3Render(Ec3Canceller* s) {
  if (s->rate > 16000) s->render->SplitIntoFrequencyBands();
  s->ec->AnalyzeRender(s->render.get());
}

void Ec3Capture(Ec3Canceller* s, bool with_linear, int level_change) {
  s->ec->AnalyzeCapture(s->capture.get());
  if (s->rate > 16000) s->capture->SplitIntoFrequencyBands();
  if (with_linear) {
    s->ec->ProcessCapture(s->capture.get(), s->linear.get(), level_change != 0);
  } else {
    s->ec->ProcessCapture(s->capture.get(), level_change != 0);
  }
  if (s->rate > 16000) s->capture->MergeFrequencyBands();
}

}  // namespace

extern "C" {

void* EC3_FN(bp_new)(const double* cfg, int rate, int num_render, int num_capture) {
  webrtc::EchoCanceller3Config config;
  Ec3ApplyConfig(cfg, &config);
  auto* s = new Ec3Bp;
  s->num_bands = rate / 16000;
  s->num_render = num_render;
  s->num_capture = num_capture;
  s->remover = new Ec3RecordingRemover;
  std::unique_ptr<webrtc::RenderDelayBuffer> rdb(webrtc::RenderDelayBuffer::Create(config, rate, num_render));
  std::unique_ptr<webrtc::RenderDelayController> rdc;
  if (!config.delay.use_external_delay_estimator) {
    rdc.reset(webrtc::RenderDelayController::Create(config, rate, num_capture));
  }
  s->bp = webrtc::BlockProcessor::Create(config, rate, num_render, num_capture, std::move(rdb), std::move(rdc),
                                         std::unique_ptr<webrtc::EchoRemover>(s->remover));
  if (config.filter.export_linear_aec_output) {
    s->linear = std::make_unique<webrtc::Block>(1, num_capture);
  }
  return s;
}

void EC3_FN(bp_free)(void* p) { delete static_cast<Ec3Bp*>(p); }

void EC3_FN(bp_buffer_render)(void* p, const float* block) {
  auto* s = static_cast<Ec3Bp*>(p);
  webrtc::Block b(s->num_bands, s->num_render);
  Ec3FillBlock(block, &b);
  s->bp->BufferRender(b);
}

void EC3_FN(bp_process_capture)(void* p, int gain_change, int saturation, float* capture, float* linear,
                                ec3_bp_record* rec) {
  auto* s = static_cast<Ec3Bp*>(p);
  webrtc::Block b(s->num_bands, s->num_capture);
  Ec3FillBlock(capture, &b);
  std::memset(&s->remover->rec, 0, sizeof(s->remover->rec));
  s->bp->ProcessCapture(gain_change != 0, saturation != 0, s->linear.get(), &b);
  Ec3ReadBlock(b, capture);
  if (linear && s->linear) Ec3ReadBlock(*s->linear, linear);
  *rec = s->remover->rec;
}

void EC3_FN(bp_set_audio_buffer_delay)(void* p, int delay_ms) {
  static_cast<Ec3Bp*>(p)->bp->SetAudioBufferDelay(delay_ms);
}

void EC3_FN(bp_metrics)(void* p, double* erl, double* erle, int* delay_ms) {
  webrtc::EchoControl::Metrics m;
  static_cast<Ec3Bp*>(p)->bp->GetMetrics(&m);
  *erl = m.echo_return_loss;
  *erle = m.echo_return_loss_enhancement;
  *delay_ms = m.delay_ms;
}

void* EC3_FN(canceller_new)(const double* cfg, const double* mc_cfg, int rate, int num_render, int num_capture) {
  webrtc::EchoCanceller3Config config;
  Ec3ApplyConfig(cfg, &config);
  std::optional<webrtc::EchoCanceller3Config> mc;
  if (mc_cfg) {
    mc.emplace();
    Ec3ApplyConfig(mc_cfg, &*mc);
  }
  auto* s = new Ec3Canceller;
  s->rate = rate;
  s->num_render = num_render;
  s->num_capture = num_capture;
  s->export_linear = config.filter.export_linear_aec_output;
  s->render_config = webrtc::StreamConfig(rate, num_render);
  s->capture_config = webrtc::StreamConfig(rate, num_capture);
  s->ec = std::make_unique<webrtc::EchoCanceller3>(webrtc::CreateEnvironment(), config, mc, rate, num_render,
                                                   num_capture);
  s->render = std::make_unique<webrtc::AudioBuffer>(rate, num_render, rate, num_render, rate, num_render);
  s->capture = std::make_unique<webrtc::AudioBuffer>(rate, num_capture, rate, num_capture, rate, num_capture);
  s->linear = std::make_unique<webrtc::AudioBuffer>(16000, num_capture, 16000, num_capture, 16000, num_capture);
  s->render_ptrs.resize(num_render);
  s->capture_ptrs.resize(num_capture);
  return s;
}

void EC3_FN(canceller_free)(void* p) { delete static_cast<Ec3Canceller*>(p); }

void EC3_FN(canceller_analyze_render)(void* p, const float* render) {
  auto* s = static_cast<Ec3Canceller*>(p);
  const int n = s->rate / 100;
  for (int ch = 0; ch < s->num_render; ++ch) {
    std::copy(&render[ch * n], &render[(ch + 1) * n], s->render->channels()[ch]);
  }
  Ec3Render(s);
}

void EC3_FN(canceller_process_capture)(void* p, float* capture, float* linear, int level_change) {
  auto* s = static_cast<Ec3Canceller*>(p);
  const int n = s->rate / 100;
  for (int ch = 0; ch < s->num_capture; ++ch) {
    std::copy(&capture[ch * n], &capture[(ch + 1) * n], s->capture->channels()[ch]);
  }
  Ec3Capture(s, linear != nullptr, level_change);
  for (int ch = 0; ch < s->num_capture; ++ch) {
    std::copy(s->capture->channels()[ch], s->capture->channels()[ch] + n, &capture[ch * n]);
    if (linear) {
      std::copy(s->linear->channels()[ch], s->linear->channels()[ch] + 160, &linear[ch * 160]);
    }
  }
}

void EC3_FN(canceller_analyze_render_float)(void* p, const float* render) {
  auto* s = static_cast<Ec3Canceller*>(p);
  const int n = s->rate / 100;
  for (int ch = 0; ch < s->num_render; ++ch) s->render_ptrs[ch] = &render[ch * n];
  s->render->CopyFrom(s->render_ptrs.data(), s->render_config);
  Ec3Render(s);
}

void EC3_FN(canceller_process_capture_float)(void* p, float* capture, float* linear, int level_change) {
  auto* s = static_cast<Ec3Canceller*>(p);
  const int n = s->rate / 100;
  for (int ch = 0; ch < s->num_capture; ++ch) s->capture_ptrs[ch] = &capture[ch * n];
  s->capture->CopyFrom(s->capture_ptrs.data(), s->capture_config);
  // The public API always hands AEC3 a linear output buffer when the export
  // is configured.
  Ec3Capture(s, s->export_linear, level_change);
  s->capture->CopyTo(s->capture_config, s->capture_ptrs.data());
  if (linear) {
    // AudioProcessingImpl::GetLinearAecOutput.
    for (int ch = 0; ch < s->num_capture; ++ch) {
      webrtc::FloatS16ToFloat(s->linear->channels_const()[ch], 160, &linear[ch * 160]);
    }
  }
}

void EC3_FN(canceller_metrics)(void* p, double* erl, double* erle, int* delay_ms) {
  webrtc::EchoControl::Metrics m = static_cast<Ec3Canceller*>(p)->ec->GetMetrics();
  *erl = m.echo_return_loss;
  *erle = m.echo_return_loss_enhancement;
  *delay_ms = m.delay_ms;
}

void EC3_FN(canceller_set_audio_buffer_delay)(void* p, int delay_ms) {
  static_cast<Ec3Canceller*>(p)->ec->SetAudioBufferDelay(delay_ms);
}

void EC3_FN(canceller_set_capture_output_usage)(void* p, int used) {
  static_cast<Ec3Canceller*>(p)->ec->SetCaptureOutputUsage(used != 0);
}

}  // extern "C"
