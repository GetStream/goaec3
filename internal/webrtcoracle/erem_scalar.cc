//go:build webrtc_oracle

// Source build of echo_remover.cc together with its FFT-using dependencies
// (subtractor.cc, adaptive_fir_filter.cc, suppression_filter.cc, aec3_fft.cc)
// on the scalar Ooura FFT (OouraFftScalarRef from ooura_scalar.cc), with
// DetectOptimization() replaced by the lane forced through erem_new. This is
// the reference for the purego lane. The aec3 kernels keep their NEON
// variants (WEBRTC_HAS_NEON stays defined). Symbols are renamed so they do
// not clash with libwebrtc.a.
#pragma clang fp contract(off)

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "api/environment/environment.h"
#include "api/field_trials_view.h"
#include "erem_shim.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter_erl.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/coarse_filter_update_gain.h"
#include "modules/audio_processing/aec3/comfort_noise_generator.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/echo_remover_metrics.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/refined_filter_update_gain.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/residual_echo_estimator.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/aec3/suppression_gain.h"
#include "modules/audio_processing/aec3/vector_math.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/checks.h"
#include "rtc_base/cpu_info.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_minmax.h"

#define DetectOptimization EremScalarForcedOptimization
#define OouraFft OouraFftScalarRef
#define Aec3Fft EremAec3FftScalarRef
#define AdaptiveFirFilter EremAdaptiveFirFilterScalarRef
#define Subtractor EremSubtractorScalarRef
#define SuppressionFilter EremSuppressionFilterScalarRef
#define EchoRemover EremEchoRemoverScalarRef
#define IsSse2Available EremIsSse2AvailableScalarRef

namespace webrtc {
Aec3Optimization EremScalarForcedOptimization() {
  return erem_forced_lane == 1 ? Aec3Optimization::kNeon : Aec3Optimization::kNone;
}
}  // namespace webrtc

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_fft.cc"

#define aec3 erem_aec3_scalar_ref
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/adaptive_fir_filter.cc"
#undef aec3

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/subtractor.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/suppression_filter.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/echo_remover.cc"

erem_shim::Base* erem_new_scalar(const float* f, const int* i, int rate, int nr, int nc, int size) {
  return new erem_shim::Impl<webrtc::EremEchoRemoverScalarRef>(f, i, rate, nr, nc, size);
}
