//go:build webrtc_oracle

// Source build of subtractor.cc (with adaptive_fir_filter.cc and
// aec3_fft.cc) whose Aec3Fft runs the scalar Ooura FFT (OouraFftScalarRef
// from ooura_scalar.cc). The prebuilt Subtractor always runs the NEON Ooura
// FFT; this build is the reference for the purego lane. The aec3 kernels keep
// their NEON variants (WEBRTC_HAS_NEON stays defined). Symbols are renamed so
// they do not clash with libwebrtc.a.
#pragma clang fp contract(off)

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/environment/environment.h"
#include "api/field_trials_view.h"
#include "modules/audio_processing/aec3/adaptive_fir_filter_erl.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/coarse_filter_update_gain.h"
#include "modules/audio_processing/aec3/echo_path_variability.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/refined_filter_update_gain.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"
#include "modules/audio_processing/aec3/subtractor_output.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/checks.h"
#include "rtc_base/cpu_info.h"
#include "rtc_base/numerics/safe_minmax.h"
#include "subt_shim.h"

#define OouraFft OouraFftScalarRef
#define Aec3Fft SubtAec3FftScalarRef
#define AdaptiveFirFilter SubtAdaptiveFirFilterScalarRef
#define Subtractor SubtSubtractorScalarRef
#define aec3 subt_aec3_scalar_ref
#define IsSse2Available SubtIsSse2AvailableScalarRef

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_fft.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/adaptive_fir_filter.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/subtractor.cc"

#undef OouraFft
#undef Aec3Fft
#undef AdaptiveFirFilter
#undef Subtractor
#undef aec3
#undef IsSse2Available

subt_shim::Base* subt_new_scalar_fft(const subt_filter_config* c, int size, int num_render, int num_capture,
                                     webrtc::Aec3Optimization opt) {
  return new subt_shim::Impl<webrtc::SubtSubtractorScalarRef>(c, size, num_render, num_capture, opt);
}
