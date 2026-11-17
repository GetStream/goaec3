//go:build webrtc_oracle

// Source build of adaptive_fir_filter.cc whose Aec3Fft runs the scalar Ooura
// FFT (OouraFftScalarRef from ooura_scalar.cc). The prebuilt
// AdaptiveFirFilter always constrains through the NEON Ooura FFT; this build
// is the reference for the purego lane. The aec3 kernels keep their NEON
// variants (WEBRTC_HAS_NEON stays defined). Symbols are renamed so they do
// not clash with libwebrtc.a.
#pragma clang fp contract(off)
#ifndef WEBRTC_APM_DEBUG_DUMP
#define WEBRTC_APM_DEBUG_DUMP 0
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/checks.h"
#include "rtc_base/cpu_info.h"

#define OouraFft OouraFftScalarRef
#define Aec3Fft AfirAec3FftScalarRef
#define AdaptiveFirFilter AfirAdaptiveFirFilterScalarRef
#define aec3 afir_aec3_scalar_ref
#define IsSse2Available AfirIsSse2AvailableScalarRef

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_fft.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/adaptive_fir_filter.cc"

#undef OouraFft
#undef Aec3Fft
#undef aec3
#undef IsSse2Available

#include "afir_filter_shim.h"

const afir_shim::FilterVTable* afir_scalar_fft_vtable() {
  static const afir_shim::FilterVTable v =
      afir_shim::MakeVTable<webrtc::AfirAdaptiveFirFilterScalarRef>();
  return &v;
}
