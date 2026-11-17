//go:build webrtc_oracle

// Scalar (Aec3Optimization::kNone, scalar Ooura) reference build of the
// render/delay path classes, compiled from source under renamed symbols so
// they do not clash with the NEON build in libwebrtc.a. Used by the purego
// lane. The scalar Ooura class OouraFftScalarRef comes from ooura_scalar.cc.
#ifndef WEBRTC_APM_DEBUG_DUMP
#define WEBRTC_APM_DEBUG_DUMP 0
#endif
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang fp contract(off)
#undef WEBRTC_HAS_NEON

#define DetectOptimization DlyScalarDetectOptimization
#define OouraFft OouraFftScalarRef
#define Aec3Fft DlyScalarAec3Fft
#define MatchedFilter DlyScalarMatchedFilter
#define MatchedFilterCore DlyScalarMatchedFilterCore
#define MaxSquarePeakIndex DlyScalarMaxSquarePeakIndex
#define MatchedFilterLagAggregator DlyScalarMatchedFilterLagAggregator
#define EchoPathDelayEstimator DlyScalarEchoPathDelayEstimator
#define RenderDelayBuffer DlyScalarRenderDelayBuffer
#define RenderDelayController DlyScalarRenderDelayController

#include "dly_oracle.h"

#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_fft.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/matched_filter.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/matched_filter_lag_aggregator.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/echo_path_delay_estimator.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/render_delay_buffer.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/render_delay_controller.cc"

namespace webrtc {
Aec3Optimization DlyScalarDetectOptimization() {
  return Aec3Optimization::kNone;
}
}  // namespace webrtc

#define DLY_FN(name) dly_scalar_##name
#include "dly_oracle_wrappers.inc"
