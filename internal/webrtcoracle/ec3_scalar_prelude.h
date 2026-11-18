// Prelude of the scalar AEC3 source build (ec3_scalar_*.cc).
//
// Every aec3 .cc file is compiled in its own translation unit (so that the
// anonymous-namespace helpers of different files cannot clash) with every
// namespace-scope AEC3 name renamed (Ec3s prefix), so that none of its
// symbols clash with the NEON build in libwebrtc.a. Everything outside
// modules/audio_processing/aec3 is included here first, unrenamed, so those
// classes (EchoCanceller3Config, Environment, AudioBuffer, HighPassFilter,
// ApmDataDumper, ...) come from libwebrtc.a.
//
// The build runs Aec3Optimization::kNone (DetectOptimization is replaced in
// ec3_scalar_aec3_common.cc) and the scalar Ooura FFT (OouraFftScalarRef
// from ooura_scalar.cc). WEBRTC_HAS_NEON stays defined; every NEON kernel in
// aec3 is selected at run time from the optimization value.
#ifndef GOAEC3_EC3_SCALAR_PRELUDE_H
#define GOAEC3_EC3_SCALAR_PRELUDE_H

// libwebrtc.a is a release build: RTC_DCHECKs are off. Some reachable states
// (e.g. an external audio buffer delay larger than the render buffer, which
// the ring index arithmetic wraps) trip a DCHECK, so the source build turns
// them off too.
#ifndef NDEBUG
#define NDEBUG
#endif

#pragma clang fp contract(off)
#pragma clang diagnostic ignored "-Wnullability-completeness"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arm_neon.h>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/audio/audio_processing.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/field_trials_view.h"
#include "common_audio/include/audio_util.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/high_pass_filter.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "modules/audio_processing/utility/cascaded_biquad_filter.h"
#include "rtc_base/checks.h"
#include "rtc_base/cpu_info.h"
#include "rtc_base/experiments/field_trial_parser.h"
#include "rtc_base/gtest_prod_util.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_minmax.h"
#include "rtc_base/race_checker.h"
#include "rtc_base/swap_queue.h"
#include "rtc_base/system/arch.h"
#include "rtc_base/thread_annotations.h"
#include "system_wrappers/include/metrics.h"

// The scalar Ooura FFT, defined in ooura_scalar.cc.
#define OouraFft OouraFftScalarRef
#include "common_audio/third_party/ooura/fft_size_128/ooura_fft.h"

// Every namespace-scope AEC3 name (from the symbols of the aec3 objects in
// libwebrtc.a) and the aec3 sub-namespace get an Ec3s prefix.
#define aec3 ec3s_aec3
#define AdaptiveFirFilter Ec3sAdaptiveFirFilter
#define AdjustConfig Ec3sAdjustConfig
#define Aec3Fft Ec3sAec3Fft
#define Aec3Optimization Ec3sAec3Optimization
#define Aec3RenderQueueItemVerifier Ec3sAec3RenderQueueItemVerifier
#define AecState Ec3sAecState
#define AlignmentMixer Ec3sAlignmentMixer
#define ApiCallJitterMetrics Ec3sApiCallJitterMetrics
#define Block Ec3sBlock
#define BlockBuffer Ec3sBlockBuffer
#define BlockDelayBuffer Ec3sBlockDelayBuffer
#define BlockFramer Ec3sBlockFramer
#define BlockProcessor Ec3sBlockProcessor
#define BlockProcessorMetrics Ec3sBlockProcessorMetrics
#define ClockdriftDetector Ec3sClockdriftDetector
#define CoarseFilterUpdateGain Ec3sCoarseFilterUpdateGain
#define ComfortNoiseGenerator Ec3sComfortNoiseGenerator
#define ConfigSelector Ec3sConfigSelector
#define Decimator Ec3sDecimator
#define DelayEstimate Ec3sDelayEstimate
#define DetectOptimization Ec3sDetectOptimization
#define DominantNearendDetector Ec3sDominantNearendDetector
#define DownsampledRenderBuffer Ec3sDownsampledRenderBuffer
#define EchoAudibility Ec3sEchoAudibility
#define EchoCanceller3 Ec3sEchoCanceller3
#define EchoPathDelayEstimator Ec3sEchoPathDelayEstimator
#define EchoPathVariability Ec3sEchoPathVariability
#define EchoRemover Ec3sEchoRemover
#define EchoRemoverMetrics Ec3sEchoRemoverMetrics
#define ErlEstimator Ec3sErlEstimator
#define ErleEstimator Ec3sErleEstimator
#define FastApproxLog2f Ec3sFastApproxLog2f
#define FftBuffer Ec3sFftBuffer
#define FftData Ec3sFftData
#define FilterAnalyzer Ec3sFilterAnalyzer
#define FrameBlocker Ec3sFrameBlocker
#define FullBandErleEstimator Ec3sFullBandErleEstimator
#define LegacyTransparentModeImpl Ec3sLegacyTransparentModeImpl
#define Log2TodB Ec3sLog2TodB
#define MatchedFilter Ec3sMatchedFilter
#define MatchedFilterLagAggregator Ec3sMatchedFilterLagAggregator
#define MovingAverage Ec3sMovingAverage
#define MultiChannelContentDetector Ec3sMultiChannelContentDetector
#define NearendDetector Ec3sNearendDetector
#define RefinedFilterUpdateGain Ec3sRefinedFilterUpdateGain
#define RenderBuffer Ec3sRenderBuffer
#define RenderDelayBuffer Ec3sRenderDelayBuffer
#define RenderDelayController Ec3sRenderDelayController
#define RenderDelayControllerMetrics Ec3sRenderDelayControllerMetrics
#define RenderSignalAnalyzer Ec3sRenderSignalAnalyzer
#define ResidualEchoEstimator Ec3sResidualEchoEstimator
#define ReverbDecayEstimator Ec3sReverbDecayEstimator
#define ReverbFrequencyResponse Ec3sReverbFrequencyResponse
#define ReverbModel Ec3sReverbModel
#define ReverbModelEstimator Ec3sReverbModelEstimator
#define SignalDependentErleEstimator Ec3sSignalDependentErleEstimator
#define SpectrumBuffer Ec3sSpectrumBuffer
#define StationarityEstimator Ec3sStationarityEstimator
#define SubbandErleEstimator Ec3sSubbandErleEstimator
#define SubbandNearendDetector Ec3sSubbandNearendDetector
#define Subtractor Ec3sSubtractor
#define SubtractorOutput Ec3sSubtractorOutput
#define SubtractorOutputAnalyzer Ec3sSubtractorOutputAnalyzer
#define SuppressionFilter Ec3sSuppressionFilter
#define SuppressionGain Ec3sSuppressionGain
#define TransparentMode Ec3sTransparentMode
#define TransparentModeImpl Ec3sTransparentModeImpl

#endif  // GOAEC3_EC3_SCALAR_PRELUDE_H
