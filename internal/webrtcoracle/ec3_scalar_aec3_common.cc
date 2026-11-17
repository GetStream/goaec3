//go:build webrtc_oracle

// Scalar source build of aec3/aec3_common.cc (see ec3_scalar_prelude.h),
// with DetectOptimization fixed to Aec3Optimization::kNone.
#include "ec3_scalar_prelude.h"

#include "modules/audio_processing/aec3/aec3_common.h"

#undef DetectOptimization
#define DetectOptimization Ec3sNativeDetectOptimization
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_common.cc"
#undef DetectOptimization
#define DetectOptimization Ec3sDetectOptimization

namespace webrtc {
Aec3Optimization DetectOptimization() {
  return Aec3Optimization::kNone;
}
}  // namespace webrtc
