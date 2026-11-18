//go:build webrtc_src_oracle && !webrtc_oracle

// libwebrtc's lane selection, unmodified: rtc_base/cpu_info.cc and
// aec3_common.cc's DetectOptimization, renamed so that they sit next to the
// build's cpu_info::Supports (src_stubs.cc), which answers from the Go lane.
#include "src_prelude.h"

#define cpu_info goaec3_real_cpu_info
#define DetectOptimization Goaec3RealDetectOptimization
#define FastApproxLog2f Goaec3RealFastApproxLog2f
#define Log2TodB Goaec3RealLog2TodB
#include "../../tmp_check/webrtc-m141/rtc_base/cpu_info.cc"
#include "../../tmp_check/webrtc-m141/modules/audio_processing/aec3/aec3_common.cc"

#include "src_oracle.h"

extern "C" int goaec3_src_cpu_info_lane(void) { return static_cast<int>(webrtc::Goaec3RealDetectOptimization()); }
