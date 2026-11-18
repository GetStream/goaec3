//go:build webrtc_src_oracle && !webrtc_oracle

// The infrastructure the webrtc_src_oracle build replaces: the Environment
// (only its field trials are used by AEC3; every trial reads as unset, i.e.
// the defaults, like the prebuilt libwebrtc without field trials configured)
// and cpu_info::Supports, which answers from the lane set by
// goaec3_src_set_lane. libwebrtc's own CPU detection (rtc_base/cpu_info.cc)
// is compiled unmodified in src_cpu_info.cc for the lane check.
#include "src_prelude.h"

#include <atomic>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/field_trials_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/cpu_info.h"
#include "src_oracle.h"

namespace {

class NoFieldTrials : public webrtc::FieldTrialsView {
 public:
  std::string Lookup(absl::string_view) const override { return ""; }
};

std::atomic<int> g_lane{0};

}  // namespace

namespace webrtc {

// Only field_trials() is reachable from AEC3; the clock, task queue factory
// and event log are null and would fault on use.
Environment EnvironmentFactory::Create() const {
  static const NoFieldTrials kNoFieldTrials;
  return Environment(nullptr, field_trials_ ? field_trials_ : &kNoFieldTrials, clock_, task_queue_factory_,
                     event_log_);
}

namespace cpu_info {

uint32_t DetectNumberOfCores() { return 1; }

bool Supports(ISA isa) {
  const auto lane = static_cast<Aec3Optimization>(g_lane.load());
  switch (isa) {
    case ISA::kSSE2:
    case ISA::kSSE3:
      return lane == Aec3Optimization::kSse2 || lane == Aec3Optimization::kAvx2;
    case ISA::kAVX2:
    case ISA::kFMA3:
      return lane == Aec3Optimization::kAvx2;
    case ISA::kNeon:
      return lane == Aec3Optimization::kNeon;
  }
  return false;
}

}  // namespace cpu_info
}  // namespace webrtc

extern "C" {

void goaec3_src_set_lane(int lane) { g_lane.store(lane); }

int goaec3_src_detect_optimization(void) { return static_cast<int>(webrtc::DetectOptimization()); }

int goaec3_src_cpu_has_fma(void) {
#if defined(__x86_64__)
  return __builtin_cpu_supports("fma") ? 1 : 0;
#else
  return 0;
#endif
}

}  // extern "C"
