#ifndef GOAEC3_SRC_ORACLE_H
#define GOAEC3_SRC_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// The lane (an Aec3Optimization value) cpu_info::Supports answers from.
void goaec3_src_set_lane(int lane);
// webrtc::DetectOptimization() of the source build.
int goaec3_src_detect_optimization(void);
// DetectOptimization's choice under the unmodified rtc_base/cpu_info.cc.
int goaec3_src_cpu_info_lane(void);
// CPUID advertises FMA (x86 only).
int goaec3_src_cpu_has_fma(void);

#ifdef __cplusplus
}
#endif
#endif
