//go:build amd64

package aec3

import "golang.org/x/sys/cpu"

// hasAVX2 reports the kAVX2 condition of rtc_base/cpu_info.cc: AVX with the
// YMM state enabled by the OS, AVX2 and BMI2. Builds with and without
// GOEXPERIMENT=simd share it, so they select the same lane.
func hasAVX2() bool {
	return cpu.X86.HasAVX && cpu.X86.HasAVX2 && cpu.X86.HasBMI2 && cpu.X86.HasOSXSAVE
}
