//go:build amd64

package aec3

import "golang.org/x/sys/cpu"

// hasAVX2 reports libwebrtc's kAVX2 condition from rtc_base/cpu_info.cc: AVX,
// XSAVE, OSXSAVE with the OS enabling the YMM state, AVX2 and BMI2. Builds
// with and without GOEXPERIMENT=simd use this same routine, so both select
// the same lane on a given machine. (cpu.X86.HasAVX already requires OSXSAVE
// and XGETBV reporting YMM state.)
func hasAVX2() bool {
	return cpu.X86.HasAVX && cpu.X86.HasAVX2 && cpu.X86.HasBMI2 && cpu.X86.HasOSXSAVE
}
