//go:build amd64 && !goexperiment.simd

package aec3

import "golang.org/x/sys/cpu"

// hasAVX2 reports libwebrtc's kAVX2 condition (cpu_info.cc: AVX, OS XSAVE
// support, AVX2 and BMI2). It still selects the AVX2 lane order without
// GOEXPERIMENT=simd; the lane then runs as its pure-Go reference.
func hasAVX2() bool { return cpu.X86.HasAVX && cpu.X86.HasAVX2 && cpu.X86.HasBMI2 && cpu.X86.HasOSXSAVE }
