//go:build amd64 && goexperiment.simd

package aec3

import "simd/archsimd"

// hasAVX2 reports libwebrtc's kAVX2 condition (cpu_info.cc: AVX, OS XSAVE
// support and AVX2). libwebrtc also requires BMI2, which archsimd does not
// report; every AVX2 CPU in practice has it.
func hasAVX2() bool { return archsimd.X86.AVX2() }
