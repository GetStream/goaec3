//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

import "testing"

// dlyBenchKernels times archsimd kernels on their own; there are none in
// this build.
func dlyBenchKernels(b *testing.B, x, h0 []float32) {}
