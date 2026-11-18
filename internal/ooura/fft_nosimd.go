//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package ooura

// Without GOEXPERIMENT=simd the lane's pure-Go implementation runs
// (fft_fast.go); it is bit-identical to the lane reference and to the
// archsimd kernel.

func fft(a *[128]float32)        { fftFast(a) }
func inverseFft(a *[128]float32) { inverseFftFast(a) }
