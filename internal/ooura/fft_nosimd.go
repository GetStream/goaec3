//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package ooura

// Without GOEXPERIMENT=simd the lane's pure-Go reference runs; it is
// bit-identical to the archsimd kernel, only slower.

func fft(a *[128]float32)        { fftRef(a) }
func inverseFft(a *[128]float32) { inverseFftRef(a) }
