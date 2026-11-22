//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package ooura

// Without GOEXPERIMENT=simd the lane's pure-Go implementation runs
// (fft_fast.go); it is bit-identical to the lane reference and to the
// archsimd kernel.

func initSimdTables() {}

// Scratch is the work area of FftSplit and InverseFftSplit; this build
// transforms in place and needs none.
type Scratch struct{}

func fftSplit(a *[128]float32, re, im *[65]float32, _ *Scratch) { fftSplitFast(a, re, im) }
func inverseFftSplit(re, im *[65]float32, a *[128]float32, _ *Scratch) {
	inverseFftSplitFast(re, im, a)
}
