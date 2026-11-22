//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package ooura

func initSimdTables() {}

// Scratch is the work area of FftSplit and InverseFftSplit. This build
// runs fft_fast.go, which needs none.
type Scratch struct{}

func fftSplit(a *[128]float32, re, im *[65]float32, _ *Scratch) { fftSplitFast(a, re, im) }
func inverseFftSplit(re, im *[65]float32, a *[128]float32, _ *Scratch) {
	inverseFftSplitFast(re, im, a)
}
