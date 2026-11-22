//go:build purego || !(arm64 || amd64)

package ooura

// The purego lane and architectures without a WebRTC SIMD path run the
// scalar C code of ooura_fft.cc.

func fftRef(a *[128]float32)        { fftScalar(a) }
func inverseFftRef(a *[128]float32) { inverseFftScalar(a) }

func fftSplitFast(a *[128]float32, re, im *[65]float32) { fftSplitFastScalar(a, re, im) }
func inverseFftSplitFast(re, im *[65]float32, a *[128]float32) {
	inverseFftSplitFastScalar(re, im, a)
}
