//go:build arm64 && !purego

package ooura

// The arm64 lane is libwebrtc's NEON build (ooura_fft_neon.cc).

func fftRef(a *[128]float32)        { fftNeonRef(a) }
func inverseFftRef(a *[128]float32) { inverseFftNeonRef(a) }

func fftSplitFast(a *[128]float32, re, im *[65]float32) { fftSplitFastLane(a, re, im) }
func inverseFftSplitFast(re, im *[65]float32, a *[128]float32) {
	inverseFftSplitFastLane(re, im, a)
}
