//go:build arm64 && !purego

package ooura

// The arm64 lane is libwebrtc's NEON build (ooura_fft_neon.cc).

func fftRef(a *[128]float32)        { fftNeonRef(a) }
func inverseFftRef(a *[128]float32) { inverseFftNeonRef(a) }

func fftFast(a *[128]float32)        { fftFastLane(a) }
func inverseFftFast(a *[128]float32) { inverseFftFastLane(a) }
