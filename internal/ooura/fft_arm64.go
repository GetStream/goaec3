//go:build arm64 && !purego

package ooura

func fft(a *[128]float32)        { fftNeonRef(a) }
func inverseFft(a *[128]float32) { inverseFftNeonRef(a) }
