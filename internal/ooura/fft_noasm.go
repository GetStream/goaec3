//go:build purego || !(arm64 || amd64)

package ooura

func fft(a *[128]float32)        { fftScalar(a) }
func inverseFft(a *[128]float32) { inverseFftScalar(a) }
