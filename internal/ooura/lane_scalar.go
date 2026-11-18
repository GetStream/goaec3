//go:build purego || !(arm64 || amd64)

package ooura

// The purego lane and architectures without a WebRTC SIMD path run the
// scalar C code of ooura_fft.cc.

func fftRef(a *[128]float32)        { fftScalar(a) }
func inverseFftRef(a *[128]float32) { inverseFftScalar(a) }

func fftFast(a *[128]float32)        { fftFastScalar(a) }
func inverseFftFast(a *[128]float32) { inverseFftFastScalar(a) }
