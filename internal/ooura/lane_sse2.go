//go:build amd64 && !purego

package ooura

// The amd64 lane is libwebrtc's SSE2 build (ooura_fft_sse2.cc), which
// OouraFft selects whenever SSE2 is available, i.e. always on amd64.

func fftRef(a *[128]float32)        { fftSSE2Ref(a) }
func inverseFftRef(a *[128]float32) { inverseFftSSE2Ref(a) }
