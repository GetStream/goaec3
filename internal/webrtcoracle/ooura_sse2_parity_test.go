//go:build ooura_sse2_oracle && amd64

package webrtcoracle

import "testing"

// TestOouraFftX86BitExact compares the Go FFT with OouraFft compiled for
// x86_64: the SSE2 path in the default build, the scalar C path under
// purego.
func TestOouraFftX86BitExact(t *testing.T) {
	oouraCheck(t, func(a *[128]float32, inverse bool) { OouraFftX86(a, inverse, !oouraX86Purego) })
}
