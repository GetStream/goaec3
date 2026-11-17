//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import "testing"

// TestSimdMatchesRef checks the archsimd kernel against both lane
// references (they are identical; see TestSSE2RefMatchesNeonRef).
func TestSimdMatchesRef(t *testing.T) {
	compareFfts(t, "fwd/neon", fftSimd, fftNeonRef)
	compareFfts(t, "inv/neon", inverseFftSimd, inverseFftNeonRef)
	compareFfts(t, "fwd/sse2", fftSimd, fftSSE2Ref)
	compareFfts(t, "inv/sse2", inverseFftSimd, inverseFftSSE2Ref)
}

func BenchmarkFftSimd(b *testing.B) {
	b.Run("fwd", func(b *testing.B) { benchFft(b, fftSimd) })
	b.Run("inv", func(b *testing.B) { benchFft(b, inverseFftSimd) })
}
