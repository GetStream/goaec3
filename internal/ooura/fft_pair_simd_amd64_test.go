//go:build goexperiment.simd && amd64 && !purego

package ooura

import (
	"math"
	"testing"
)

func poisonScratch(s *PairScratch) {
	nan := float32(math.NaN())
	for i := range s.t {
		s.t[i], s.u[i] = nan, nan
	}
}

// TestPairMatchesSingle checks the 256-bit pair kernel against two runs of
// the 128-bit kernel and of the SSE2 reference, whatever the CPU reports.
func TestPairMatchesSingle(t *testing.T) {
	comparePairs(t, "fwd/simd", fftPairSimd, fftSimd)
	comparePairs(t, "inv/simd", inverseFftPairSimd, inverseFftSimd)
	comparePairs(t, "fwd/sse2", fftPairSimd, fftSSE2Ref)
	comparePairs(t, "inv/sse2", inverseFftPairSimd, inverseFftSSE2Ref)
}

func BenchmarkFftPairSimd(b *testing.B) {
	b.Run("fwd/pair", func(b *testing.B) { benchFftPair(b, fftPairSimd) })
	b.Run("fwd/two", func(b *testing.B) { benchFftPair(b, twice(fftSimd)) })
	b.Run("inv/pair", func(b *testing.B) { benchFftPair(b, inverseFftPairSimd) })
	b.Run("inv/two", func(b *testing.B) { benchFftPair(b, twice(inverseFftSimd)) })
}
