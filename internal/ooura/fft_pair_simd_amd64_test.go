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
// the 128-bit split kernel and of the SSE2 reference with the FftData copy,
// whatever the CPU reports; with NaNs and infinities in either frame it has
// a NaN wherever the single transform has one and the same bits elsewhere.
func TestPairMatchesSingle(t *testing.T) {
	comparePairs(t, "simd", testInputs(), false, fftSplitPairSimd, inverseFftSplitPairSimd, fftSplitSimd, inverseFftSplitSimd)
	comparePairs(t, "sse2", testInputs(), false, fftSplitPairSimd, inverseFftSplitPairSimd,
		func(a *[128]float32, re, im *[65]float32) { fftSSE2Ref(a); unpack(a, re, im) },
		func(re, im *[65]float32, a *[128]float32) { pack(re, im, a); inverseFftSSE2Ref(a) })
	comparePairs(t, "nonfinite", nonFiniteInputs(), true, fftSplitPairSimd, inverseFftSplitPairSimd, fftSplitSimd, inverseFftSplitSimd)
}

func BenchmarkFftSplitPairSimd(b *testing.B) {
	b.Run("pair", func(b *testing.B) { benchSplitPair(b, fftSplitPairSimd, inverseFftSplitPairSimd) })
	b.Run("two", func(b *testing.B) { benchSplitPair(b, twoSplits(fftSplitSimd), twoInverseSplits(inverseFftSplitSimd)) })
}
