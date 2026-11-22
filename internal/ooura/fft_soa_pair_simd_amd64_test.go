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
	for i := range s.s.t {
		s.s.t[i] = nan
	}
}

// TestSoaPairMatchesSingle checks the 256-bit pair kernel against two runs
// of the 128-bit split kernel and of the SSE2 reference with the FftData
// copy, on a CPU with AVX2; with NaNs and infinities in either frame it has
// a NaN wherever the single transform has one and the same bits elsewhere.
func TestSoaPairMatchesSingle(t *testing.T) {
	if !usePair {
		t.Skip("no AVX2")
	}
	comparePairs(t, "soa", testInputs(), false, fftSplitPairSoa, inverseFftSplitPairSoa, soaFftSplit, soaInverseFftSplit)
	comparePairs(t, "sse2", testInputs(), false, fftSplitPairSoa, inverseFftSplitPairSoa,
		func(a *[128]float32, re, im *[65]float32) { fftSSE2Ref(a); unpack(a, re, im) },
		func(re, im *[65]float32, a *[128]float32) { pack(re, im, a); inverseFftSSE2Ref(a) })
	comparePairs(t, "nonfinite", nonFiniteInputs(), true, fftSplitPairSoa, inverseFftSplitPairSoa, soaFftSplit, soaInverseFftSplit)
}

func BenchmarkFftSplitPairSimd(b *testing.B) {
	if !usePair {
		b.Skip("no AVX2")
	}
	b.Run("soa", func(b *testing.B) { benchSplitPair(b, fftSplitPairSoa, inverseFftSplitPairSoa) })
	b.Run("two", func(b *testing.B) { benchSplitPair(b, twoSplits(soaFftSplit), twoInverseSplits(soaInverseFftSplit)) })
}
