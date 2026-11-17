//go:build goexperiment.simd && arm64 && !purego

package aec3

import (
	"math/rand"
	"testing"
)

// Direct archsimd NEON kernel vs pure-Go lane reference checks.
func TestAfirSimdKernelsArm64(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for trial := 0; trial < 2000; trial++ {
		var x, h, s FftData
		afirTestRandFft(rng, &x, 100)
		afirTestRandFft(rng, &h, 0.1)
		afirTestRandFft(rng, &s, 10)
		g, w := s, s
		afirApplyBlockNEON(&x, &h, &g)
		afirApplyBinsGo(&x, &h, &w, 0, FftLengthBy2)
		afirEqualFft(t, "apply", &g, &w)
		g, w = s, s
		afirAdaptBlockNEON(&x, &h, &g)
		afirAdaptBinsGo(&x, &h, &w, 0, FftLengthBy2)
		afirEqualFft(t, "adapt", &g, &w)
		g2, w2 := s.Re, s.Re
		afirFreqRespBlockNEON(&h, &g2)
		afirFreqRespBinsGo(&h, &w2, 0, FftLengthBy2)
		afirEqualArr(t, "fr", g2[:], w2[:])
		g2, w2 = s.Re, s.Re
		afirErlBlockNEON(&h.Re, &g2)
		afirErlBlockGo(&h.Re, &w2)
		afirEqualArr(t, "erl", g2[:], w2[:])
	}
}
