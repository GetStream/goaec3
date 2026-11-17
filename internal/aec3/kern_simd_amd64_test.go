//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
	"math/rand"
	"os"
	"testing"
)

// TestKernSpectrumAvx2Kernel checks the Float32x8.MulAdd kernel against
// kernSpectrumAvx2Ref. It runs when CPUID reports FMA, or when
// GOAEC3_FORCE_FMA=1: Rosetta 2 executes VFMADD but does not advertise FMA,
// so the dispatcher never selects this kernel there.
func TestKernSpectrumAvx2Kernel(t *testing.T) {
	t.Logf("AVX=%v AVX2=%v FMA(with AVX2)=%v", kernHasAVX, kernHasAVX2, kernHasFMA)
	if !kernHasFMA && os.Getenv("GOAEC3_FORCE_FMA") != "1" {
		t.Log("FMA not reported: kernSpectrumAvx2 not exercised")
		return
	}
	rng := rand.New(rand.NewSource(4))
	for trial := 0; trial < 5000; trial++ {
		var re, im, got, want [FftLengthBy2Plus1]float32
		copy(re[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(im[:], kernRandVec(rng, FftLengthBy2Plus1))
		kernSpectrumAvx2(&re, &im, &got)
		kernSpectrumAvx2Ref(&re, &im, &want)
		kernCheck(t, "SpectrumAvx2", got[:], want[:])
	}
}
