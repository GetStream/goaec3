//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
	"math/rand"
	"os"
	"testing"
)

// TestExpectAVXOnly fails under GOAEC3_EXPECT_AVX_ONLY=1 unless the CPU has
// AVX and no AVX2, so that the avx-only job of ci.yml cannot pass on another
// CPU.
func TestExpectAVXOnly(t *testing.T) {
	if os.Getenv("GOAEC3_EXPECT_AVX_ONLY") != "1" {
		t.Skip("GOAEC3_EXPECT_AVX_ONLY is not set")
	}
	if !kernHasAVX || kernHasAVX2 || kernHasFMA || DetectOptimization() != OptSse2 {
		t.Fatalf("AVX=%v AVX2=%v FMA(with AVX2)=%v lane=%v: want AVX without AVX2 and the SSE2 lane",
			kernHasAVX, kernHasAVX2, kernHasFMA, DetectOptimization())
	}
}

// TestKernSpectrumAvx2Kernel checks the fused kernel against
// kernSpectrumAvx2Ref. It runs when CPUID reports FMA, or with
// GOAEC3_FORCE_FMA=1 on a CPU with AVX2: Rosetta 2 executes FMA without
// reporting it.
func TestKernSpectrumAvx2Kernel(t *testing.T) {
	t.Logf("AVX=%v AVX2=%v FMA(with AVX2)=%v", kernHasAVX, kernHasAVX2, kernHasFMA)
	if !kernHasFMA {
		if !kernHasAVX2 || os.Getenv("GOAEC3_FORCE_FMA") != "1" {
			t.Log("FMA not reported: kernSpectrumAvx2 not exercised")
			return
		}
		kernHasFMA = true
		defer func() { kernHasFMA = false }()
	}
	rng := rand.New(rand.NewSource(4))
	for trial := 0; trial < 5000; trial++ {
		var re, im, got, want [FftLengthBy2Plus1]float32
		copy(re[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(im[:], kernRandVec(rng, FftLengthBy2Plus1))
		fftDataSpectrum(OptAvx2, &re, &im, &got)
		kernSpectrumAvx2Ref(&re, &im, &want)
		kernCheck(t, "SpectrumAvx2", got[:], want[:])
	}
}
