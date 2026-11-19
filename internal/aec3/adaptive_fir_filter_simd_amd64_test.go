//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
	"math/rand"
	"os"
	"testing"
)

// Direct archsimd kernel vs pure-Go lane reference checks.
func TestAfirSimdKernelsAmd64(t *testing.T) {
	t.Logf("AVX=%v AVX2=%v AVX2+FMA=%v", afirHasAVX, afirHasAVX2, afirHasAVX2FMA)
	if !afirHasAVX {
		t.Skip("no AVX: SSE2 lane runs its pure-Go reference")
	}
	rng := rand.New(rand.NewSource(7))
	im := float32(4097.0 / 4096.0)
	for trial := 0; trial < 2000; trial++ {
		var x, h, s FftData
		afirTestRandFft(rng, &x, 100)
		afirTestRandFft(rng, &h, 0.1)
		afirTestRandFft(rng, &s, 10)
		if trial%5 == 0 {
			for j := range h.Im {
				h.Im[j] = im
				h.Re[j] = float32(1e-17 * float64(j+1))
			}
		}
		type kern struct {
			name         string
			apply, adapt func(a, b, c *FftData)
			fr           func(*FftData, *[FftLengthBy2Plus1]float32)
			frRef        func(*FftData, *[FftLengthBy2Plus1]float32)
			erl          func(a, b *[FftLengthBy2Plus1]float32)
			needAVX2     bool
		}
		ks := []kern{
			{"sse2", afirApplyBlockSSE2, afirAdaptBlockSSE2, afirFreqRespBlockSSE2,
				afirFreqRespBlockSse2Go, afirErlBlockSSE2, false},
			{"avx2", afirApplyBlockAVX2, afirAdaptBlockAVX2, afirFreqRespBlockAVX2, afirFreqRespBlockAvx2Go, afirErlBlockAVX2, true},
		}
		for _, k := range ks {
			if k.needAVX2 && !afirHasAVX2 {
				continue
			}
			g, w := s, s
			k.apply(&x, &h, &g)
			afirApplyBinsGo(&x, &h, &w, 0, FftLengthBy2)
			afirEqualFft(t, k.name+" apply", &g, &w)
			g, w = s, s
			k.adapt(&x, &h, &g)
			afirAdaptBinsGo(&x, &h, &w, 0, FftLengthBy2)
			afirEqualFft(t, k.name+" adapt", &g, &w)
			g2, w2 := s.Re, s.Re
			if k.name == "sse2" || afirHasAVX2FMA {
				k.fr(&h, &g2)
				k.frRef(&h, &w2)
				afirEqualArr(t, k.name+" fr", g2[:], w2[:])
			}
			g2, w2 = s.Re, s.Re
			k.erl(&h.Re, &g2)
			afirErlBlockGo(&h.Re, &w2)
			afirEqualArr(t, k.name+" erl", g2[:], w2[:])
		}
	}
}

// TestAfirAVXOnlyKernels runs the lane and row-sum checks with the 128-bit
// AVX paths of the term and row kernels (CPUs with AVX but not AVX2).
func TestAfirAVXOnlyKernels(t *testing.T) {
	if !afirHasAVX {
		t.Skip("no AVX")
	}
	a, k := afirHasAVX2, kernHasAVX2
	afirHasAVX2, kernHasAVX2 = false, false
	defer func() { afirHasAVX2, kernHasAVX2 = a, k }()
	TestAfirLanesMatchScalar(t)
	TestKernSumRows(t)
	TestRenderBufferSpectralSums(t)
	TestKernBinsElementwise(t)
	TestKernSpectrum(t)
}

// TestAfirFreqRespPartitionX86 checks the partition kernels against the
// per-channel lane references, the AVX2 one also when the CPU executes FMA
// without reporting it (GOAEC3_FORCE_FMA=1, Rosetta 2).
func TestAfirFreqRespPartitionX86(t *testing.T) {
	if !afirHasAVX {
		t.Skip("no AVX")
	}
	rng := rand.New(rand.NewSource(23))
	im := float32(4097.0 / 4096.0)
	for _, opt := range []Aec3Optimization{OptSse2, OptAvx2} {
		if opt == OptAvx2 && !afirHasAVX2FMA && os.Getenv("GOAEC3_FORCE_FMA") != "1" {
			t.Log("AVX2 kernel not run: the CPU does not report FMA (GOAEC3_FORCE_FMA=1 forces it)")
			continue
		}
		for trial := 0; trial < 2000; trial++ {
			hs := make([]FftData, 1+rng.Intn(3))
			for i := range hs {
				if trial%2 == 0 {
					mmEdgeFft(rng, &hs[i])
				} else {
					afirTestRandFft(rng, &hs[i], 0.1)
				}
				if trial%5 == 1 {
					for j := range hs[i].Im {
						hs[i].Im[j] = im
						hs[i].Re[j] = float32(1e-17 * float64(j+1))
					}
				}
			}
			var got, want [FftLengthBy2Plus1]float32
			copy(got[:], kernEdgeVec(rng, FftLengthBy2Plus1))
			afirFreqRespPartitionX86(opt, hs, &got)
			for i := range hs {
				if opt == OptSse2 {
					afirFreqRespBlockSse2Go(&hs[i], &want)
				} else {
					afirFreqRespBlockAvx2Go(&hs[i], &want)
				}
				afirFreqRespBinsGo(&hs[i], &want, FftLengthBy2, FftLengthBy2Plus1)
			}
			kernCheckExact(t, "freqresp partition", got[:], want[:])
		}
	}
}
