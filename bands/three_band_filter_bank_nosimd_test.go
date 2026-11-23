//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package bands

import (
	"math"
	"math/rand"
	"testing"
)

// The pure-Go loops have a form with one output sample per step and one
// with two; a build runs one of them, which TestThreeBandFilterBankKernel
// and TestThreeBandFilterBankSignedZeros check against the reference. Here
// the two forms run on the same histories, with zeros of both signs and
// values that cancel, and must agree bit for bit.
func TestThreeBandFilterBankStepForms(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	negZero := float32(math.Copysign(0, -1))
	fill := func(h *[bsHistSize]float32, trial int) {
		for k := range h {
			switch {
			case trial%4 == 1 && rng.Intn(3) == 0:
				h[k] = 0
			case trial%4 == 2 && rng.Intn(3) == 0:
				h[k] = negZero
			case trial%4 == 3:
				h[k] = [3]float32{0, negZero, 1}[rng.Intn(3)]
			default:
				h[k] = float32(rng.NormFloat64() * 3000)
			}
		}
	}
	for trial := 0; trial < 400; trial++ {
		var ha [bsSubSampling][bsHistSize]float32
		for i := range ha {
			fill(&ha[i], trial)
		}
		var a1, a2 [NumBands][SplitBandSize]float32
		bsAnalysis1Go(&ha, &a1[0], &a1[1], &a1[2])
		bsAnalysis2Go(&ha, &a2[0], &a2[1], &a2[2])
		for band := range a1 {
			for k := range a1[band] {
				if math.Float32bits(a1[band][k]) != math.Float32bits(a2[band][k]) {
					t.Fatalf("analysis trial %d band %d sample %d: %v (one per step) and %v (two per step)",
						trial, band, k, a1[band][k], a2[band][k])
				}
			}
		}

		var hs [bsNumSynthesisInputs][bsHistSize]float32
		for i := range hs {
			fill(&hs[i], trial)
		}
		var s1, s2 [FullBandSize]float32
		bsSynthesisFilter1Go(&hs, &s1)
		bsSynthesisFilter2Go(&hs, &s2)
		for k := range s1 {
			if math.Float32bits(s1[k]) != math.Float32bits(s2[k]) {
				t.Fatalf("synthesis trial %d sample %d: %v (one per step) and %v (two per step)", trial, k, s1[k], s2[k])
			}
		}
	}
}
