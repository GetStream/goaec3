//go:build goexperiment.simd && (arm64 || amd64) && !purego

package bands

import (
	"math"
	"math/rand"
	"testing"
)

// TestFilterBankFiniteChecks puts one infinity or NaN at every position of
// the inputs and of the filter state in turn: the checks that route a frame
// to the reference must see each of them, and must pass finite frames.
func TestFilterBankFiniteChecks(t *testing.T) {
	if !bsHasSIMD {
		t.Skip("no archsimd kernels on this CPU")
	}
	rng := rand.New(rand.NewSource(5))
	bad := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), math.Float32frombits(0xffc00123)}
	fill := func(v []float32) {
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 30000
		}
		v[rng.Intn(len(v))] = math.MaxFloat32
		v[rng.Intn(len(v))] = -math.MaxFloat32
		v[rng.Intn(len(v))] = float32(math.Copysign(0, -1))
	}

	var ah [bsSubSampling][bsHistSize]float32
	var in [FullBandSize]float32
	for d := range ah {
		fill(ah[d][:])
	}
	fill(in[:])
	if !bsAnalysisFinite(&ah, &in) {
		t.Fatal("analysis: finite frame reported as not finite")
	}
	for i := range in {
		old := in[i]
		in[i] = bad[i%len(bad)]
		if bsAnalysisFinite(&ah, &in) {
			t.Fatalf("analysis: %v at input %d not seen", in[i], i)
		}
		in[i] = old
	}
	for d := range ah {
		for i := 0; i < bsMemorySize; i++ {
			old := ah[d][i]
			ah[d][i] = bad[(d+i)%len(bad)]
			if bsAnalysisFinite(&ah, &in) {
				t.Fatalf("analysis: %v in state %d at %d not seen", ah[d][i], d, i)
			}
			ah[d][i] = old
		}
	}

	var sh [bsNumSynthesisInputs][bsHistSize]float32
	var b [NumBands][SplitBandSize]float32
	for i := range sh {
		fill(sh[i][:])
	}
	for i := range b {
		fill(b[i][:])
	}
	if !bsSynthesisFinite(&sh, &b[0], &b[1], &b[2]) {
		t.Fatal("synthesis: finite frame reported as not finite")
	}
	for band := range b {
		for i := range b[band] {
			old := b[band][i]
			b[band][i] = bad[(band+i)%len(bad)]
			if bsSynthesisFinite(&sh, &b[0], &b[1], &b[2]) {
				t.Fatalf("synthesis: %v in band %d at %d not seen", b[band][i], band, i)
			}
			b[band][i] = old
		}
	}
	for fi := range sh {
		for i := 0; i < bsMemorySize; i++ {
			old := sh[fi][i]
			sh[fi][i] = bad[(fi+i)%len(bad)]
			if bsSynthesisFinite(&sh, &b[0], &b[1], &b[2]) {
				t.Fatalf("synthesis: %v in state %d at %d not seen", sh[fi][i], fi, i)
			}
			sh[fi][i] = old
		}
	}
}
