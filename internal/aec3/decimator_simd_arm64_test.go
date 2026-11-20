//go:build goexperiment.simd && arm64 && !purego

package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// TestDecimateDs4Kernel runs dlyDecimateDs4 against the sections applied one
// after the other: the outputs and the section states are bit-identical
// after every block, and a block with a value that is not finite is refused
// with the states untouched.
func TestDecimateDs4Kernel(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	got := NewDecimator(4)
	want := NewDecimator(4)
	refused := 0
	for block := 0; block < 20000; block++ {
		var in [BlockSize]float32
		scale := []float64{1, 4000, 32768, 1e-3, 1e-30, 1e15}[rng.Intn(6)]
		for i := range in {
			in[i] = float32(rng.NormFloat64() * scale)
		}
		switch rng.Intn(400) {
		case 0:
			in[rng.Intn(BlockSize)] = float32(math.Inf(1 - 2*rng.Intn(2)))
		case 1:
			in[rng.Intn(BlockSize)] = float32(math.NaN())
		case 2:
			in[rng.Intn(BlockSize)] = math.MaxFloat32
		case 3:
			clear(in[:])
		}
		if block%50 == 49 {
			for k := range got.sections {
				got.sections[k].Reset()
				want.sections[k].Reset()
			}
		}
		var outGot, outWant [BlockSize / 4]float32
		before := got.sections
		if !dlyDecimateDs4(&in, &outGot, (*[4]BiQuad)(got.sections[:4])) {
			refused++
			if !sectionsEqualBits(got.sections[:], before[:]) {
				t.Fatalf("block %d: refused block changed the section states", block)
			}
			got.Decimate(in[:], outGot[:])
		}
		refDecimate(want, in[:], outWant[:])
		if i := equalBits(outGot[:], outWant[:]); i >= 0 {
			t.Fatalf("block %d: out[%d] = %v, want %v", block, i, outGot[i], outWant[i])
		}
		if !sectionsEqualBits(got.sections[:], want.sections[:]) {
			t.Fatalf("block %d: section states differ", block)
		}
	}
	if refused == 0 || refused > 10000 {
		t.Fatalf("%d of 20000 blocks refused", refused)
	}
}

func sectionsEqualBits(a, b []BiQuad) bool {
	for k := range a {
		if equalBits(append(a[k].X[:], a[k].Y[:]...), append(b[k].X[:], b[k].Y[:]...)) >= 0 {
			return false
		}
	}
	return true
}

func TestDecimateDs4KernelAllocationFree(t *testing.T) {
	d := NewDecimator(4)
	var in [BlockSize]float32
	var out [BlockSize / 4]float32
	if a := testing.AllocsPerRun(100, func() { dlyDecimateDs4(&in, &out, (*[4]BiQuad)(d.sections[:4])) }); a != 0 {
		t.Fatalf("dlyDecimateDs4 allocates %v times per call", a)
	}
}
