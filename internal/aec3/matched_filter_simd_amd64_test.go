//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
	"math"
	"math/rand"
	"os"
	"simd/archsimd"
	"testing"
)

func TestDlyAmd64KernelSelection(t *testing.T) {
	t.Logf("AVX=%v AVX2=%v FMA=%v", archsimd.X86.AVX(), archsimd.X86.AVX2(), archsimd.X86.FMA())
	t.Logf("SSE2-lane archsimd kernels: %v, AVX2-lane archsimd kernels: %v", dlyUseAvx, dlyUseFma)
	if !dlyUseFma {
		t.Log("AVX2/FMA not available: the AVX2 lane runs its pure-Go reference")
	}
}

// TestDlyAvx2KernelsForced checks the archsimd kernels of the AVX2 lane
// against their references. DLY_FORCE_FMA=1 runs it on a CPU that executes
// FMA without reporting it.
func TestDlyAvx2KernelsForced(t *testing.T) {
	if os.Getenv("DLY_FORCE_FMA") == "" {
		t.Skip("set DLY_FORCE_FMA=1 to force the AVX2-lane kernels")
	}
	saved := dlyUseFma
	dlyUseFma = true
	defer func() { dlyUseFma = saved }()
	TestDlyMatchedFilterKernelsMatchReference(t)
	TestDlyMatchedFilterCoreLanesMatchReference(t)
	TestDlyMatchedFilterAccErrCoresMatchReference(t)
	TestDlyAxpyFma128MatchesReference(t)
	TestDlyWrapSamplesMatchLiteral(t)
}

// TestDlyAxpyFma128MatchesReference checks dlyAxpyFma128 against its
// reference for every length around its vector sizes.
func TestDlyAxpyFma128MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for trial := 0; trial < 2000; trial++ {
		n := trial % 41
		x := dlyRandSlice(rng, n, 3000)
		h := dlyRandSlice(rng, n, 0.01)
		alpha := float32(rng.NormFloat64() * 1e-4)
		if trial%7 == 0 {
			for k := range x {
				x[k] = float32(rng.Intn(7) - 3)
				h[k] = float32(rng.Intn(5)-2) / 4
			}
			alpha = float32(rng.Intn(5)-2) / 8
		}
		want := append([]float32(nil), h...)
		dlyAxpyFmaRef(want, x, alpha)
		dlyAxpyFma128(h, x, alpha)
		dlyRequireBits(t, "h", h, want)
	}
}

// TestDlyWrapSamplesMatchLiteral checks the SSE2 and AVX2 lane cores against
// the literal cores from every start index of the render buffer: the first
// chunk of a sample takes every length and the pending update every position
// against it. Every sample updates in one run, some in the other.
func TestDlyWrapSamplesMatchLiteral(t *testing.T) {
	rng := rand.New(rand.NewSource(22))
	for _, hSize := range []int{512, 64, 48, 40, 32, 24, 16, 8, 21} {
		xSize := hSize + 1 + rng.Intn(40)
		x := dlyRandSlice(rng, xSize, 3000)
		h0 := dlyRandSlice(rng, hSize, 0.01)
		step := 1
		if hSize == 512 && testing.Short() {
			step = 7
		}
		for xs := 0; xs < xSize; xs += step {
			y := dlyRandSlice(rng, 1+(xs*5)%19, 3000)
			thr := float32(1)
			if xs%3 == 1 {
				thr = float32(float64(hSize) * 3000 * 3000 * (0.7 + 0.6*rng.Float64()))
			}
			for _, g := range []int{8, 16} {
				hWant := append([]float32(nil), h0...)
				var updWant bool
				esWant := float32(0.5)
				dlyLiteralCore(g, xs, thr, 0.7, x, y, hWant, &updWant, &esWant)
				h := append([]float32(nil), h0...)
				var upd bool
				es := float32(0.5)
				if g == 8 {
					MatchedFilterCoreSse2(xs, thr, 0.7, x, y, h, &upd, &es, false, nil, nil)
				} else {
					MatchedFilterCoreAvx2(xs, thr, 0.7, x, y, h, &upd, &es, false, nil, nil)
				}
				dlyRequireBits(t, "h", h, hWant)
				if upd != updWant || math.Float32bits(es) != math.Float32bits(esWant) {
					t.Fatalf("g %d hSize %d xs %d: updated %v/%v errorSum %v/%v", g, hSize, xs, upd, updWant, es, esWant)
				}
			}
		}
	}
}
