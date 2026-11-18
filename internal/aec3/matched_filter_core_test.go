package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// dlyCoreLaneRef is the per-sample MatchedFilterCore loop of one SIMD lane
// written directly from the C++: dot product over the two circular-buffer
// chunks, reduction, then a separate update pass. It uses only the pure-Go
// lane references.
func dlyCoreLaneRef(opt Aec3Optimization, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	g := map[Aec3Optimization]int{OptNeon: 4, OptSse2: 8, OptAvx2: 16}[opt]
	hSize := len(h)
	xSize := len(x)
	for i := range y {
		var a8 [8]float32
		var a16 [16]float32
		var a32 [32]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		hp, xp := 0, xStartIndex
		for _, limit := range [2]int{chunk1, chunk2} {
			v := limit &^ (g - 1)
			switch opt {
			case OptNeon:
				dlyDotNeonRef(h[hp:hp+v], x[xp:xp+v], &a8)
			case OptSse2:
				dlyDotSse2Ref(h[hp:hp+v], x[xp:xp+v], &a16)
			default:
				dlyDotAvx2Ref(h[hp:hp+v], x[xp:xp+v], &a32)
			}
			hp += v
			xp += v
			for k := limit - v; k > 0; k-- {
				xk := x[xp]
				x2Sum += float32(xk * xk)
				s += float32(h[hp] * xk)
				hp++
				xp++
			}
			xp = 0
		}
		switch opt {
		case OptNeon:
			s, x2Sum = dlyReduceNeon(&a8, s, x2Sum)
		case OptSse2:
			s, x2Sum = dlyReduceSse2(&a16, s, x2Sum)
		default:
			s, x2Sum = dlyReduceAvx2(&a32, s, x2Sum)
		}
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			hp, xp := 0, xStartIndex
			for _, limit := range [2]int{chunk1, chunk2} {
				v := 0
				if opt == OptAvx2 {
					v = limit &^ 7
					dlyAxpyFmaRef(h[hp:hp+v], x[xp:xp+v], alpha)
				}
				dlyAxpyRef(h[hp+v:hp+limit], x[xp+v:xp+limit], alpha)
				hp += limit
				xp = 0
			}
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// TestDlyMatchedFilterCoreLanesMatchReference drives the lane cores (with
// the update deferred into the next dot product) against the direct lane
// references over filter sizes of every residue, start indices sweeping the
// circular-buffer wrap, and both updating and non-updating samples.
func TestDlyMatchedFilterCoreLanesMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 3000; trial++ {
		hSize := 1 + rng.Intn(80)
		if trial%4 == 0 {
			hSize = 4 * (1 + rng.Intn(140))
		}
		xSize := hSize + 1 + rng.Intn(3*hSize+40)
		x := dlyRandSlice(rng, xSize, 3000)
		y := dlyRandSlice(rng, 1+rng.Intn(20), 3000)
		if rng.Intn(4) == 0 {
			y[rng.Intn(len(y))] = 32000
		}
		h0 := dlyRandSlice(rng, hSize, 0.01)
		// Start indices near the wrap exercise chunk2 > 0 and its shrinking.
		xs := rng.Intn(xSize)
		if trial%2 == 0 {
			xs = max(0, xSize-hSize+rng.Intn(hSize)-len(y)/2)
			xs = min(xs, xSize-1)
		}
		var thr float32 = 1
		if trial%5 == 0 {
			// Mixed updating / non-updating samples.
			thr = float32(float64(hSize) * 3000 * 3000 * (0.7 + 0.6*rng.Float64()))
		}
		for _, opt := range []Aec3Optimization{OptNeon, OptSse2, OptAvx2} {
			hA := append([]float32(nil), h0...)
			hB := append([]float32(nil), h0...)
			var updA, updB bool
			esA, esB := float32(rng.NormFloat64()), float32(0)
			esB = esA
			mf := &MatchedFilter{optimization: opt}
			mf.runCore(xs, thr, 0.7, x, y, hA, &updA, &esA, false)
			dlyCoreLaneRef(opt, xs, thr, 0.7, x, y, hB, &updB, &esB)
			dlyRequireBits(t, "h", hA, hB)
			if updA != updB || math.Float32bits(esA) != math.Float32bits(esB) {
				t.Fatalf("opt %d trial %d: updated %v/%v errorSum %v/%v", opt, trial, updA, updB, esA, esB)
			}
		}
	}
}

// TestDlyMaxSquarePeakIndexMatchesReference checks MaxSquarePeakIndex
// against the scalar reference on random data with ties, zeros, infinities
// and NaNs.
func TestDlyMaxSquarePeakIndexMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	special := []float32{0, float32(math.Copysign(0, -1)), 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), 1e-30, 1e30, 2, -2}
	for trial := 0; trial < 20000; trial++ {
		n := rng.Intn(70)
		if trial%10 == 0 {
			n = rng.Intn(1100)
		}
		h := make([]float32, n)
		mode := rng.Intn(4)
		for i := range h {
			switch mode {
			case 0:
				h[i] = float32(rng.NormFloat64())
			case 1:
				h[i] = float32(rng.Intn(5) - 2)
			case 2:
				h[i] = special[rng.Intn(len(special))]
			default:
				h[i] = float32(rng.NormFloat64())
				if rng.Intn(8) == 0 {
					h[i] = special[rng.Intn(len(special))]
				}
			}
		}
		if got, want := MaxSquarePeakIndex(h), dlyMaxSquarePeakIndexRef(h); got != want {
			t.Fatalf("trial %d n=%d: got %d want %d (%v)", trial, n, got, want, h)
		}
	}
}

func BenchmarkMatchedFilterCoreUpdating(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	x := dlyRandSlice(rng, GetDownSampledBufferSize(4, 5), 3000)
	y := dlyRandSlice(rng, 16, 3000)
	h0 := dlyRandSlice(rng, 512, 0.01)
	for _, tc := range []struct {
		name     string
		opt      Aec3Optimization
		startIdx int
	}{
		{"None", OptNone, 100}, {"Neon", OptNeon, 100}, {"Sse2", OptSse2, 100}, {"Avx2", OptAvx2, 100},
		{"NeonWrap", OptNeon, len(x) - 300},
	} {
		b.Run(tc.name, func(b *testing.B) {
			mf := NewMatchedFilter(tc.opt, 16, 32, 5, 24, 150, 0.7, 0.7, 0.2, true)
			h := make([]float32, len(h0))
			var updated bool
			var errSum float32
			for i := 0; i < b.N; i++ {
				copy(h, h0)
				mf.runCore(tc.startIdx, 1, 0.7, x, y, h, &updated, &errSum, false)
			}
		})
	}
}

func BenchmarkMaxSquarePeakIndex(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	h := dlyRandSlice(rng, 512, 1)
	b.Run("kernel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MaxSquarePeakIndex(h)
		}
	})
	b.Run("ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			dlyMaxSquarePeakIndexRef(h)
		}
	})
}
