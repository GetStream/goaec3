package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// dlyLiteralReduce is the reduction of a lane's vector accumulators, added
// to the scalar tail sum, in the order of the C++: NEON adds lanes pairwise;
// SSE2 adds its two accumulators and then the four lanes in sequence; AVX2
// adds its two accumulators and reduces pairwise (hsum_ab), or, in the
// accumulated-error core, adds the two halves and then the four lanes in
// sequence.
func dlyLiteralReduce(g int, accErrX2 bool, lanes *[16]float32, tail float32) float32 {
	switch {
	case g == 4:
		tail += (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
	case g == 8:
		var v [4]float32
		for l := range v {
			v[l] = lanes[l] + lanes[4+l]
		}
		tail += ((v[0] + v[1]) + v[2]) + v[3]
	case accErrX2:
		var a [8]float32
		for l := range a {
			a[l] = lanes[l] + lanes[8+l]
		}
		var v [4]float32
		for l := range v {
			v[l] = a[l] + a[4+l]
		}
		tail += ((v[0] + v[1]) + v[2]) + v[3]
	default:
		var a [8]float32
		for l := range a {
			a[l] = lanes[l] + lanes[8+l]
		}
		tail += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
	}
	return tail
}

// dlyLiteralCore is the per-sample loop of MatchedFilterCore_NEON (g = 4),
// _SSE2 (8) and _AVX2 (16) written directly from the C++, one scalar
// operation per lane: h*x and x*x accumulated side by side over the two
// circular-buffer chunks, reduction, then a separate update pass. It shares
// no code with the lane cores under test.
func dlyLiteralCore(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	hSize := len(h)
	xSize := len(x)
	for i := range y {
		var sl, x2l [16]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		hp, xp := 0, xStartIndex
		for _, limit := range [2]int{chunk1, chunk2} {
			for grp := 0; grp < limit/g; grp++ {
				for l := 0; l < g; l++ {
					xk, hk := x[xp+l], h[hp+l]
					if g == 16 {
						x2l[l] = fmaf(xk, xk, x2l[l])
						sl[l] = fmaf(hk, xk, sl[l])
					} else {
						x2l[l] += float32(xk * xk)
						sl[l] += float32(hk * xk)
					}
				}
				hp += g
				xp += g
			}
			for k := limit % g; k > 0; k-- {
				xk := x[xp]
				x2Sum += float32(xk * xk)
				s += float32(h[hp] * xk)
				hp++
				xp++
			}
			xp = 0
		}
		x2Sum = dlyLiteralReduce(g, false, &x2l, x2Sum)
		s = dlyLiteralReduce(g, false, &sl, s)
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			hp, xp := 0, xStartIndex
			for _, limit := range [2]int{chunk1, chunk2} {
				for k := 0; k < limit; k++ {
					// The AVX2 update is fused over whole groups of 8 of
					// each chunk and unfused in the chunk's tail.
					if g == 16 && k < limit&^7 {
						h[hp] = fmaf(x[xp], alpha, h[hp])
					} else {
						h[hp] += float32(alpha * x[xp])
					}
					hp++
					xp++
				}
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

// dlyLiteralAccErrCore is the per-sample loop of the accumulated-error
// cores (MatchedFilterCoreWithAccumulatedError_NEON,
// MatchedFilterCore_AccumulatedError_SSE2 and _AVX2) written directly from
// the C++ for filter lengths that are a multiple of g: the window copied
// out of the circular buffer, x*x in lanes, the filter output as a running
// scalar sum over sub-groups of four with the squared error accumulated
// after each, then a separate update pass.
func dlyLiteralAccErrCore(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, accumulatedError []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	win := make([]float32, hSize)
	for i := range y {
		for k := range win {
			win[k] = x[(xStartIndex+k)%xSize]
		}
		var x2l [16]float32
		var s float32
		for grp := 0; grp < hSize/g; grp++ {
			for l := 0; l < g; l++ {
				xk := win[grp*g+l]
				if g == 16 {
					x2l[l] = fmaf(xk, xk, x2l[l])
				} else {
					x2l[l] += float32(xk * xk)
				}
			}
			var e [4]float32
			for sub := 0; sub < g/4; sub++ {
				o := grp*g + 4*sub
				p0, p1 := float32(h[o]*win[o]), float32(h[o+1]*win[o+1])
				p2, p3 := float32(h[o+2]*win[o+2]), float32(h[o+3]*win[o+3])
				if g == 8 {
					// SSE2 adds the four products in sequence.
					s += ((p0 + p1) + p2) + p3
				} else {
					s += (p0 + p1) + (p2 + p3)
				}
				e[sub] = s - y[i]
			}
			for sub := 0; sub < g/4; sub++ {
				a := &accumulatedError[grp*g/4+sub]
				if g == 16 {
					*a = fmaf(e[sub], e[sub], *a)
				} else {
					*a += float32(e[sub] * e[sub])
				}
			}
		}
		x2Sum := dlyLiteralReduce(g, true, &x2l, 0)
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			for k := range h {
				if g == 16 {
					h[k] = fmaf(win[k], alpha, h[k])
				} else {
					h[k] += float32(alpha * win[k])
				}
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

func dlyLaneGroup(opt Aec3Optimization) int {
	return map[Aec3Optimization]int{OptNeon: 4, OptSse2: 8, OptAvx2: 16}[opt]
}

// dlyCoreTrial is one random core call: filter and buffer sizes of every
// residue, start indices sweeping the circular-buffer wrap and the run-down
// to index zero, updating and non-updating samples, saturated captures.
type dlyCoreTrial struct {
	x, y, h []float32
	xs      int
	thr     float32
}

func dlyRandCoreTrial(rng *rand.Rand, trial, multiple int) dlyCoreTrial {
	hSize := multiple * (1 + rng.Intn(80/multiple+1))
	if trial%4 == 0 {
		hSize = 16 * (1 + rng.Intn(40))
	}
	if trial%16 == 1 {
		hSize = 512
	}
	xSize := hSize + 1 + rng.Intn(3*hSize+40)
	var c dlyCoreTrial
	c.x = dlyRandSlice(rng, xSize, 3000)
	c.y = dlyRandSlice(rng, 1+rng.Intn(20), 3000)
	if trial%8 < 3 {
		c.y = dlyRandSlice(rng, 16, 3000)
	}
	if rng.Intn(4) == 0 {
		c.y[rng.Intn(len(c.y))] = 32000
	}
	c.h = dlyRandSlice(rng, hSize, 0.01)
	switch trial % 3 {
	case 0:
		// Near the wrap: chunk2 > 0 and shrinking.
		c.xs = min(max(0, xSize-hSize+rng.Intn(hSize)-len(c.y)/2), xSize-1)
	case 1:
		// Running down through index zero into the wrap.
		c.xs = rng.Intn(len(c.y) + 2)
		if c.xs >= xSize {
			c.xs = xSize - 1
		}
	default:
		c.xs = rng.Intn(xSize)
	}
	c.thr = 1
	if trial%5 == 0 {
		// Mixed updating / non-updating samples.
		c.thr = float32(float64(hSize) * 3000 * 3000 * (0.7 + 0.6*rng.Float64()))
	}
	if trial%11 == 0 {
		// Exact zeros and small integers put the software fma on its
		// boundary tests.
		for k := range c.x {
			c.x[k] = float32(rng.Intn(7) - 3)
		}
		for k := range c.h {
			c.h[k] = float32(rng.Intn(5)-2) / 4
		}
		c.thr = 0.5
	}
	if trial%11 == 5 {
		// Sums of few bits below 2^-126 send the software fma to its
		// per-element path.
		for k := range c.x {
			c.x[k] = float32(rng.Intn(9)-4) * 0x1p-70
		}
		for k := range c.h {
			c.h[k] = float32(rng.Intn(5)-2) * 0x1p-80
		}
		for k := range c.y {
			c.y[k] = float32(rng.Intn(9)-4) * 0x1p-60
		}
		c.thr = 0
	}
	return c
}

// TestDlyMatchedFilterCoreLanesMatchReference drives the lane cores (x*x
// from the shared lane chains, the update deferred into the next dot
// product) against the literal cores, one filter at a time and two filters
// in step, with and without the two-filter kernels.
func TestDlyMatchedFilterCoreLanesMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	saved := [3]bool{dlyPairNeon, dlyPairSse2, dlyPairAvx2}
	defer func() { dlyPairNeon, dlyPairSse2, dlyPairAvx2 = saved[0], saved[1], saved[2] }()
	for trial := 0; trial < 4000; trial++ {
		c := dlyRandCoreTrial(rng, trial, 1)
		// A second filter on the same buffer, as Update pairs them.
		hB0 := dlyRandSlice(rng, len(c.h), 0.01)
		xsB := rng.Intn(len(c.x))
		if trial%2 == 0 {
			xsB = (c.xs + 3*len(c.h)/4) % len(c.x)
		}
		for _, opt := range []Aec3Optimization{OptNeon, OptSse2, OptAvx2} {
			g := dlyLaneGroup(opt)
			es0 := float32(rng.NormFloat64())
			hWant := append([]float32(nil), c.h...)
			var updWant bool
			esWant := es0
			dlyLiteralCore(g, c.xs, c.thr, 0.7, c.x, c.y, hWant, &updWant, &esWant)
			hWantB := append([]float32(nil), hB0...)
			var updWantB bool
			var esWantB float32
			dlyLiteralCore(g, xsB, c.thr, 0.7, c.x, c.y, hWantB, &updWantB, &esWantB)

			hA := append([]float32(nil), c.h...)
			var updA bool
			esA := es0
			mf := &MatchedFilter{optimization: opt}
			mf.runCore(c.xs, c.thr, 0.7, c.x, c.y, hA, &updA, &esA, false)
			dlyRequireBits(t, "h", hA, hWant)
			if updA != updWant || math.Float32bits(esA) != math.Float32bits(esWant) {
				t.Fatalf("opt %d trial %d: updated %v/%v errorSum %v/%v", opt, trial, updA, updWant, esA, esWant)
			}

			for _, pair := range []bool{false, true} {
				dlyPairNeon, dlyPairSse2, dlyPairAvx2 = pair, pair, pair
				fa := dlyLaneFilter{h: append([]float32(nil), c.h...), xStart: c.xs, errorSum: es0}
				fb := dlyLaneFilter{h: append([]float32(nil), hB0...), xStart: xsB}
				dlyCoreLanes(g, &fa, &fb, c.thr, 0.7, c.x, c.y)
				dlyRequireBits(t, "paired h A", fa.h, hWant)
				dlyRequireBits(t, "paired h B", fb.h, hWantB)
				if fa.updated != updWant || math.Float32bits(fa.errorSum) != math.Float32bits(esWant) ||
					fb.updated != updWantB || math.Float32bits(fb.errorSum) != math.Float32bits(esWantB) {
					t.Fatalf("opt %d trial %d pair %v: updated %v/%v %v/%v errorSum %v/%v %v/%v", opt, trial, pair,
						fa.updated, updWant, fb.updated, updWantB, fa.errorSum, esWant, fb.errorSum, esWantB)
				}
			}
			dlyPairNeon, dlyPairSse2, dlyPairAvx2 = saved[0], saved[1], saved[2]
		}
	}
}

// TestDlyMatchedFilterAccErrCoresMatchReference drives the accumulated-error
// cores (windows of the whole call on one contiguous stretch, x*x from the
// shared lane chains, the update fused into the next filter loop) against
// the literal cores, with a scratch for the whole call and with one for a
// single window, which takes the per-sample fallback when the stretch of
// the call wraps.
func TestDlyMatchedFilterAccErrCoresMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for trial := 0; trial < 3000; trial++ {
		for _, opt := range []Aec3Optimization{OptNeon, OptSse2, OptAvx2} {
			g := dlyLaneGroup(opt)
			c := dlyRandCoreTrial(rng, trial, g)
			hSize := len(c.h)
			es0 := float32(rng.NormFloat64())
			hWant := append([]float32(nil), c.h...)
			accWant := dlyRandSlice(rng, hSize/4, 10)
			var updWant bool
			esWant := es0
			dlyLiteralAccErrCore(g, c.xs, c.thr, 0.7, c.x, c.y, hWant, &updWant, &esWant, accWant)

			// A scratch for the whole call, and one for a single window,
			// which takes the fallback when the call's stretch wraps.
			for _, scratchSize := range []int{hSize + len(c.y), hSize} {
				hA := append([]float32(nil), c.h...)
				accA := dlyRandSlice(rng, hSize/4, 10)
				scratch := dlyRandSlice(rng, scratchSize, 1)
				var updA bool
				esA := es0
				mf := &MatchedFilter{optimization: opt, instantaneousAccumulatedError: accA, scratchMemory: scratch}
				mf.runCore(c.xs, c.thr, 0.7, c.x, c.y, hA, &updA, &esA, true)
				dlyRequireBits(t, "h", hA, hWant)
				dlyRequireBits(t, "accumulated error", accA, accWant)
				if updA != updWant || math.Float32bits(esA) != math.Float32bits(esWant) {
					t.Fatalf("opt %d trial %d scratch %d: updated %v/%v errorSum %v/%v", opt, trial, scratchSize,
						updA, updWant, esA, esWant)
				}
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
		{"NeonWrap", OptNeon, len(x) - 300}, {"Sse2Wrap", OptSse2, len(x) - 300}, {"Avx2Wrap", OptAvx2, len(x) - 300},
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
	// Two updating filters per iteration through dlyCoreLanes, as Update
	// runs them: in step with the two-filter kernels (Pair) and in step
	// with the one-filter kernels (Two).
	for _, tc := range []struct {
		name     string
		g        int
		startIdx int
		pair     bool
	}{
		{"NeonTwo", 4, 100, false}, {"NeonPair", 4, 100, true},
		{"NeonWrapTwo", 4, len(x) - 300, false}, {"NeonWrapPair", 4, len(x) - 300, true},
		{"Sse2Two", 8, 100, false}, {"Sse2Pair", 8, 100, true},
		{"Avx2Two", 16, 100, false}, {"Avx2Pair", 16, 100, true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			saved := [3]bool{dlyPairNeon, dlyPairSse2, dlyPairAvx2}
			defer func() { dlyPairNeon, dlyPairSse2, dlyPairAvx2 = saved[0], saved[1], saved[2] }()
			dlyPairNeon, dlyPairSse2, dlyPairAvx2 = tc.pair, tc.pair, tc.pair
			hA := make([]float32, len(h0))
			hB := make([]float32, len(h0))
			for i := 0; i < b.N; i++ {
				copy(hA, h0)
				copy(hB, h0)
				fa := dlyLaneFilter{h: hA, xStart: tc.startIdx}
				fb := dlyLaneFilter{h: hB, xStart: (tc.startIdx + 384) % len(x)}
				dlyCoreLanes(tc.g, &fa, &fb, 1, 0.7, x, y)
			}
		})
	}
	// The x*x sums of the 16 samples of one sub-block for one filter: from
	// the shared lane chains (windows that do not wrap) and from the chunk
	// loops (windows that wrap).
	for _, g := range []int{4, 8, 16} {
		b.Run(fmt.Sprintf("X2Lanes%d", g), func(b *testing.B) {
			var out [dlyX2Run]float32
			for i := 0; i < b.N; i++ {
				dlyX2Sums(g, false, x[100:], 512, 16, &out)
			}
		})
		b.Run(fmt.Sprintf("X2Wrapped%d", g), func(b *testing.B) {
			var out [dlyX2Run]float32
			for i := 0; i < b.N; i++ {
				dlyX2Wrapped(g, x, 512, len(x)-300, 16, &out)
			}
		})
	}
	dlyBenchKernels(b, x, h0)
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
