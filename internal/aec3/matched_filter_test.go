package aec3

import (
	"math"
	"math/rand"
	"testing"
)

func dlyRandSlice(rng *rand.Rand, n int, scale float64) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64() * scale)
	}
	return s
}

func dlyRequireBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (%#x), want %v (%#x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

// dlyLiteralDot accumulates a*b lane by lane over len(a)/g groups as the
// lane with group size g does, fused for g = 16. It states what the Dot (a =
// h, b = x) and the Sq (a = b = x) kernels compute.
func dlyLiteralDot(g int, a, b []float32, acc []float32) {
	for grp := 0; grp < len(a)/g; grp++ {
		for l := 0; l < g; l++ {
			k := grp*g + l
			if g == 16 {
				acc[l] = fmaf(a[k], b[k], acc[l])
			} else {
				acc[l] += float32(a[k] * b[k])
			}
		}
	}
}

func dlyRandAcc(rng *rand.Rand) [16]float32 {
	var a [16]float32
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	return a
}

// TestDlyMatchedFilterKernelsMatchReference checks the kernels of the build
// against the pure-Go lane references, and the references against the
// literal lane arithmetic.
func TestDlyMatchedFilterKernelsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 4000; trial++ {
		// Every residue modulo the unroll widths, lengths below one unrolled
		// iteration, and an x longer than h.
		n := rng.Intn(80)
		if trial%3 == 0 {
			n = rng.Intn(700)
		}
		h := dlyRandSlice(rng, n, 0.3)
		x := dlyRandSlice(rng, n+rng.Intn(20), 1000)
		xa := dlyRandSlice(rng, n+rng.Intn(20), 1000)
		if trial%9 == 0 {
			// Small integers and zeros: sums that are exact in few bits.
			for k := range x {
				x[k] = float32(rng.Intn(9) - 4)
			}
			for k := range h {
				h[k] = float32(rng.Intn(5)-2) / 2
			}
		}
		y := float32(rng.NormFloat64() * 1000)
		alpha := float32(rng.NormFloat64() * 1e-4)
		if trial%9 == 1 {
			// Sums of few bits below 2^-126: the software fma leaves for
			// fmaf in the first group.
			for k := range x {
				x[k] = float32(rng.Intn(9)-4) * 0x1p-70
			}
			for k := range xa {
				xa[k] = float32(rng.Intn(9)-4) * 0x1p-70
			}
			for k := range h {
				h[k] = float32(rng.Intn(5)-2) * 0x1p-80
			}
			alpha = float32(rng.Intn(5)-2) * 0x1p-10
			y = float32(rng.Intn(5)-2) * 0x1p-75
		}

		for _, g := range []int{4, 8, 16} {
			mg := n &^ (g - 1)
			acc0 := dlyRandAcc(rng)

			// Dot: kernel, reference, literal.
			a, b, c := acc0, acc0, acc0
			dlyDotS(g, h[:mg], x, &a)
			switch g {
			case 4:
				dlyDotSNeonRef(h[:mg], x, (*[4]float32)(b[:4]))
			case 8:
				dlyDotSSse2Ref(h[:mg], x, (*[8]float32)(b[:8]))
			default:
				dlyDotSAvx2Ref(h[:mg], x, &b)
			}
			dlyLiteralDot(g, h[:mg], x, c[:g])
			dlyRequireBits(t, "dot", a[:], b[:])
			dlyRequireBits(t, "dot reference", b[:], c[:])

			// Sq: kernel, reference, literal.
			a, b, c = acc0, acc0, acc0
			dlySq(g, x[:mg], &a)
			switch g {
			case 4:
				dlySqNeonRef(x[:mg], (*[4]float32)(b[:4]))
			case 8:
				dlySqSse2Ref(x[:mg], (*[8]float32)(b[:8]))
			default:
				dlySqAvx2Ref(x[:mg], &b)
			}
			dlyLiteralDot(g, x[:mg], x[:mg], c[:g])
			dlyRequireBits(t, "sq", a[:], b[:])
			dlyRequireBits(t, "sq reference", b[:], c[:])

			// Update and dot: kernel, reference, and the separate update
			// pass followed by the literal dot.
			a, b, c = acc0, acc0, acc0
			hA := append([]float32(nil), h[:mg]...)
			hB := append([]float32(nil), h[:mg]...)
			hC := append([]float32(nil), h[:mg]...)
			dlyAxpyDotS(g, hA, xa, x, alpha, &a)
			switch g {
			case 4:
				dlyAxpyDotSNeonRef(hB, xa, x, alpha, (*[4]float32)(b[:4]))
			case 8:
				dlyAxpyDotSSse2Ref(hB, xa, x, alpha, (*[8]float32)(b[:8]))
			default:
				dlyAxpyFmaDotSAvx2Ref(hB, xa, x, alpha, &b)
			}
			for k := range hC {
				if g == 16 {
					hC[k] = fmaf(xa[k], alpha, hC[k])
				} else {
					hC[k] += float32(alpha * xa[k])
				}
			}
			dlyLiteralDot(g, hC, x, c[:g])
			dlyRequireBits(t, "axpyDot h", hA, hB)
			dlyRequireBits(t, "axpyDot acc", a[:], b[:])
			dlyRequireBits(t, "axpyDot reference h", hB, hC)
			dlyRequireBits(t, "axpyDot reference acc", b[:], c[:])

			// Sum kernels: from zero accumulators, reduced.
			var z [16]float32
			dlyLiteralDot(g, h[:mg], x, z[:g])
			dlyRequireBits(t, "dotSum", []float32{dlyDotSSum(g, h[:mg], x)}, []float32{dlyLiteralReduce(g, false, &z, 0)})
			hA = append(hA[:0], h[:mg]...)
			z = [16]float32{}
			dlyLiteralDot(g, hC, x, z[:g])
			sum := dlyAxpyDotSSum(g, hA, xa, x, alpha)
			dlyRequireBits(t, "axpyDotSum", append(hA, sum), append(hC, dlyLiteralReduce(g, false, &z, 0)))

			// Reductions.
			tail := float32(rng.NormFloat64())
			if trial%7 == 0 {
				tail = 0
			}
			for _, accErr := range []bool{false, true} {
				dlyRequireBits(t, "reduce", []float32{dlyReduceX2(g, accErr, &acc0, tail)},
					[]float32{dlyLiteralReduce(g, accErr, &acc0, tail)})
			}

			// Reductions of the lanes of a run of samples.
			var runLanes [dlyX2Run + 15]float32
			for i := range runLanes {
				runLanes[i] = float32(rng.NormFloat64() * 1e6)
				if trial%13 == 0 {
					runLanes[i] = float32(rng.Intn(3))
				}
			}
			for _, accErr := range []bool{false, true} {
				r := 1 + rng.Intn(dlyX2Run)
				var got, want [dlyX2Run]float32
				dlyX2Reduce(g, accErr, &runLanes, r, &got)
				for j := 0; j < r; j++ {
					var a [16]float32
					copy(a[:g], runLanes[j:j+g])
					want[j] = dlyLiteralReduce(g, accErr, &a, 0)
				}
				dlyRequireBits(t, "x2 reduce", got[:], want[:])
			}

			// Lane chains: lanes[a] is the chain that starts at x[a].
			if mg > 0 && len(x) > mg {
				lanes := make([]float32, min(len(x)-mg+g, g+rng.Intn(24)))
				dlyX2Lanes(g, x, mg, lanes)
				lanesM := make([]float32, len(lanes))
				dlyX2LanesMulti(g, x, mg, lanesM)
				dlyRequireBits(t, "x2 lanes multi", lanesM, lanes)
				lanesA := make([]float32, len(lanes))
				lanesB := make([]float32, len(lanes))
				dlyX2Lanes2(g, x, x[len(x)-mg-len(lanes)+g:], mg, lanesA, lanesB)
				lanesC := make([]float32, len(lanes))
				dlyX2LanesRef(g, x[len(x)-mg-len(lanes)+g:], mg, lanesC)
				dlyRequireBits(t, "x2 lanes pair A", lanesA, lanes)
				dlyRequireBits(t, "x2 lanes pair B", lanesB, lanesC)
				if g != 16 {
					// The sums of the windows from a table of squares.
					r := 1 + rng.Intn(min(dlyX2Run, len(x)-mg+1))
					var got, want [dlyX2Run]float32
					dlyX2Sums(g, false, x, mg, r, &want)
					dlyX2SumsSq(g, x, mg, r, dlyRandSlice(rng, dlyX2SqScratch(mg), 1), &got)
					dlyRequireBits(t, "x2 sums from squares", got[:r], want[:r])
				}
				for a0 := range lanes {
					var w [1]float32
					for m := 0; m < mg/g; m++ {
						xk := x[a0+g*m]
						if g == 16 {
							w[0] = fmaf(xk, xk, w[0])
						} else {
							w[0] += float32(xk * xk)
						}
					}
					dlyRequireBits(t, "x2 lanes", lanes[a0:a0+1], w[:])
				}
			}

			// Accumulated-error filter loops: kernel against reference, with
			// and without the fused update.
			accA := dlyRandSlice(rng, mg/4, 10)
			accB := append([]float32(nil), accA...)
			var sA, sB float32
			sA = dlyAccErr(g, h[:mg], x, y, accA)
			switch g {
			case 4:
				sB = dlyAccErrNeonRef(h[:mg], x, y, accB)
			case 8:
				sB = dlyAccErrSse2Ref(h[:mg], x, y, accB)
			default:
				sB = dlyAccErrAvx2Ref(h[:mg], x, y, accB)
			}
			dlyRequireBits(t, "accErr", append(accA, sA), append(accB, sB))

			accA = dlyRandSlice(rng, mg/4, 10)
			accB = append([]float32(nil), accA...)
			accC := append([]float32(nil), accA...)
			hA = append(hA[:0], h[:mg]...)
			hB = append(hB[:0], h[:mg]...)
			sA = dlyAxpyAccErr(g, hA, xa, x, alpha, y, accA)
			var sC float32
			switch g {
			case 4:
				sB = dlyAxpyAccErrNeonRef(hB, xa, x, alpha, y, accB)
				sC = dlyAccErrNeonRef(hC, x, y, accC)
			case 8:
				sB = dlyAxpyAccErrSse2Ref(hB, xa, x, alpha, y, accB)
				sC = dlyAccErrSse2Ref(hC, x, y, accC)
			default:
				sB = dlyAxpyFmaAccErrAvx2Ref(hB, xa, x, alpha, y, accB)
				sC = dlyAccErrAvx2Ref(hC, x, y, accC)
			}
			dlyRequireBits(t, "axpyAccErr h", hA, hB)
			dlyRequireBits(t, "axpyAccErr", append(accA, sA), append(accB, sB))
			dlyRequireBits(t, "axpyAccErr reference h", hB, hC)
			dlyRequireBits(t, "axpyAccErr reference", append(accB, sB), append(accC, sC))
		}

		// x*x sums of windows that wrap or are not whole groups, for runs of
		// start indices, against the literal chunk loops.
		if n > 0 {
			xSize := n + 1 + rng.Intn(2*n+20)
			xw := dlyRandSlice(rng, xSize, 1000)
			r := 1 + rng.Intn(dlyX2Run)
			lo := rng.Intn(xSize)
			if trial%2 == 0 {
				lo = max(0, xSize-n-rng.Intn(r+2))
			}
			r = min(r, xSize-lo)
			for _, g := range []int{4, 8, 16} {
				var got, ref [dlyX2Run]float32
				dlyX2Wrapped(g, xw, n, lo, r, &got)
				dlyX2WrappedRef(g, xw, n, lo, r, &ref)
				dlyRequireBits(t, "x2 wrapped", got[:r], ref[:r])
				var multi [dlyX2Run]float32
				dlyX2WrappedMulti(g, xw, n, lo, r, &multi)
				dlyRequireBits(t, "x2 wrapped multi", multi[:r], ref[:r])
				if g != 16 && n+r-1 <= xSize {
					var sq [dlyX2Run]float32
					dlyX2WrappedSq(g, xw, n, lo, r, dlyRandSlice(rng, dlyX2SqScratch(n), 1), &sq)
					dlyRequireBits(t, "x2 wrapped from squares", sq[:r], ref[:r])
				}
				for i := 0; i < r; i++ {
					a := lo + i
					var lanes [16]float32
					var tail float32
					chunk1 := min(n, xSize-a)
					xp := a
					for _, limit := range [2]int{chunk1, n - chunk1} {
						dlyLiteralDot(g, xw[xp:xp+limit&^(g-1)], xw[xp:xp+limit&^(g-1)], lanes[:g])
						for k := limit &^ (g - 1); k < limit; k++ {
							tail += float32(xw[xp+k] * xw[xp+k])
						}
						xp = 0
					}
					want := dlyLiteralReduce(g, false, &lanes, tail)
					dlyRequireBits(t, "x2 wrapped reference", ref[i:i+1], []float32{want})
				}
			}
		}

		// Two NEON-lane filters at once against one at a time.
		{
			m4 := n &^ 3
			hB0 := dlyRandSlice(rng, m4, 0.3)
			xB := dlyRandSlice(rng, m4+rng.Intn(5), 1000)
			xaB := dlyRandSlice(rng, m4+rng.Intn(5), 1000)
			alphaB := float32(rng.NormFloat64() * 1e-4)
			acc0, acc1 := dlyRandAcc(rng), dlyRandAcc(rng)
			a, b, c, d := acc0, acc1, acc0, acc1
			hA := append([]float32(nil), h[:m4]...)
			hB := append([]float32(nil), hB0...)
			hC := append([]float32(nil), h[:m4]...)
			hD := append([]float32(nil), hB0...)
			dlyDotS2Neon(hA, x, (*[4]float32)(a[:4]), hB, xB, (*[4]float32)(b[:4]))
			dlyDotSNeonRef(hC, x, (*[4]float32)(c[:4]))
			dlyDotSNeonRef(hD, xB, (*[4]float32)(d[:4]))
			dlyRequireBits(t, "dot2 A", a[:], c[:])
			dlyRequireBits(t, "dot2 B", b[:], d[:])
			dlyAxpyDotS2Neon(hA, xa, x, alpha, (*[4]float32)(a[:4]), hB, xaB, xB, alphaB, (*[4]float32)(b[:4]))
			dlyAxpyDotSNeonRef(hC, xa, x, alpha, (*[4]float32)(c[:4]))
			dlyAxpyDotSNeonRef(hD, xaB, xB, alphaB, (*[4]float32)(d[:4]))
			dlyAxpyDotDotS2Neon(hA, xa, x, alpha, (*[4]float32)(a[:4]), hB, xB, (*[4]float32)(b[:4]))
			dlyAxpyDotSNeonRef(hC, xa, x, alpha, (*[4]float32)(c[:4]))
			dlyDotSNeonRef(hD, xB, (*[4]float32)(d[:4]))
			dlyRequireBits(t, "axpyDot2 h A", hA, hC)
			dlyRequireBits(t, "axpyDot2 h B", hB, hD)
			dlyRequireBits(t, "axpyDot2 A", a[:], c[:])
			dlyRequireBits(t, "axpyDot2 B", b[:], d[:])
		}

		m := rng.Intn(n + 1)
		hA := append([]float32(nil), h[:m]...)
		hB := append([]float32(nil), h[:m]...)
		dlyAxpy(hA, x, alpha)
		dlyAxpyRef(hB, x, alpha)
		dlyRequireBits(t, "axpy", hA, hB)

		hA = append(hA[:0], h[:m]...)
		hB = append(hB[:0], h[:m]...)
		hC := append([]float32(nil), h[:m]...)
		dlyAxpyFma(hA, x, alpha)
		dlyAxpyFmaRef(hB, x, alpha)
		for k := range hC {
			hC[k] = fmaf(x[k], alpha, hC[k])
		}
		dlyRequireBits(t, "axpyFma", hA, hB)
		dlyRequireBits(t, "axpyFma reference", hB, hC)
	}
}

// TestFmafTieMatchesFmaf checks that the rounded sum is the result of fmaf
// wherever fmafTie does not hold, on random operands and on operands at and
// next to float32 rounding boundaries: normal and subnormal midpoints and
// the overflow threshold.
func TestFmafTieMatchesFmaf(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	check := func(a, b, c float32) {
		s := fmafSum(a, b, c)
		r := float32(s)
		suspect := fmafTie(s, float64(r))
		want := fmaf(a, b, c)
		if !suspect && math.Float32bits(r) != math.Float32bits(want) && !(r != r && want != want) {
			t.Fatalf("fmafTie(%v, %v, %v) = %v unflagged, fmaf = %v", a, b, c, r, want)
		}
		if fmafHard(s) && !suspect {
			// A sum that only fmafHard flags is tiny and not a boundary:
			// fmafSlow agrees with the rounded sum there.
			if m := math.Abs(s); !(m > 0 && m < 0x1p-126) {
				t.Fatalf("fmafTie(%v, %v, %v) misses a boundary pattern", a, b, c)
			}
		}
	}
	randF := func() float32 {
		switch rng.Intn(6) {
		case 0:
			return float32(rng.Intn(1<<12) - 1<<11)
		case 1:
			return math.Float32frombits(rng.Uint32()) // any pattern: subnormals, infinities, NaNs
		case 2:
			return float32(math.Ldexp(rng.Float64(), rng.Intn(80)-150))
		case 3:
			return float32(math.Ldexp(float64(1+rng.Intn(1<<13)), rng.Intn(30)-15))
		}
		return float32(rng.NormFloat64() * math.Pow(10, float64(rng.Intn(12)-6)))
	}
	iterations := 3000000
	if testing.Short() {
		iterations = 200000
	}
	for i := 0; i < iterations; i++ {
		a, b, c := randF(), randF(), randF()
		check(a, b, c)
		if i%4 == 0 {
			// a*b + c on the midpoint between c and its neighbor, or off it
			// by the last bits of the product.
			u := math.Float32frombits(math.Float32bits(c)+1) - c
			half := u / 2
			if half != 0 && half == half {
				check(half, 1, c)
				check(half, math.Float32frombits(math.Float32bits(1)+uint32(rng.Intn(3))), c)
				check(-half, 1, c)
				check(a, half/a, c)
			}
		}
	}
	for _, c := range []float32{math.MaxFloat32, -math.MaxFloat32, 0x1p-126, 0x1p-149, 0, 1} {
		for _, p := range []float32{0x1p103, 0x1p102, 0x1p-150, 0x1p-75, 0x1.8p-75, 0x1p-151, 0x1p-25} {
			check(p, 1, c)
			check(-p, 1, c)
			check(p, p, c)
			check(p, 0x1.000002p0, c)
		}
	}
}

// TestDlyMatchedFilterCoreLanesDiffer checks that the lanes have not
// collapsed into one: their reduction orders give different results on
// generic data.
func TestDlyMatchedFilterCoreLanesDiffer(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const hSize = 512
	x := dlyRandSlice(rng, 1200, 3000)
	y := dlyRandSlice(rng, 16, 3000)
	results := map[Aec3Optimization][]float32{}
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		h := make([]float32, hSize)
		mf := &MatchedFilter{matchedFilterState: &matchedFilterState{optimization: opt}}
		var updated bool
		var errSum float32
		for rep := 0; rep < 20; rep++ {
			mf.runCore(1190-rep, 1, 0.7, x, y, h, &updated, &errSum, false)
		}
		results[opt] = append(h, errSum)
	}
	for _, a := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		for _, b := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
			if a >= b {
				continue
			}
			same := true
			for i := range results[a] {
				if math.Float32bits(results[a][i]) != math.Float32bits(results[b][i]) {
					same = false
					break
				}
			}
			if same {
				t.Errorf("lanes %d and %d produced identical results", a, b)
			}
		}
	}
}

func BenchmarkMatchedFilterCore(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	x := dlyRandSlice(rng, GetDownSampledBufferSize(4, 5), 3000)
	y := dlyRandSlice(rng, 16, 3000)
	for _, tc := range []struct {
		name     string
		opt      Aec3Optimization
		preEcho  bool
		startIdx int
	}{
		{"None", OptNone, false, 100}, {"Neon", OptNeon, false, 100}, {"Sse2", OptSse2, false, 100},
		{"Avx2", OptAvx2, false, 100}, {"NeonAccErr", OptNeon, true, 100}, {"Sse2AccErr", OptSse2, true, 100},
		{"Avx2AccErr", OptAvx2, true, 100},
	} {
		b.Run(tc.name, func(b *testing.B) {
			mf := NewMatchedFilter(tc.opt, 16, 32, 5, 24, 150, 0.7, 0.7, 0.2, true)
			h := dlyRandSlice(rng, 512, 0.01)
			var updated bool
			var errSum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mf.runCore(tc.startIdx, 1e30, 0.7, x, y, h, &updated, &errSum, tc.preEcho)
			}
		})
	}
}

// TestDlyAvx2RefFlaggedSums checks the AVX2 lane references against fmaf
// loops on data with float32 rounding ties at random places, so that their
// loops leave for the From functions in every lane and pass, after any
// number of groups.
//
// (2^12+1)^2 = 2^24 + 2^13 + 1 has 25 significant bits, and so has its sum
// with a multiple of 2^14: a tie, which rounds to even. Its sum with 2^-40
// lies above the tie, which is all the float64 sum keeps of it, and rounds
// up.
func TestDlyAvx2RefFlaggedSums(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	const tie = 4097
	// addend returns a value that leaves tie*tie on or just above a tie.
	addend := func() float32 {
		if rng.Intn(2) == 0 {
			return 0x1p-40
		}
		return float32(rng.Intn(3)) * 0x1p14
	}
	flagged := 0
	count := func(a, b, c float32) {
		s := fmafSum(a, b, c)
		if fmafTie(s, float64(float32(s))) {
			flagged++
		}
	}
	trials := 6000
	if testing.Short() {
		trials = 600
	}
	for trial := 0; trial < trials; trial++ {
		n := 16 * rng.Intn(9)
		h := dlyRandSlice(rng, n, 0.3)
		x := dlyRandSlice(rng, n, 1000)
		xa := dlyRandSlice(rng, n, 1000)
		alpha := float32(rng.NormFloat64() * 1e-4)
		var acc0 [16]float32
		if trial%2 == 0 {
			acc0 = dlyRandAcc(rng)
		}
		ties := rng.Intn(4)
		if trial%7 == 0 {
			// Ties in the update: alpha*xa + h.
			alpha = tie
		}
		for i := 0; i < ties && n > 0; i++ {
			k := rng.Intn(n)
			if alpha == tie {
				xa[k] = tie
				h[k] = addend()
			} else {
				// A tie in the accumulation needs the accumulator of the
				// lane on the grid of the product: zero the earlier elements
				// of the lane.
				for j := k % 16; j < k; j += 16 {
					h[j] = 0
				}
				acc0[k%16] = addend()
				h[k], x[k] = tie, tie
				if rng.Intn(2) == 0 {
					xa[k] = 0 // the update leaves the tie in place
				}
			}
		}

		// Dot.
		got, want := acc0, acc0
		dlyDotSAvx2Ref(h, x, &got)
		for k := range h {
			count(h[k], x[k], want[k%16])
			want[k%16] = fmaf(h[k], x[k], want[k%16])
		}
		dlyRequireBits(t, "dot", got[:], want[:])
		// The Wide references read the render samples widened.
		xw := make([]float64, n)
		xaw := make([]float64, n)
		for k := range xw {
			xw[k], xaw[k] = float64(x[k]), float64(xa[k])
		}
		got = acc0
		dlyDotSAvx2Wide(h, xw, &got)
		dlyRequireBits(t, "dot wide", got[:], want[:])

		// Squares: a tie element is a tie when the sum of its lane is on the
		// grid.
		got, want = acc0, acc0
		dlySqAvx2Ref(x, &got)
		for k := range x {
			count(x[k], x[k], want[k%16])
			want[k%16] = fmaf(x[k], x[k], want[k%16])
		}
		dlyRequireBits(t, "sq", got[:], want[:])
		got = acc0
		dlySqAvx2Wide(xw, &got)
		dlyRequireBits(t, "sq wide", got[:], want[:])

		// Update.
		hA := append([]float32(nil), h...)
		hW := append([]float32(nil), h...)
		dlyAxpyFmaRef(hA, xa, alpha)
		for k := range hW {
			count(xa[k], alpha, hW[k])
			hW[k] = fmaf(xa[k], alpha, hW[k])
		}
		dlyRequireBits(t, "axpy", hA, hW)

		// Update and dot.
		got, want = acc0, acc0
		hA = append(hA[:0], h...)
		dlyAxpyFmaDotSAvx2Ref(hA, xa, x, alpha, &got)
		for k := range hW {
			count(hW[k], x[k], want[k%16])
			want[k%16] = fmaf(hW[k], x[k], want[k%16])
		}
		dlyRequireBits(t, "axpyDot h", hA, hW)
		dlyRequireBits(t, "axpyDot", got[:], want[:])
		got = acc0
		hA = append(hA[:0], h...)
		dlyAxpyFmaDotSAvx2Wide(hA, xaw, xw, alpha, &got)
		dlyRequireBits(t, "axpyDot wide h", hA, hW)
		dlyRequireBits(t, "axpyDot wide", got[:], want[:])

		// Accumulated error, with and without the update: the error of a
		// group is a tie when its sum is zero and y is the root of the tie.
		y := float32(rng.NormFloat64() * 1000)
		if trial%3 == 0 {
			y = tie
			for k := range h[:n/2] {
				h[k] = 0
			}
		}
		accErr0 := make([]float32, n/4)
		for i := range accErr0 {
			accErr0[i] = addend()
		}
		literal := func(hh []float32) ([]float32, float32) {
			ae := append([]float32(nil), accErr0...)
			var s float32
			for g := range ae {
				p0, p1 := float32(hh[4*g]*x[4*g]), float32(hh[4*g+1]*x[4*g+1])
				p2, p3 := float32(hh[4*g+2]*x[4*g+2]), float32(hh[4*g+3]*x[4*g+3])
				s += (p0 + p1) + (p2 + p3)
				e := s - y
				count(e, e, ae[g])
				ae[g] = fmaf(e, e, ae[g])
			}
			return ae, s
		}
		aeA := append([]float32(nil), accErr0...)
		sA := dlyAccErrAvx2Ref(h, x, y, aeA)
		aeW, sW := literal(h)
		dlyRequireBits(t, "accErr", append(aeA, sA), append(aeW, sW))

		aeA = append(aeA[:0], accErr0...)
		hA = append(hA[:0], h...)
		hW = append(hW[:0], h...)
		sA = dlyAxpyFmaAccErrAvx2Ref(hA, xa, x, alpha, y, aeA)
		for k := range hW {
			hW[k] = fmaf(xa[k], alpha, hW[k])
		}
		aeW, sW = literal(hW)
		dlyRequireBits(t, "axpyAccErr h", hA, hW)
		dlyRequireBits(t, "axpyAccErr", append(aeA, sA), append(aeW, sW))
	}
	if flagged < trials {
		t.Fatalf("only %d flagged sums in %d trials", flagged, trials)
	}
}

// TestDlyMatchedFilterWideUpdateMatchesNarrow checks that an AVX2-lane
// MatchedFilter with work space for the widened render samples and one
// without have the same state after every block. A build that does not widen
// runs the same path twice.
func TestDlyMatchedFilterWideUpdateMatchesNarrow(t *testing.T) {
	numBlocks := 1500
	if testing.Short() {
		numBlocks = 300
	}
	for _, factor := range []int{4, 8} {
		for _, preEcho := range []bool{false, true} {
			limit := float32(150)
			if factor == 8 {
				limit = 20
			}
			sub := BlockSize / factor
			newFilter := func() *MatchedFilter {
				return NewMatchedFilter(OptAvx2, sub, MatchedFilterWindowSizeSubBlocks, 5,
					MatchedFilterAlignmentShiftSizeSubBlocks, limit, 0.7, 0.3, 0.2, preEcho)
			}
			wide, narrow := newFilter(), newFilter()
			room := new(dlyWide)
			s := newDlyQuietScene(int64(700+factor), factor, 5, math.Float32frombits(wide.quietLimitBits))
			reported := 0
			for blk := 0; blk < numBlocks; blk++ {
				mode := dlyQuietLoud
				if blk%400 > 300 {
					mode = dlyQuietBarelyAbove
				}
				y := s.block(mode, limit)
				if s.rng.Intn(40) == 0 {
					y[s.rng.Intn(len(y))] = 32700
				}
				slow := s.rng.Intn(3) == 0
				wide.UpdateWide(s.buf, y, slow, room)
				narrow.Update(s.buf, y, slow)
				if d := dlyQuietStateDiff(wide, narrow); d != "" {
					t.Fatalf("ds%d preEcho %v block %d: %s", factor, preEcho, blk, d)
				}
				if narrow.hasReportedLagEstimate {
					reported++
				}
				if s.rng.Intn(500) == 0 {
					wide.Reset(false)
					narrow.Reset(false)
				}
			}
			if reported < numBlocks/10 {
				t.Errorf("ds%d preEcho %v: only %d lag reports", factor, preEcho, reported)
			}
		}
	}
}
