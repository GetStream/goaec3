package ooura

import (
	"math"
	"testing"
)

type (
	fftSplitPairFunc        func(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch)
	inverseFftSplitPairFunc func(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch)
)

// comparePairs runs the pair transforms on every input together with a
// different input and checks both results against the single split
// transforms; with nan set, a NaN matches any NaN. The scratch and the
// outputs start poisoned with NaNs and the scratch is reused, so a result
// that depends on their contents fails.
func comparePairs(t *testing.T, name string, in [][128]float32, nan bool, fwd fftSplitPairFunc, inv inverseFftSplitPairFunc,
	sfwd fftSplitFunc, sinv inverseFftSplitFunc) {
	t.Helper()
	same := func(a, b float32) bool {
		return math.Float32bits(a) == math.Float32bits(b) || (nan && a != a && b != b)
	}
	var s PairScratch
	poisonScratch(&s)
	q := float32(math.NaN())
	for n := range in {
		a, b := in[n], in[(n*7+3)%len(in)]
		var re, im, wre, wim [2][65]float32
		for i := range re[0] {
			re[0][i], im[0][i], re[1][i], im[1][i] = q, q, q, q
		}
		x, y := a, b
		sfwd(&x, &wre[0], &wim[0])
		sfwd(&y, &wre[1], &wim[1])
		x, y = a, b
		fwd(&x, &y, &re[0], &im[0], &re[1], &im[1], &s)
		for f := 0; f < 2; f++ {
			for i := range re[f] {
				if !same(re[f][i], wre[f][i]) || !same(im[f][i], wim[f][i]) {
					t.Fatalf("%s fwd input %d frame %d bin %d: got (%v, %v) want (%v, %v)", name, n, f, i,
						re[f][i], im[f][i], wre[f][i], wim[f][i])
				}
			}
		}

		// The inverse of the inputs taken as bins.
		unpack(&a, &re[0], &im[0])
		unpack(&b, &re[1], &im[1])
		re[0][64], re[1][64] = a[1], b[1]
		im[0][0], im[0][64], im[1][0], im[1][64] = q, q, q, q
		var want, got [2][128]float32
		sinv(&re[0], &im[0], &want[0])
		sinv(&re[1], &im[1], &want[1])
		for i := range got[0] {
			got[0][i], got[1][i] = q, q
		}
		inv(&re[0], &im[0], &re[1], &im[1], &got[0], &got[1], &s)
		for f := 0; f < 2; f++ {
			for i := range got[f] {
				if !same(got[f][i], want[f][i]) {
					t.Fatalf("%s inv input %d frame %d idx %d: got %v want %v", name, n, f, i, got[f][i], want[f][i])
				}
			}
		}
	}
}

func TestFftSplitPairMatchesSingle(t *testing.T) {
	comparePairs(t, "build", testInputs(), false, FftSplitPair, InverseFftSplitPair, buildFftSplit, buildInverseFftSplit)
	comparePairs(t, "fast", testInputs(), false, FftSplitPair, InverseFftSplitPair, fftSplitFast, inverseFftSplitFast)
	comparePairs(t, "nonfinite", nonFiniteInputs(), true, FftSplitPair, InverseFftSplitPair, buildFftSplit, buildInverseFftSplit)
}

func TestFftSplitNoAllocs(t *testing.T) {
	var a, b [128]float32
	var re, im [2][65]float32
	var s PairScratch
	var s1 Scratch
	if n := testing.AllocsPerRun(100, func() {
		FftSplit(&a, &re[0], &im[0], &s1)
		InverseFftSplit(&re[0], &im[0], &a, &s1)
		FftSplitPair(&a, &b, &re[0], &im[0], &re[1], &im[1], &s)
		InverseFftSplitPair(&re[0], &im[0], &re[1], &im[1], &a, &b, &s)
	}); n != 0 {
		t.Fatalf("the split transforms allocate %v times", n)
	}
}

func benchSplitPair(b *testing.B, fwd fftSplitPairFunc, inv inverseFftSplitPairFunc) {
	in := testInputs()
	x0, y0 := in[0], in[1]
	var x, y [128]float32
	var re, im [2][65]float32
	var s PairScratch
	b.Run("fwd", func(b *testing.B) {
		for b.Loop() {
			x, y = x0, y0
			fwd(&x, &y, &re[0], &im[0], &re[1], &im[1], &s)
		}
	})
	b.Run("inv", func(b *testing.B) {
		for b.Loop() {
			inv(&re[0], &im[0], &re[1], &im[1], &x, &y, &s)
		}
	})
}

func twoSplits(f fftSplitFunc) fftSplitPairFunc {
	return func(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, _ *PairScratch) {
		f(a, re0, im0)
		f(b, re1, im1)
	}
}

func twoInverseSplits(f inverseFftSplitFunc) inverseFftSplitPairFunc {
	return func(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, _ *PairScratch) {
		f(re0, im0, a)
		f(re1, im1, b)
	}
}

// BenchmarkFftSplit times two transforms between frames and bins: the pair
// entry points and two single split transforms.
func BenchmarkFftSplit(b *testing.B) {
	b.Run("pair", func(b *testing.B) { benchSplitPair(b, FftSplitPair, InverseFftSplitPair) })
	b.Run("two", func(b *testing.B) {
		benchSplitPair(b, twoSplits(buildFftSplit), twoInverseSplits(buildInverseFftSplit))
	})
}
