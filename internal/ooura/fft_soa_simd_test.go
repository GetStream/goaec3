//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import (
	"math"
	"testing"
)

func soaFftSplit(a *[128]float32, re, im *[65]float32) { fftSplitSoa(a, re, im, &splitScratch.t) }

func soaInverseFftSplit(re, im *[65]float32, a *[128]float32) {
	inverseFftSplitSoa(re, im, a, &splitScratch.t)
}

// TestSoaMatchesRef checks the split-form transforms against the lane
// reference with the FftData copy, with a poisoned work area.
func TestSoaMatchesRef(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	for i := range splitScratch.t {
		splitScratch.t[i] = float32(math.NaN())
	}
	compareSplit(t, "neon", soaFftSplit, soaInverseFftSplit, fftNeonRef, inverseFftNeonRef)
	compareSplit(t, "sse2", soaFftSplit, soaInverseFftSplit, fftSSE2Ref, inverseFftSSE2Ref)
}

// With NaNs and infinities in the input the split-form transforms produce
// the packed archsimd transforms, a NaN wherever those have one.
func TestSoaNonFinite(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	for n, a := range nonFiniteInputs() {
		var re, im, wre, wim [65]float32
		x, y := a, a
		fftSimd(&y)
		unpack(&y, &wre, &wim)
		soaFftSplit(&x, &re, &im)
		for i := range re {
			if !same32(re[i], wre[i]) || !same32(im[i], wim[i]) {
				t.Fatalf("fwd input %d bin %d: got (%#x, %#x) want (%#x, %#x)", n, i, math.Float32bits(re[i]),
					math.Float32bits(im[i]), math.Float32bits(wre[i]), math.Float32bits(wim[i]))
			}
		}
		unpack(&a, &re, &im)
		re[64] = a[1]
		var want, got [128]float32
		pack(&re, &im, &want)
		inverseFftSimd(&want)
		soaInverseFftSplit(&re, &im, &got)
		checkSameNaN(t, "inv", n, &got, &want)
	}
}

// The transposition of the cft1st passes.
func TestSoaShuffles(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	var got [4][4]float32
	r0, r1 := ld(&[4]float32{0, 1, 2, 3}), ld(&[4]float32{4, 5, 6, 7})
	r2, r3 := ld(&[4]float32{8, 9, 10, 11}), ld(&[4]float32{12, 13, 14, 15})
	soaLane0(r0, r1, r2, r3).StoreArray(&got[0])
	soaLane1(r0, r1, r2, r3).StoreArray(&got[1])
	soaLane2(r0, r1, r2, r3).StoreArray(&got[2])
	soaLane3(r0, r1, r2, r3).StoreArray(&got[3])
	if want := [4][4]float32{{0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}}; got != want {
		t.Errorf("soaLane0..3 = %v, want %v", got, want)
	}
	neg(ld(&[4]float32{0, 1, -2, 3})).StoreArray(&got[0])
	if want := [4]float32{float32(math.Copysign(0, -1)), -1, 2, -3}; math.Float32bits(got[0][0]) != math.Float32bits(want[0]) || got[0] != want {
		t.Errorf("neg = %v, want %v", got[0], want)
	}
}

// BenchmarkFftSoa times one split-form transform against the transform of
// fft_simd.go.
func BenchmarkFftSoa(b *testing.B) {
	if !useSimd {
		b.Skip("no AVX")
	}
	in := testInputs()
	var x [128]float32
	var re, im [65]float32
	var s Scratch
	b.Run("fwd", func(b *testing.B) {
		for b.Loop() {
			x = in[0]
			fftSplitSoa(&x, &re, &im, &s.t)
		}
	})
	b.Run("inv", func(b *testing.B) {
		for b.Loop() {
			inverseFftSplitSoa(&re, &im, &x, &s.t)
		}
	})
	b.Run("packed/fwd", func(b *testing.B) {
		for b.Loop() {
			x = in[0]
			fftSplitSimd(&x, &re, &im, &s.t)
		}
	})
	b.Run("packed/inv", func(b *testing.B) {
		for b.Loop() {
			inverseFftSplitSimd(&re, &im, &x, &s.t)
		}
	})
}
