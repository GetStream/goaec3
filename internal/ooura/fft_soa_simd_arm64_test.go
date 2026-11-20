//go:build goexperiment.simd && arm64 && !purego

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
	for i := range splitScratch.t {
		splitScratch.t[i] = float32(math.NaN())
	}
	compareSplit(t, "neon", soaFftSplit, soaInverseFftSplit, fftNeonRef, inverseFftNeonRef)
}

// With NaNs and infinities in the input the split-form transforms produce
// the packed archsimd transforms, a NaN wherever those have one.
func TestSoaNonFinite(t *testing.T) {
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

// The transposition helpers of the cft1st passes.
func TestSoaShuffles(t *testing.T) {
	x := ld(&[4]float32{0, 1, 2, 3})
	y := ld(&[4]float32{4, 5, 6, 7})
	var got [4]float32
	trn1(x, y).StoreArray(&got)
	if want := [4]float32{0, 4, 2, 6}; got != want {
		t.Errorf("trn1 = %v, want %v", got, want)
	}
	trn2(x, y).StoreArray(&got)
	if want := [4]float32{1, 5, 3, 7}; got != want {
		t.Errorf("trn2 = %v, want %v", got, want)
	}
}

// BenchmarkFftSoa times one split-form transform against the transform of
// fft_simd.go.
func BenchmarkFftSoa(b *testing.B) {
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
