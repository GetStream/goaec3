//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import (
	"math"
	"simd/archsimd"
	"testing"
)

// TestSimdMatchesRef checks the archsimd kernel against both lane
// references (they are identical; see TestSSE2RefMatchesNeonRef).
func TestSimdMatchesRef(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	compareFfts(t, "fwd/neon", fftSimd, fftNeonRef)
	compareFfts(t, "inv/neon", inverseFftSimd, inverseFftNeonRef)
	compareFfts(t, "fwd/sse2", fftSimd, fftSSE2Ref)
	compareFfts(t, "inv/sse2", inverseFftSimd, inverseFftSSE2Ref)
}

// TestSplitSimdMatchesRef checks the split archsimd transforms against the
// lane references with the FftData copy.
func TestSplitSimdMatchesRef(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	for i := range splitScratch.t {
		splitScratch.t[i] = float32(math.NaN())
	}
	compareSplit(t, "neon", simdFftSplit, simdInverseFftSplit, fftNeonRef, inverseFftNeonRef)
	compareSplit(t, "sse2", simdFftSplit, simdInverseFftSplit, fftSSE2Ref, inverseFftSSE2Ref)
}

func simdFftSplit(a *[128]float32, re, im *[65]float32) { fftSplitSimd(a, re, im, &splitScratch.t) }

func simdInverseFftSplit(re, im *[65]float32, a *[128]float32) {
	inverseFftSplitSimd(re, im, a, &splitScratch.t)
}

// With NaNs and infinities in the input the split transforms produce the
// packed archsimd transforms, a NaN wherever those have one.
func TestSplitSimdNonFinite(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	for n, a := range nonFiniteInputs() {
		var re, im, wre, wim [65]float32
		x, y := a, a
		fftSimd(&y)
		unpack(&y, &wre, &wim)
		simdFftSplit(&x, &re, &im)
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
		simdInverseFftSplit(&re, &im, &got)
		checkSameNaN(t, "inv", n, &got, &want)
	}
}

// The lane helpers of the split passes.
func TestSplitShuffles(t *testing.T) {
	if !useSimd {
		t.Skip("no AVX")
	}
	x := archsimd.LoadFloat32x4Array(&[4]float32{0, 1, 2, 3})
	y := archsimd.LoadFloat32x4Array(&[4]float32{4, 5, 6, 7})
	check := func(name string, v vec, want [4]float32) {
		t.Helper()
		var got [4]float32
		v.StoreArray(&got)
		if got != want {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	check("rev4", rev4(x), [4]float32{3, 2, 1, 0})
	check("revEven", revEven(y, x), [4]float32{6, 4, 2, 0})
	check("revOdd", revOdd(y, x), [4]float32{7, 5, 3, 1})
	check("ext3", ext3(x, y), [4]float32{3, 4, 5, 6})
	check("loHi", loHi(x, y), [4]float32{0, 1, 6, 7})
}

func BenchmarkFftSimd(b *testing.B) {
	if !useSimd {
		b.Skip("no AVX")
	}
	b.Run("fwd", func(b *testing.B) { benchFft(b, fftSimd) })
	b.Run("inv", func(b *testing.B) { benchFft(b, inverseFftSimd) })
}
