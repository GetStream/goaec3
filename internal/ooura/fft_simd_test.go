//go:build goexperiment.simd && (arm64 || amd64) && !purego

package ooura

import (
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
	check("ext3", ext3(x, y), [4]float32{3, 4, 5, 6})
}

func BenchmarkFftSimd(b *testing.B) {
	if !useSimd {
		b.Skip("no AVX")
	}
	b.Run("fwd", func(b *testing.B) { benchFft(b, fftSimd) })
	b.Run("inv", func(b *testing.B) { benchFft(b, inverseFftSimd) })
}
