//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
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

// TestDlyAvx2KernelsForced runs the AVX2-lane archsimd kernels against their
// references even when CPUID does not report FMA (set DLY_FORCE_FMA=1; some
// emulators execute FMA instructions without advertising them).
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
}

// dlyBenchLegacyAxpyFmaDotAvx2 is the one-filter AVX2 kernel that
// accumulates x*x next to h*x (the layout of the C++ loop), kept as a
// benchmark baseline for the kernels that leave x*x to dlyX2Sums.
//
//go:noinline
func dlyBenchLegacyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, r *[4]float32) {
	n := len(h) &^ 15
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	var sv, s8, x2, x28 archsimd.Float32x8
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		sv = hk.MulAdd(xk, sv)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	ts, tx := dlyHsumSAvx2(sv, s8), dlyHsumSAvx2(x2, x28)
	ts.ConcatPermuteScalars(0, 0, 4, 4, tx).StoreArray(r)
	archsimd.ClearAVXUpperBits()
}

// dlyBenchLegacyAxpyDotSse2 is the SSE2-lane counterpart of
// dlyBenchLegacyAxpyFmaDotAvx2.
func dlyBenchLegacyAxpyDotSse2(h, xa, xb []float32, alpha float32, r *[4]float32) {
	n := len(h) &^ 7
	h = h[:n:n]
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	var sv, s4, x2, x24 archsimd.Float32x4
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		sv = sv.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
	}
	ts, tx := dlyHsumSSse2(sv, s4), dlyHsumSSse2(x2, x24)
	ts.ConcatPermuteScalars(0, 0, 4, 4, tx).StoreArray(r)
}

// dlyBenchKernels times the kernels of one capture sample on their own:
// 512 taps per filter, updating, with the layouts compared when the cores
// were arranged: x*x inside the kernel (Legacy) or not (Sum), one filter or
// two (Pair), accumulators in registers (Sum) or in memory (Mem).
func dlyBenchKernels(b *testing.B, x, h0 []float32) {
	hA := append([]float32(nil), h0...)
	hB := append([]float32(nil), h0...)
	const alpha = 1e-12
	win := func(i, shift int) (xa, xb []float32) {
		o := 100 + shift - i&63
		return x[o+1 : o+513], x[o : o+512]
	}
	if dlyUseFma {
		b.Run("KernAvx2Legacy", func(b *testing.B) {
			var r [4]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				dlyBenchLegacyAxpyFmaDotAvx2(hA, xa, xb, alpha, &r)
			}
		})
		b.Run("KernAvx2Sum", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				dlyAxpyFmaDotSSumAvx2(hA, xa, xb, alpha)
			}
		})
		b.Run("KernAvx2PairMem", func(b *testing.B) {
			var a, c [16]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				xc, xd := win(i, 384)
				dlyAxpyFmaDotS2Avx2(hA, xa, xb, alpha, &a, hB, xc, xd, alpha, &c)
			}
		})
		b.Run("KernAvx2PairSum", func(b *testing.B) {
			var r [4]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				xc, xd := win(i, 384)
				dlyAxpyFmaDotS2SumAvx2(hA, xa, xb, alpha, hB, xc, xd, alpha, &r)
			}
		})
	}
	if dlyUseAvx {
		b.Run("KernSse2Legacy", func(b *testing.B) {
			var r [4]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				dlyBenchLegacyAxpyDotSse2(hA, xa, xb, alpha, &r)
			}
		})
		b.Run("KernSse2Sum", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				dlyAxpyDotSSumSse2(hA, xa, xb, alpha)
			}
		})
		b.Run("KernSse2PairMem", func(b *testing.B) {
			var a, c [8]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				xc, xd := win(i, 384)
				dlyAxpyDotS2Sse2(hA, xa, xb, alpha, &a, hB, xc, xd, alpha, &c)
			}
		})
		b.Run("KernSse2PairSum", func(b *testing.B) {
			var r [4]float32
			for i := 0; i < b.N; i++ {
				xa, xb := win(i, 0)
				xc, xd := win(i, 384)
				dlyAxpyDotS2SumSse2(hA, xa, xb, alpha, hB, xc, xd, alpha, &r)
			}
		})
	}
}
