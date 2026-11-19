//go:build goexperiment.simd && arm64 && !purego

package aec3

import "testing"

// dlyBenchKernels times the NEON kernels of one capture sample on their
// own: 512 taps per filter, updating, one filter (Sum) or two (Pair), and
// the accumulated-error filter loop.
func dlyBenchKernels(b *testing.B, x, h0 []float32) {
	hA := append([]float32(nil), h0...)
	hB := append([]float32(nil), h0...)
	accErr := make([]float32, len(h0)/4)
	const alpha = 1e-12
	win := func(i, shift int) (xa, xb []float32) {
		o := 100 + shift - i&63
		return x[o+1 : o+513], x[o : o+512]
	}
	b.Run("KernNeonSum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			xa, xb := win(i, 0)
			dlyAxpyDotSSum(4, hA, xa, xb, alpha)
		}
	})
	b.Run("KernNeonPair", func(b *testing.B) {
		var a, c [4]float32
		for i := 0; i < b.N; i++ {
			xa, xb := win(i, 0)
			xc, xd := win(i, 384)
			dlyAxpyDotS2Neon(hA, xa, xb, alpha, &a, hB, xc, xd, alpha, &c)
		}
	})
	b.Run("KernNeonAccErr", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			xa, xb := win(i, 0)
			dlyAxpyAccErrNeon(hA, xa, xb, alpha, 100, accErr)
		}
	})
}
