//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// Every lane runs its pure-Go reference.

func afirApplyBlock(_ Aec3Optimization, x, h, s *FftData) {
	afirApplyBinsGo(x, h, s, 0, FftLengthBy2)
}

func afirAdaptBlock(_ Aec3Optimization, x, g, h *FftData) {
	afirAdaptBinsGo(x, g, h, 0, FftLengthBy2)
}

func afirFreqRespBlock(opt Aec3Optimization, h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	if opt == OptAvx2 {
		afirFreqRespBlockAvx2Go(h, h2)
		return
	}
	afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
}

func afirErlBlock(_ Aec3Optimization, h2, erl *[FftLengthBy2Plus1]float32) {
	afirErlBlockGo(h2, erl)
}
