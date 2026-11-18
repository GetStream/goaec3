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
	switch opt {
	case OptNeon:
		afirFreqRespBlockNeonGo(h, h2)
	case OptSse2:
		afirFreqRespBlockSse2Go(h, h2)
	case OptAvx2:
		afirFreqRespBlockAvx2Go(h, h2)
	default:
		afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
	}
}

func afirErlBlock(_ Aec3Optimization, h2, erl *[FftLengthBy2Plus1]float32) {
	afirErlBlockGo(h2, erl)
}

// afirHasTermKernels is false: the lanes run their per-partition loops.
const afirHasTermKernels = false

func afirApplyTerms(xs, hs []*FftData, s *FftData) { afirApplyTermsGo(xs, hs, s) }

func afirAdaptTerms(xs, hs []*FftData, g *FftData) { afirAdaptTermsGo(xs, hs, g) }

// afirHasNeonFreqResp is false: the NEON lane runs afirFreqRespBlock per
// channel.
const afirHasNeonFreqResp = false

func afirFreqRespPartitionNEON(hs []FftData, h2 *[FftLengthBy2Plus1]float32) {
	panic("aec3: no NEON frequency response kernel")
}
