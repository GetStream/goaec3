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
		afirFreqRespBlockSse2Fast(h, h2)
	case OptAvx2:
		afirFreqRespBlockAvx2Go(h, h2)
	default:
		afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
	}
}

func afirErlBlock(_ Aec3Optimization, h2, erl *[FftLengthBy2Plus1]float32) {
	afirErlBlockGo(h2, erl)
}

// afirHasTermKernels reports that afirApplyTerms/afirAdaptTerms are faster
// than the per-partition loops. Every lane computes the same rounded
// products and per-bin accumulation order, so one pure-Go kernel serves the
// NEON, SSE2 and AVX2 lanes.
const afirHasTermKernels = true

func afirApplyTerms(xs, hs []*FftData, s *FftData) { afirApplyTermsFastGo(xs, hs, s) }

func afirAdaptTerms(xs, hs []*FftData, g *FftData) { afirAdaptTermsFastGo(xs, hs, g) }

// afirHasNeonFreqResp is false: the NEON lane runs afirFreqRespBlock per
// channel.
const afirHasNeonFreqResp = false

func afirFreqRespPartitionNEON(hs []FftData, h2 *[FftLengthBy2Plus1]float32) {
	panic("aec3: no NEON frequency response kernel")
}

func afirMaxAbsUpdate(ir, h *[FftLengthBy2]float32) { afirMaxAbsUpdateGo(ir, h) }

// afirHasX86FreqResp is false: this build has no SSE2 or AVX2 partition
// kernel for the frequency response.
func afirHasX86FreqResp(Aec3Optimization) bool { return false }

func afirFreqRespPartitionX86(Aec3Optimization, []FftData, *[FftLengthBy2Plus1]float32) {
	panic("aec3: no x86 frequency response kernel")
}
