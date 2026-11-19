package aec3

// Port of adaptive_fir_filter_avx2.cc.

// ComputeFrequencyResponseAvx2 is aec3::ComputeFrequencyResponse_Avx2: bins
// 0..63 compute fma(im, im, re*re) with a single rounding.
func ComputeFrequencyResponseAvx2(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptAvx2, numPartitions, H, H2)
}

// AdaptPartitionsAvx2 is aec3::AdaptPartitions_Avx2.
func AdaptPartitionsAvx2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptAvx2, renderBuffer, G, numPartitions, H)
}

// ApplyFilterAvx2 is aec3::ApplyFilter_Avx2.
func ApplyFilterAvx2(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptAvx2, renderBuffer, numPartitions, H, S)
}

// afirFreqRespBlockAvx2Go is the pure-Go reference of the AVX2 64-bin
// frequency-response block: re2 = re*re; re2 = fma(im, im, re2);
// H2 = _mm256_max_ps(H2, re2).
//
// The loop computes the fast path of fmaf (fmaf.go) for four bins at a time
// and makes no call: the first group with a sum that needs the slow path
// hands the rest of the block to afirFreqRespBlockAvx2From and returns.
func afirFreqRespBlockAvx2Go(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 4 {
		re := (*[4]float32)(h.Re[j : j+4])
		im := (*[4]float32)(h.Im[j : j+4])
		o := (*[4]float32)(h2[j : j+4])
		q0 := fmafSum(im[0], im[0], float32(re[0]*re[0]))
		q1 := fmafSum(im[1], im[1], float32(re[1]*re[1]))
		q2 := fmafSum(im[2], im[2], float32(re[2]*re[2]))
		q3 := fmafSum(im[3], im[3], float32(re[3]*re[3]))
		if fmafHard(q0) || fmafHard(q1) || fmafHard(q2) || fmafHard(q3) {
			afirFreqRespBlockAvx2From(h, h2, j)
			return
		}
		o[0] = sseMax(o[0], float32(q0))
		o[1] = sseMax(o[1], float32(q1))
		o[2] = sseMax(o[2], float32(q2))
		o[3] = sseMax(o[3], float32(q3))
	}
}

// afirFreqRespBlockAvx2From is the block for bins from..63, one fmaf per
// bin.
//
//go:noinline
func afirFreqRespBlockAvx2From(h *FftData, h2 *[FftLengthBy2Plus1]float32, from int) {
	for j := from; j < FftLengthBy2; j++ {
		h2[j] = sseMax(h2[j], fmaf(h.Im[j], h.Im[j], float32(h.Re[j]*h.Re[j])))
	}
}
