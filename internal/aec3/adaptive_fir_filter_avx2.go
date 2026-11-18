package aec3

// Port of adaptive_fir_filter_avx2.cc.

// ComputeFrequencyResponseAvx2 is aec3::ComputeFrequencyResponse_Avx2: bins
// 0..63 compute fma(im, im, re*re) with a single rounding.
func ComputeFrequencyResponseAvx2(numPartitions int, H [][]FftData, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptAvx2, numPartitions, H, H2)
}

// AdaptPartitionsAvx2 is aec3::AdaptPartitions_Avx2.
func AdaptPartitionsAvx2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H [][]FftData) {
	afirAdaptPartitionsSimd(OptAvx2, renderBuffer, G, numPartitions, H)
}

// ApplyFilterAvx2 is aec3::ApplyFilter_Avx2.
func ApplyFilterAvx2(renderBuffer *RenderBuffer, numPartitions int, H [][]FftData, S *FftData) {
	afirApplyFilterSimd(OptAvx2, renderBuffer, numPartitions, H, S)
}

// afirFreqRespBlockAvx2Go is the pure-Go reference of the AVX2 64-bin
// frequency-response block: re2 = re*re; re2 = fma(im, im, re2);
// H2 = _mm256_max_ps(H2, re2).
func afirFreqRespBlockAvx2Go(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j++ {
		re2 := float32(h.Re[j] * h.Re[j])
		im := h.Im[j]
		q := fmafSum(im, im, re2)
		v := float32(q)
		if fmafHard(q) {
			v = fmafSlow(im, im, re2)
		}
		h2[j] = sseMax(h2[j], v)
	}
}
