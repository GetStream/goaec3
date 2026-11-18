package aec3

import "math"

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
		v := afirFmaf32(h.Im[j], h.Im[j], re2)
		h2[j] = sseMax(h2[j], v)
	}
}

// afirFmaf32 returns a*b+c rounded once to float32 (fmaf). The float64
// product of two float32 values is exact; the float64 sum is rounded to odd
// before the final conversion, which makes the double rounding exact.
func afirFmaf32(a, b, c float32) float32 {
	p := float64(float64(a) * float64(b)) // exact; explicit conversion blocks FMA contraction
	cc := float64(c)
	s := p + cc
	if math.IsInf(s, 0) || math.IsNaN(s) {
		return float32(s)
	}
	// TwoSum error term.
	bb := s - p
	e := (p - (s - bb)) + (cc - bb)
	if e != 0 {
		bits := math.Float64bits(s)
		if bits&1 == 0 {
			if (e > 0) == (s > 0) {
				bits++
			} else {
				bits--
			}
			s = math.Float64frombits(bits)
		}
	}
	return float32(s)
}
