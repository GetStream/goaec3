package aec3

// Port of adaptive_fir_filter_avx2.cc.

import "math"

// ComputeFrequencyResponseAvx2 mirrors aec3::ComputeFrequencyResponse_Avx2.
// Bins 0..63 are fma(im, im, re*re).
func ComputeFrequencyResponseAvx2(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptAvx2, numPartitions, H, H2)
}

// AdaptPartitionsAvx2 mirrors aec3::AdaptPartitions_Avx2.
func AdaptPartitionsAvx2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptAvx2, renderBuffer, G, numPartitions, H)
}

// ApplyFilterAvx2 mirrors aec3::ApplyFilter_Avx2.
func ApplyFilterAvx2(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptAvx2, renderBuffer, numPartitions, H, S)
}

// afirFreqRespBlockAvx2Go is the AVX2 lane reference for bins 0..63: h2 =
// _mm256_max_ps(h2, fma(im, im, re*re)).
//
// The loop makes no call (see fmaf). A group of four bins with a sum that
// needs fmafSlow, or with an operand of the maximum that is a NaN, negative or
// -0 (see afirFreqRespBlockSse2Fast), hands the rest of the block to
// afirFreqRespBlockAvx2From.
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
		m0 := max(math.Float32bits(o[0]), math.Float32bits(float32(q0)))
		m1 := max(math.Float32bits(o[1]), math.Float32bits(float32(q1)))
		m2 := max(math.Float32bits(o[2]), math.Float32bits(float32(q2)))
		m3 := max(math.Float32bits(o[3]), math.Float32bits(float32(q3)))
		if m0 > afirInfBits || m1 > afirInfBits || m2 > afirInfBits || m3 > afirInfBits {
			afirFreqRespBlockAvx2From(h, h2, j)
			return
		}
		o[0] = math.Float32frombits(m0)
		o[1] = math.Float32frombits(m1)
		o[2] = math.Float32frombits(m2)
		o[3] = math.Float32frombits(m3)
	}
}

// afirFreqRespBlockAvx2From is afirFreqRespBlockAvx2Go for bins from..63, one
// fmaf per bin.
//
//go:noinline
func afirFreqRespBlockAvx2From(h *FftData, h2 *[FftLengthBy2Plus1]float32, from int) {
	for j := from; j < FftLengthBy2; j++ {
		h2[j] = sseMax(h2[j], fmaf(h.Im[j], h.Im[j], float32(h.Re[j]*h.Re[j])))
	}
}

// afirFreqRespFirstAvx2Go is afirFreqRespBlockAvx2Go on an h2 of +0 (see
// afirFreqRespFirstGo): it stores fma(im, im, re*re) for bins 0..63.
func afirFreqRespFirstAvx2Go(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	hre, him := &h.Re, &h.Im
	for j := 0; j < FftLengthBy2; j += 4 {
		re := (*[4]float32)(hre[j : j+4])
		im := (*[4]float32)(him[j : j+4])
		o := (*[4]float32)(h2[j : j+4])
		q0 := fmafSum(im[0], im[0], float32(re[0]*re[0]))
		q1 := fmafSum(im[1], im[1], float32(re[1]*re[1]))
		q2 := fmafSum(im[2], im[2], float32(re[2]*re[2]))
		q3 := fmafSum(im[3], im[3], float32(re[3]*re[3]))
		if fmafHard(q0) || fmafHard(q1) || fmafHard(q2) || fmafHard(q3) {
			afirFreqRespFirstAvx2From(h, h2, j)
			return
		}
		o[0], o[1], o[2], o[3] = float32(q0), float32(q1), float32(q2), float32(q3)
	}
}

// afirFreqRespFirstAvx2From is afirFreqRespFirstAvx2Go for bins from..63,
// one fmaf per bin.
//
//go:noinline
func afirFreqRespFirstAvx2From(h *FftData, h2 *[FftLengthBy2Plus1]float32, from int) {
	for j := from; j < FftLengthBy2; j++ {
		h2[j] = fmaf(h.Im[j], h.Im[j], float32(h.Re[j]*h.Re[j]))
	}
}
