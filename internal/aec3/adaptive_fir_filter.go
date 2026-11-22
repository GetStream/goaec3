package aec3

// Port of adaptive_fir_filter.{h,cc}.
//
// The scalar, NEON, SSE2 and AVX2 lanes of ApplyFilter and AdaptPartitions
// compute every bin with the same rounded products in the same accumulation
// order; they differ in how the partition loop is split around the ring wrap
// and in running bin kFftLengthBy2 in a second pass. Only
// ComputeFrequencyResponse_Avx2 differs numerically: it is fused.

import "math"

// ComputeFrequencyResponse mirrors aec3::ComputeFrequencyResponse.
func ComputeFrequencyResponse(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	for i := range H2 {
		H2[i] = [FftLengthBy2Plus1]float32{}
	}
	numRenderChannels := H.NumChannels()
	for p := 0; p < numPartitions; p++ {
		h2 := &H2[p]
		for ch := 0; ch < numRenderChannels; ch++ {
			afirFreqRespBinsGo(&H.At(p)[ch], h2, 0, FftLengthBy2Plus1)
		}
	}
}

// ComputeFrequencyResponseNeon mirrors aec3::ComputeFrequencyResponse_Neon.
func ComputeFrequencyResponseNeon(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptNeon, numPartitions, H, H2)
}

// ComputeFrequencyResponseSse2 mirrors aec3::ComputeFrequencyResponse_Sse2.
func ComputeFrequencyResponseSse2(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptSse2, numPartitions, H, H2)
}

func afirComputeFrequencyResponseSimd(opt Aec3Optimization, numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	if opt == OptNeon && afirHasNeonFreqResp {
		for p := 0; p < numPartitions; p++ {
			afirFreqRespPartitionNEON(H.At(p), &H2[p])
		}
		for i := numPartitions; i < len(H2); i++ {
			H2[i] = [FftLengthBy2Plus1]float32{}
		}
		return
	}
	if afirHasX86FreqResp(opt) {
		for p := 0; p < numPartitions; p++ {
			afirFreqRespPartitionX86(opt, H.At(p), &H2[p])
		}
		for i := numPartitions; i < len(H2); i++ {
			H2[i] = [FftLengthBy2Plus1]float32{}
		}
		return
	}
	numRenderChannels := H.NumChannels()
	if numRenderChannels == 0 {
		numPartitions = 0
	}
	for p := 0; p < numPartitions; p++ {
		h2 := &H2[p]
		hp := H.At(p)
		// The maximum of the cleared H2 and the first channel is that channel.
		afirFreqRespFirstGo(opt, &hp[0], h2)
		afirFreqRespBinsGo(&hp[0], h2, FftLengthBy2, FftLengthBy2Plus1)
		for ch := 1; ch < numRenderChannels; ch++ {
			afirFreqRespBlock(opt, &hp[ch], h2)
			afirFreqRespBinsGo(&hp[ch], h2, FftLengthBy2, FftLengthBy2Plus1)
		}
	}
	for i := numPartitions; i < len(H2); i++ {
		H2[i] = [FftLengthBy2Plus1]float32{}
	}
}

// afirFreqRespFirstGo sets bins 0..63 of h2 to the frequency response of h and
// bin 64 to +0. It is afirFreqRespBlock on an h2 of +0: a power is +0,
// positive, +Inf or NaN, and the maximum of +0 and such a value is the value
// in every SIMD lane.
func afirFreqRespFirstGo(opt Aec3Optimization, h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	switch opt {
	case OptNeon, OptSse2:
		hre, him := &h.Re, &h.Im
		for j := 0; j < FftLengthBy2; j += 4 {
			re := (*[4]float32)(hre[j : j+4])
			im := (*[4]float32)(him[j : j+4])
			o := (*[4]float32)(h2[j : j+4])
			t0 := float32(re[0]*re[0]) + float32(im[0]*im[0])
			t1 := float32(re[1]*re[1]) + float32(im[1]*im[1])
			t2 := float32(re[2]*re[2]) + float32(im[2]*im[2])
			t3 := float32(re[3]*re[3]) + float32(im[3]*im[3])
			o[0], o[1], o[2], o[3] = t0, t1, t2, t3
		}
		h2[FftLengthBy2] = 0
	case OptAvx2:
		afirFreqRespFirstAvx2Go(h, h2)
		h2[FftLengthBy2] = 0
	default:
		*h2 = [FftLengthBy2Plus1]float32{}
		afirFreqRespBlock(opt, h, h2)
	}
}

// afirFreqRespBinsGo sets h2 = std::max(h2, |h|^2) over bins [from, to).
func afirFreqRespBinsGo(h *FftData, h2 *[FftLengthBy2Plus1]float32, from, to int) {
	for j := from; j < to; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		if h2[j] < tmp {
			h2[j] = tmp
		}
	}
}

// afirFreqRespBlockNeonGo is the NEON lane reference for bins 0..63: h2 =
// vmaxq_f32(h2, re*re + im*im), unfused. vmaxq_f32 is Go's max.
func afirFreqRespBlockNeonGo(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		h2[j] = max(h2[j], tmp)
	}
}

// afirFreqRespBlockSse2Go is the SSE2 lane reference for bins 0..63: h2 =
// _mm_max_ps(h2, re*re + im*im).
func afirFreqRespBlockSse2Go(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		h2[j] = sseMax(h2[j], tmp)
	}
}

// afirInfBits is the bit pattern of +Inf. The patterns at or below it are
// those of +0, the positive numbers and +Inf, ordered like their values.
const afirInfBits = 0x7F800000

// afirFreqRespBlockSse2Fast is afirFreqRespBlockSse2Go without a branch on the
// data.
//
// For a and b that are +0, positive or +Inf, _mm_max_ps(a, b) is the value
// with the larger bit pattern, and that pattern exceeds afirInfBits exactly
// when a or b is a NaN, negative or -0. The first group of four bins with such
// a value hands the rest of the block to afirFreqRespBlockSse2From.
func afirFreqRespBlockSse2Fast(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 4 {
		re := (*[4]float32)(h.Re[j : j+4])
		im := (*[4]float32)(h.Im[j : j+4])
		o := (*[4]float32)(h2[j : j+4])
		t0 := float32(re[0]*re[0]) + float32(im[0]*im[0])
		t1 := float32(re[1]*re[1]) + float32(im[1]*im[1])
		t2 := float32(re[2]*re[2]) + float32(im[2]*im[2])
		t3 := float32(re[3]*re[3]) + float32(im[3]*im[3])
		m0 := max(math.Float32bits(o[0]), math.Float32bits(t0))
		m1 := max(math.Float32bits(o[1]), math.Float32bits(t1))
		m2 := max(math.Float32bits(o[2]), math.Float32bits(t2))
		m3 := max(math.Float32bits(o[3]), math.Float32bits(t3))
		if m0 > afirInfBits || m1 > afirInfBits || m2 > afirInfBits || m3 > afirInfBits {
			afirFreqRespBlockSse2From(h, h2, j)
			return
		}
		o[0] = math.Float32frombits(m0)
		o[1] = math.Float32frombits(m1)
		o[2] = math.Float32frombits(m2)
		o[3] = math.Float32frombits(m3)
	}
}

// afirFreqRespBlockSse2From is afirFreqRespBlockSse2Go for bins from..63.
//
//go:noinline
func afirFreqRespBlockSse2From(h *FftData, h2 *[FftLengthBy2Plus1]float32, from int) {
	for j := from; j < FftLengthBy2; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		h2[j] = sseMax(h2[j], tmp)
	}
}

// AdaptPartitions mirrors aec3::AdaptPartitions: H += G*conj(X).
func AdaptPartitions(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	data := renderBuffer.GetFftBuffer()
	index := renderBuffer.Position()
	numRenderChannels := data.NumChannels()
	for p := 0; p < numPartitions; p++ {
		x := data.At(index)
		for ch := 0; ch < numRenderChannels; ch++ {
			afirAdaptAllGo(&x[ch], G, &H.At(p)[ch])
		}
		if index < data.Size-1 {
			index++
		} else {
			index = 0
		}
	}
}

// AdaptPartitionsNeon mirrors aec3::AdaptPartitions_Neon.
func AdaptPartitionsNeon(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptNeon, renderBuffer, G, numPartitions, H)
}

// AdaptPartitionsSse2 mirrors aec3::AdaptPartitions_Sse2.
func AdaptPartitionsSse2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptSse2, renderBuffer, G, numPartitions, H)
}

func afirAdaptPartitionsSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if n := numPartitions * numRenderChannels; afirHasTermKernels && n <= afirMaxTerms {
		// The lists are zeroed per call: short filters take short ones.
		if n <= afirFewTerms {
			var xs, hs [afirFewTerms]*FftData
			afirTerms(data, renderBuffer.Position(), H, xs[:n], hs[:n])
			afirAdaptTerms(xs[:n], hs[:n], G)
			return
		}
		var xs, hs [afirMaxTerms]*FftData
		afirTerms(data, renderBuffer.Position(), H, xs[:n], hs[:n])
		afirAdaptTerms(xs[:n], hs[:n], G)
		return
	}
	lim1 := min(data.Size-renderBuffer.Position(), numPartitions)
	lim2 := numPartitions

	xPartition := renderBuffer.Position()
	limit := lim1
	p := 0
	for {
		for ; p < limit; p, xPartition = p+1, xPartition+1 {
			for ch := 0; ch < numRenderChannels; ch++ {
				afirAdaptBlock(opt, &data.At(xPartition)[ch], G, &H.At(p)[ch])
			}
		}
		xPartition = 0
		limit = lim2
		if p >= lim2 {
			break
		}
	}

	xPartition = renderBuffer.Position()
	limit = lim1
	p = 0
	for {
		for ; p < limit; p, xPartition = p+1, xPartition+1 {
			for ch := 0; ch < numRenderChannels; ch++ {
				afirAdaptBinsGo(&data.At(xPartition)[ch], G, &H.At(p)[ch], FftLengthBy2, FftLengthBy2Plus1)
			}
		}
		xPartition = 0
		limit = lim2
		if p >= lim2 {
			break
		}
	}
}

// afirAdaptBinsGo adds conj(x)*g to h over bins [from, to).
func afirAdaptBinsGo(x, g, h *FftData, from, to int) {
	for k := from; k < to; k++ {
		h.Re[k] += float32(x.Re[k]*g.Re[k]) + float32(x.Im[k]*g.Im[k])
		h.Im[k] += float32(x.Re[k]*g.Im[k]) - float32(x.Im[k]*g.Re[k])
	}
}

// ApplyFilter mirrors aec3::ApplyFilter.
func ApplyFilter(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	S.Re = [FftLengthBy2Plus1]float32{}
	S.Im = [FftLengthBy2Plus1]float32{}
	data := renderBuffer.GetFftBuffer()
	index := renderBuffer.Position()
	numRenderChannels := data.NumChannels()
	for p := 0; p < numPartitions; p++ {
		x := data.At(index)
		for ch := 0; ch < numRenderChannels; ch++ {
			afirApplyAllGo(&x[ch], &H.At(p)[ch], S)
		}
		if index < data.Size-1 {
			index++
		} else {
			index = 0
		}
	}
}

// ApplyFilterNeon mirrors aec3::ApplyFilter_Neon.
func ApplyFilterNeon(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptNeon, renderBuffer, numPartitions, H, S)
}

// ApplyFilterSse2 mirrors aec3::ApplyFilter_Sse2.
func ApplyFilterSse2(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptSse2, renderBuffer, numPartitions, H, S)
}

func afirApplyFilterSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if n := numPartitions * numRenderChannels; afirHasTermKernels && n <= afirMaxTerms {
		// The lists are zeroed per call: short filters take short ones.
		if n <= afirFewTerms {
			var xs, hs [afirFewTerms]*FftData
			afirTerms(data, renderBuffer.Position(), H, xs[:n], hs[:n])
			afirApplyTerms(xs[:n], hs[:n], S)
			return
		}
		var xs, hs [afirMaxTerms]*FftData
		afirTerms(data, renderBuffer.Position(), H, xs[:n], hs[:n])
		afirApplyTerms(xs[:n], hs[:n], S)
		return
	}
	S.Clear()
	lim1 := min(data.Size-renderBuffer.Position(), numPartitions)
	lim2 := numPartitions

	xPartition := renderBuffer.Position()
	p := 0
	limit := lim1
	for {
		for ; p < limit; p, xPartition = p+1, xPartition+1 {
			for ch := 0; ch < numRenderChannels; ch++ {
				afirApplyBlock(opt, &data.At(xPartition)[ch], &H.At(p)[ch], S)
			}
		}
		limit = lim2
		xPartition = 0
		if p >= lim2 {
			break
		}
	}

	xPartition = renderBuffer.Position()
	p = 0
	limit = lim1
	for {
		for ; p < limit; p, xPartition = p+1, xPartition+1 {
			for ch := 0; ch < numRenderChannels; ch++ {
				afirApplyBinsGo(&data.At(xPartition)[ch], &H.At(p)[ch], S, FftLengthBy2, FftLengthBy2Plus1)
			}
		}
		limit = lim2
		xPartition = 0
		if p >= lim2 {
			break
		}
	}
}

// afirMaxTerms is the largest number of (partition, channel) pairs the term
// kernels take; larger filters run the per-partition loops.
const afirMaxTerms = 64

// afirFewTerms is the length of the short pair lists. The default filters of
// one render channel have 13 partitions.
const afirFewTerms = 16

// afirTerms fills xs and hs with the (X, H) operand pairs of ApplyFilter and
// AdaptPartitions in the C++ accumulation order: partition-major,
// channel-minor, the render ring running from position and wrapping to 0.
func afirTerms(data *FftBuffer, position int, H FirCoefficients, xs, hs []*FftData) {
	nc := data.numChannels
	x := data.data
	h := H.data[:len(xs)]
	hs = hs[:len(xs)]
	if H.numChannels != nc {
		panic("aec3: filter and render buffer shapes differ")
	}
	// Both arrays are partition-major, so the pairs are consecutive elements.
	xi := position * nc
	for t := range h {
		if xi == len(x) {
			xi = 0
		}
		xs[t], hs[t] = &x[xi], &h[t]
		xi++
	}
}

// afirApplyTermsGo is the term kernel reference: it sets s to the sum of x*h
// over the pairs, accumulated per bin in pair order from +0.
func afirApplyTermsGo(xs, hs []*FftData, s *FftData) {
	s.Clear()
	hs = hs[:len(xs)]
	for t, x := range xs {
		afirApplyBinsGo(x, hs[t], s, 0, FftLengthBy2Plus1)
	}
}

// afirAdaptTermsGo is the term kernel reference: it adds conj(x)*g to the h of
// every pair.
func afirAdaptTermsGo(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	for t, x := range xs {
		afirAdaptBinsGo(x, g, hs[t], 0, FftLengthBy2Plus1)
	}
}

// afirApplyBinsGo adds x*h to s over bins [from, to).
func afirApplyBinsGo(x, h, s *FftData, from, to int) {
	for k := from; k < to; k++ {
		s.Re[k] += float32(x.Re[k]*h.Re[k]) - float32(x.Im[k]*h.Im[k])
		s.Im[k] += float32(x.Re[k]*h.Im[k]) + float32(x.Im[k]*h.Re[k])
	}
}

// afirApplyAllGo is afirApplyBinsGo over all bins.
func afirApplyAllGo(x, h, s *FftData) {
	xr, xi, hr, hi := &x.Re, &x.Im, &h.Re, &h.Im
	sr, si := &s.Re, &s.Im
	for k := range sr {
		a, b, c, d := xr[k], xi[k], hr[k], hi[k]
		sr[k] += float32(a*c) - float32(b*d)
		si[k] += float32(a*d) + float32(b*c)
	}
}

// afirAdaptAllGo is afirAdaptBinsGo over all bins. A step is eight bins where
// the float registers hold their operands and four on amd64. Loading the
// operands of a step before storing its results changes no result: a store to
// h reaches no operand of another bin, whatever h aliases.
func afirAdaptAllGo(x, g, h *FftData) {
	if isAMD64 {
		afirAdaptAll4Go(x, g, h)
	} else {
		afirAdaptAll8Go(x, g, h)
	}
}

func afirAdaptAll4Go(x, g, h *FftData) {
	xre, xim, gre, gim, hre, him := &x.Re, &x.Im, &g.Re, &g.Im, &h.Re, &h.Im
	for k := 0; k < FftLengthBy2; k += 4 {
		xr := (*[4]float32)(xre[k : k+4])
		xi := (*[4]float32)(xim[k : k+4])
		gr := (*[4]float32)(gre[k : k+4])
		gi := (*[4]float32)(gim[k : k+4])
		hr := (*[4]float32)(hre[k : k+4])
		hi := (*[4]float32)(him[k : k+4])
		// One bin after the other: four values are live at a time.
		a, b, c, d := xr[0], xi[0], gr[0], gi[0]
		hr[0] += float32(a*c) + float32(b*d)
		hi[0] += float32(a*d) - float32(b*c)
		a, b, c, d = xr[1], xi[1], gr[1], gi[1]
		hr[1] += float32(a*c) + float32(b*d)
		hi[1] += float32(a*d) - float32(b*c)
		a, b, c, d = xr[2], xi[2], gr[2], gi[2]
		hr[2] += float32(a*c) + float32(b*d)
		hi[2] += float32(a*d) - float32(b*c)
		a, b, c, d = xr[3], xi[3], gr[3], gi[3]
		hr[3] += float32(a*c) + float32(b*d)
		hi[3] += float32(a*d) - float32(b*c)
	}
	afirAdaptLastBin(x, g, h)
}

func afirAdaptAll8Go(x, g, h *FftData) {
	xre, xim, gre, gim, hre, him := &x.Re, &x.Im, &g.Re, &g.Im, &h.Re, &h.Im
	for k := 0; k < FftLengthBy2; k += 8 {
		xr := (*[8]float32)(xre[k : k+8])
		xi := (*[8]float32)(xim[k : k+8])
		gr := (*[8]float32)(gre[k : k+8])
		gi := (*[8]float32)(gim[k : k+8])
		hr := (*[8]float32)(hre[k : k+8])
		hi := (*[8]float32)(him[k : k+8])
		a0, b0, c0, d0 := xr[0], xi[0], gr[0], gi[0]
		a1, b1, c1, d1 := xr[1], xi[1], gr[1], gi[1]
		a2, b2, c2, d2 := xr[2], xi[2], gr[2], gi[2]
		a3, b3, c3, d3 := xr[3], xi[3], gr[3], gi[3]
		a4, b4, c4, d4 := xr[4], xi[4], gr[4], gi[4]
		a5, b5, c5, d5 := xr[5], xi[5], gr[5], gi[5]
		a6, b6, c6, d6 := xr[6], xi[6], gr[6], gi[6]
		a7, b7, c7, d7 := xr[7], xi[7], gr[7], gi[7]
		r0 := hr[0] + (float32(a0*c0) + float32(b0*d0))
		i0 := hi[0] + (float32(a0*d0) - float32(b0*c0))
		r1 := hr[1] + (float32(a1*c1) + float32(b1*d1))
		i1 := hi[1] + (float32(a1*d1) - float32(b1*c1))
		r2 := hr[2] + (float32(a2*c2) + float32(b2*d2))
		i2 := hi[2] + (float32(a2*d2) - float32(b2*c2))
		r3 := hr[3] + (float32(a3*c3) + float32(b3*d3))
		i3 := hi[3] + (float32(a3*d3) - float32(b3*c3))
		r4 := hr[4] + (float32(a4*c4) + float32(b4*d4))
		i4 := hi[4] + (float32(a4*d4) - float32(b4*c4))
		r5 := hr[5] + (float32(a5*c5) + float32(b5*d5))
		i5 := hi[5] + (float32(a5*d5) - float32(b5*c5))
		r6 := hr[6] + (float32(a6*c6) + float32(b6*d6))
		i6 := hi[6] + (float32(a6*d6) - float32(b6*c6))
		r7 := hr[7] + (float32(a7*c7) + float32(b7*d7))
		i7 := hi[7] + (float32(a7*d7) - float32(b7*c7))
		hr[0], hr[1], hr[2], hr[3], hr[4], hr[5], hr[6], hr[7] = r0, r1, r2, r3, r4, r5, r6, r7
		hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7] = i0, i1, i2, i3, i4, i5, i6, i7
	}
	afirAdaptLastBin(x, g, h)
}

// afirAdaptLastBin is afirAdaptBinsGo for bin FftLengthBy2.
func afirAdaptLastBin(x, g, h *FftData) {
	const n = FftLengthBy2
	a, b, c, d := x.Re[n], x.Im[n], g.Re[n], g.Im[n]
	h.Re[n] += float32(a*c) + float32(b*d)
	h.Im[n] += float32(a*d) - float32(b*c)
}

// afirApplyTermsFastGo is afirApplyTermsGo with the bin loop outside the pair
// loop: the sums of a group of bins stay in locals while the pairs accumulate
// into them in pair order from +0.
func afirApplyTermsFastGo(xs, hs []*FftData, s *FftData) {
	if isAMD64 {
		afirApplyTermsQuadGo(xs, hs, s)
	} else {
		afirApplyTermsWideGo(xs, hs, s)
	}
}

// afirApplyTermsWideGo keeps eight bins (sixteen sums) in locals, for
// architectures with thirty-two float registers.
func afirApplyTermsWideGo(xs, hs []*FftData, s *FftData) {
	afirApplyTermsWide(xs, hs[:len(xs)], s, len(xs))
	afirApplyTermsLastBin(xs, hs, s)
}

// afirApplyTermsWide is afirApplyTermsWideGo for the bins below FftLengthBy2;
// n is len(xs).
//
// A pair is two basic blocks of four bins, so that the sums and the terms of a
// block fit the registers. The test between the blocks never holds; the
// function does not inline, which keeps the compiler from removing it.
//
//go:noinline
func afirApplyTermsWide(xs, hs []*FftData, s *FftData, n int) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 8 {
		var r0, r1, r2, r3, r4, r5, r6, r7 float32
		var i0, i1, i2, i3, i4, i5, i6, i7 float32
		for t, x := range xs {
			h := hs[t]
			xr := (*[8]float32)(x.Re[k : k+8])
			xi := (*[8]float32)(x.Im[k : k+8])
			hr := (*[8]float32)(h.Re[k : k+8])
			hi := (*[8]float32)(h.Im[k : k+8])
			r0 += float32(xr[0]*hr[0]) - float32(xi[0]*hi[0])
			i0 += float32(xr[0]*hi[0]) + float32(xi[0]*hr[0])
			r1 += float32(xr[1]*hr[1]) - float32(xi[1]*hi[1])
			i1 += float32(xr[1]*hi[1]) + float32(xi[1]*hr[1])
			r2 += float32(xr[2]*hr[2]) - float32(xi[2]*hi[2])
			i2 += float32(xr[2]*hi[2]) + float32(xi[2]*hr[2])
			r3 += float32(xr[3]*hr[3]) - float32(xi[3]*hi[3])
			i3 += float32(xr[3]*hi[3]) + float32(xi[3]*hr[3])
			if t >= n {
				break
			}
			r4 += float32(xr[4]*hr[4]) - float32(xi[4]*hi[4])
			i4 += float32(xr[4]*hi[4]) + float32(xi[4]*hr[4])
			r5 += float32(xr[5]*hr[5]) - float32(xi[5]*hi[5])
			i5 += float32(xr[5]*hi[5]) + float32(xi[5]*hr[5])
			r6 += float32(xr[6]*hr[6]) - float32(xi[6]*hi[6])
			i6 += float32(xr[6]*hi[6]) + float32(xi[6]*hr[6])
			r7 += float32(xr[7]*hr[7]) - float32(xi[7]*hi[7])
			i7 += float32(xr[7]*hi[7]) + float32(xi[7]*hr[7])
		}
		sr := (*[8]float32)(s.Re[k : k+8])
		si := (*[8]float32)(s.Im[k : k+8])
		sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6], sr[7] = r0, r1, r2, r3, r4, r5, r6, r7
		si[0], si[1], si[2], si[3], si[4], si[5], si[6], si[7] = i0, i1, i2, i3, i4, i5, i6, i7
	}
}

// afirApplyTermsQuadGo keeps four bins (eight sums) in locals, for amd64. The
// loop is bound by its loads, so it takes one pass that spills rather than two
// passes that load every operand twice.
func afirApplyTermsQuadGo(xs, hs []*FftData, s *FftData) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 4 {
		var r0, r1, r2, r3, i0, i1, i2, i3 float32
		for t, x := range xs {
			h := hs[t]
			xr := (*[4]float32)(x.Re[k : k+4])
			xi := (*[4]float32)(x.Im[k : k+4])
			hr := (*[4]float32)(h.Re[k : k+4])
			hi := (*[4]float32)(h.Im[k : k+4])
			r0 += float32(xr[0]*hr[0]) - float32(xi[0]*hi[0])
			i0 += float32(xr[0]*hi[0]) + float32(xi[0]*hr[0])
			r1 += float32(xr[1]*hr[1]) - float32(xi[1]*hi[1])
			i1 += float32(xr[1]*hi[1]) + float32(xi[1]*hr[1])
			r2 += float32(xr[2]*hr[2]) - float32(xi[2]*hi[2])
			i2 += float32(xr[2]*hi[2]) + float32(xi[2]*hr[2])
			r3 += float32(xr[3]*hr[3]) - float32(xi[3]*hi[3])
			i3 += float32(xr[3]*hi[3]) + float32(xi[3]*hr[3])
		}
		sr := (*[4]float32)(s.Re[k : k+4])
		si := (*[4]float32)(s.Im[k : k+4])
		sr[0], sr[1], sr[2], sr[3] = r0, r1, r2, r3
		si[0], si[1], si[2], si[3] = i0, i1, i2, i3
	}
	afirApplyTermsLastBin(xs, hs, s)
}

// afirApplyTermsLastBin is afirApplyTermsGo for bin FftLengthBy2.
func afirApplyTermsLastBin(xs, hs []*FftData, s *FftData) {
	hs = hs[:len(xs)]
	const n = FftLengthBy2
	var re, im float32
	for t, x := range xs {
		h := hs[t]
		re += float32(x.Re[n]*h.Re[n]) - float32(x.Im[n]*h.Im[n])
		im += float32(x.Re[n]*h.Im[n]) + float32(x.Im[n]*h.Re[n])
	}
	s.Re[n], s.Im[n] = re, im
}

// afirAdaptTermsFastGo is afirAdaptTermsGo without bounds checks.
func afirAdaptTermsFastGo(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	for t, x := range xs {
		afirAdaptAllGo(x, g, hs[t])
	}
}

// afirZeroFilter clears the partitions [oldSize, newSize).
func afirZeroFilter(oldSize, newSize int, H FirCoefficients) {
	for p := oldSize; p < newSize; p++ {
		hp := H.At(p)
		for ch := range hp {
			hp[ch].Clear()
		}
	}
}

// afirMinSizeMinus1 returns std::min(a, n-1) in size_t arithmetic: n-1 wraps
// for n == 0.
func afirMinSizeMinus1(a, n int) int {
	if n == 0 {
		return a
	}
	return min(a, n-1)
}

// afirNextPartition returns a < n-1 ? a+1 : 0 in size_t arithmetic.
func afirNextPartition(a, n int) int {
	if uint(a) < uint(n)-1 {
		return a + 1
	}
	return 0
}

// FirCoefficients holds the coefficients of a partitioned filter,
// partition-major, in one pointer-free array.
type FirCoefficients struct {
	numChannels int
	data        []FftData
}

func NewFirCoefficients(numPartitions, numChannels int) FirCoefficients {
	return FirCoefficients{numChannels: numChannels, data: make([]FftData, numPartitions*numChannels)}
}

// At returns the coefficients of partition p, one element per channel.
func (h FirCoefficients) At(p int) []FftData {
	n := h.numChannels
	return h.data[p*n : (p+1)*n : (p+1)*n]
}

func (h FirCoefficients) NumChannels() int { return h.numChannels }

// NumPartitions returns the number of partitions.
func (h FirCoefficients) NumPartitions() int {
	if h.numChannels == 0 {
		return 0
	}
	return len(h.data) / h.numChannels
}

// Flat returns the coefficients partition-major. The result aliases h.
func (h FirCoefficients) Flat() []FftData { return h.data }

// Clone returns a copy with its own storage.
func (h FirCoefficients) Clone() FirCoefficients {
	return FirCoefficients{numChannels: h.numChannels, data: append([]FftData(nil), h.data...)}
}

// AdaptiveFirFilter is a frequency-domain partitioned adaptive filter
// (adaptive_fir_filter.h).
type AdaptiveFirFilter struct {
	*afirState
	H FirCoefficients
}

// afirState is the state of an AdaptiveFirFilter except its coefficients.
type afirState struct {
	hSpan                         span
	fft                           Aec3Fft
	optimization                  Aec3Optimization
	numRenderChannels             int
	maxSizePartitions             int
	sizeChangeDurationBlocks      int
	oneBySizeChangeDurationBlocks float32
	currentSizePartitions         int
	targetSizePartitions          int
	oldTargetSizePartitions       int
	sizeChangeCounter             int
	partitionToConstrain          int

	// Bookkeeping of the zero-gain skip and of updateFrequencyResponse, for
	// filters of at most afirTrackedPartitions partitions (tracked).
	//
	// A partition outside negZero holds no coefficient equal to -0. h2Stale
	// holds every partition written since updateFrequencyResponse computed its
	// response.
	tracked       bool
	skipDisabled  bool // tests: always adapt, always compute every response
	negZero       afirPartitionSet
	h2Stale       afirPartitionSet
	zeroGainSkips uint64 // adaptations skipped, for tests
}

// afirTrackedPartitions is the largest filter the zero-gain skip handles.
const afirTrackedPartitions = 256

// afirPartitionSet is a set of partitions in [0, afirTrackedPartitions).
type afirPartitionSet [afirTrackedPartitions / 64]uint64

func (s *afirPartitionSet) add(p int)      { s[uint(p)>>6&3] |= 1 << (uint(p) & 63) }
func (s *afirPartitionSet) remove(p int)   { s[uint(p)>>6&3] &^= 1 << (uint(p) & 63) }
func (s *afirPartitionSet) has(p int) bool { return s[uint(p)>>6&3]>>(uint(p)&63)&1 != 0 }

// afirBelowMask returns the elements of [0, n) that lie in word w.
func afirBelowMask(w, n int) uint64 {
	switch lo := w * 64; {
	case n >= lo+64:
		return ^uint64(0)
	case n > lo:
		return 1<<uint(n-lo) - 1
	}
	return 0
}

// addBelow adds [0, n).
func (s *afirPartitionSet) addBelow(n int) {
	for w := range s {
		s[w] |= afirBelowMask(w, n)
	}
}

// removeBelow removes [0, n).
func (s *afirPartitionSet) removeBelow(n int) {
	for w := range s {
		s[w] &^= afirBelowMask(w, n)
	}
}

// anyBelow reports whether the set meets [0, n).
func (s *afirPartitionSet) anyBelow(n int) bool {
	var acc uint64
	for w := range s {
		acc |= s[w] & afirBelowMask(w, n)
	}
	return acc != 0
}

// allBelow reports whether the set contains [0, n).
func (s *afirPartitionSet) allBelow(n int) bool {
	var acc uint64
	for w := range s {
		acc |= afirBelowMask(w, n) &^ s[w]
	}
	return acc == 0
}

// afirGainIsZero reports whether every element of g is +0 or -0.
func afirGainIsZero(g *FftData) bool {
	var acc uint32
	for k := range g.Re {
		acc |= math.Float32bits(g.Re[k]) | math.Float32bits(g.Im[k])
	}
	return acc<<1 == 0
}

// afirAllFinite reports whether no element of x is an infinity or a NaN.
func afirAllFinite(x *[FftLengthBy2Plus1]float32) bool {
	const exponent = 0x7f800000
	for _, v := range x {
		if math.Float32bits(v)&exponent == exponent {
			return false
		}
	}
	return true
}

// afirHoldsNegZero reports whether a coefficient of h is -0.
func afirHoldsNegZero(h []FftData) bool {
	const negZero = 0x80000000
	for ch := range h {
		c := &h[ch]
		for k := range c.Re {
			if math.Float32bits(c.Re[k]) == negZero || math.Float32bits(c.Im[k]) == negZero {
				return true
			}
		}
	}
	return false
}

func NewAdaptiveFirFilter(maxSizePartitions, initialSizePartitions, sizeChangeDurationBlocks, numRenderChannels int, optimization Aec3Optimization) *AdaptiveFirFilter {
	f := &AdaptiveFirFilter{}
	f.bind(&afirState{}, bindAlone)
	f.init(maxSizePartitions, initialSizePartitions, sizeChangeDurationBlocks, numRenderChannels, optimization,
		make([]FftData, maxSizePartitions*numRenderChannels))
	return f
}

// init sets up f in place. The coefficients live in storage, which holds
// maxSizePartitions*numRenderChannels elements.
func (f *AdaptiveFirFilter) init(maxSizePartitions, initialSizePartitions, sizeChangeDurationBlocks, numRenderChannels int,
	optimization Aec3Optimization, storage []FftData) {
	n := maxSizePartitions * numRenderChannels
	f.H = FirCoefficients{numChannels: numRenderChannels, data: storage[:n:n]}
	*f.afirState = afirState{
		optimization:             optimization,
		numRenderChannels:        numRenderChannels,
		maxSizePartitions:        maxSizePartitions,
		sizeChangeDurationBlocks: sizeChangeDurationBlocks,
		currentSizePartitions:    initialSizePartitions,
		targetSizePartitions:     initialSizePartitions,
		oldTargetSizePartitions:  initialSizePartitions,
	}
	f.oneBySizeChangeDurationBlocks = 1 / float32(f.sizeChangeDurationBlocks)
	f.tracked = maxSizePartitions <= afirTrackedPartitions
	f.zeroPartitions(0, f.maxSizePartitions)
	f.SetSizePartitions(f.currentSizePartitions, true)
}

// bind makes f the view of st.
func (f *AdaptiveFirFilter) bind(st *afirState, b *binder) {
	f.afirState = st
	f.H.numChannels = st.numRenderChannels
	b.fft(&f.H.data, &st.hSpan)
}

// HandleEchoPathChange clears the partitions beyond the current size.
func (f AdaptiveFirFilter) HandleEchoPathChange() {
	f.zeroPartitions(f.currentSizePartitions, f.maxSizePartitions)
}

// zeroPartitions sets the partitions [from, to) to +0.
func (f AdaptiveFirFilter) zeroPartitions(from, to int) {
	afirZeroFilter(from, to, f.H)
	if f.tracked {
		for p := from; p < to; p++ {
			f.negZero.remove(p)
			f.h2Stale.add(p)
		}
	}
}

// partitionWritten records a write to partition p that may have stored -0.
func (f AdaptiveFirFilter) partitionWritten(p int) {
	if f.tracked {
		f.negZero.add(p)
		f.h2Stale.add(p)
	}
}

// ZeroGainSkipsForTesting returns the number of adaptations the zero-gain skip
// left out.
func (f AdaptiveFirFilter) ZeroGainSkipsForTesting() uint64 { return f.zeroGainSkips }

// SizePartitions returns the current filter size.
func (f AdaptiveFirFilter) SizePartitions() int { return f.currentSizePartitions }

// MaxFilterSizePartitions returns the maximum filter size.
func (f AdaptiveFirFilter) MaxFilterSizePartitions() int { return f.maxSizePartitions }

// GetFilter returns the filter coefficients.
func (f AdaptiveFirFilter) GetFilter() FirCoefficients { return f.H }

// SetSizePartitions sets the target filter size, immediately or gradually.
func (f AdaptiveFirFilter) SetSizePartitions(size int, immediateEffect bool) {
	f.targetSizePartitions = min(f.maxSizePartitions, size)
	if immediateEffect {
		oldSizePartitions := f.currentSizePartitions
		f.currentSizePartitions = f.targetSizePartitions
		f.oldTargetSizePartitions = f.targetSizePartitions
		f.zeroPartitions(oldSizePartitions, f.currentSizePartitions)
		f.partitionToConstrain = afirMinSizeMinus1(f.partitionToConstrain, f.currentSizePartitions)
		f.sizeChangeCounter = 0
	} else {
		f.sizeChangeCounter = f.sizeChangeDurationBlocks
	}
}

func afirAverage(from, to, fromWeight float32) float32 {
	return float32(from*fromWeight) + float32(to*(1-fromWeight))
}

func (f AdaptiveFirFilter) updateSize() {
	oldSizePartitions := f.currentSizePartitions
	if f.sizeChangeCounter > 0 {
		f.sizeChangeCounter--
		changeFactor := float32(float32(f.sizeChangeCounter) * f.oneBySizeChangeDurationBlocks)
		f.currentSizePartitions = int(afirAverage(float32(f.oldTargetSizePartitions), float32(f.targetSizePartitions), changeFactor))
		f.partitionToConstrain = afirMinSizeMinus1(f.partitionToConstrain, f.currentSizePartitions)
	} else {
		f.currentSizePartitions = f.targetSizePartitions
		f.oldTargetSizePartitions = f.targetSizePartitions
	}
	f.zeroPartitions(oldSizePartitions, f.currentSizePartitions)
}

// Filter produces the filter output S.
func (f AdaptiveFirFilter) Filter(renderBuffer *RenderBuffer, S *FftData) {
	switch f.optimization {
	case OptSse2:
		ApplyFilterSse2(renderBuffer, f.currentSizePartitions, f.H, S)
	case OptAvx2:
		ApplyFilterAvx2(renderBuffer, f.currentSizePartitions, f.H, S)
	case OptNeon:
		ApplyFilterNeon(renderBuffer, f.currentSizePartitions, f.H, S)
	default:
		ApplyFilter(renderBuffer, f.currentSizePartitions, f.H, S)
	}
}

// Adapt mirrors Adapt(render_buffer, G).
func (f *AdaptiveFirFilter) Adapt(renderBuffer *RenderBuffer, G *FftData) {
	f.adaptAndUpdateSize(renderBuffer, G)
	f.constrain()
}

// AdaptWithImpulseResponse mirrors Adapt(render_buffer, G, impulse_response).
// *impulseResponse has capacity GetTimeDomainLength(MaxFilterSizePartitions())
// and is resized like the std::vector.
func (f *AdaptiveFirFilter) AdaptWithImpulseResponse(renderBuffer *RenderBuffer, G *FftData, impulseResponse *[]float32) {
	f.adaptAndUpdateSize(renderBuffer, G)
	f.constrainAndUpdateImpulseResponse(impulseResponse)
}

// ComputeFrequencyResponse resizes *H2 to SizePartitions() elements and fills
// it. The capacity of *H2 is at least MaxFilterSizePartitions().
func (f AdaptiveFirFilter) ComputeFrequencyResponse(H2 *[][FftLengthBy2Plus1]float32) {
	afirResize(H2, f.currentSizePartitions)
	f.frequencyResponse(f.currentSizePartitions, f.H, *H2)
}

// frequencyResponse fills H2 with the response of the first numPartitions
// partitions of H, in the lane of f.
func (f AdaptiveFirFilter) frequencyResponse(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	switch f.optimization {
	case OptSse2:
		ComputeFrequencyResponseSse2(numPartitions, H, H2)
	case OptAvx2:
		ComputeFrequencyResponseAvx2(numPartitions, H, H2)
	case OptNeon:
		ComputeFrequencyResponseNeon(numPartitions, H, H2)
	default:
		ComputeFrequencyResponse(numPartitions, H, H2)
	}
}

// updateFrequencyResponse is ComputeFrequencyResponse for an array that only
// this method writes. The response of a partition depends on that partition
// alone, so it recomputes the partitions in h2Stale.
//
// h2Stale covers every write: adaptation, constraining, scaling, SetFilter and
// clearing. A partition that enters a growing filter is cleared, so the
// element afirResize zeroes for it is stale too.
func (f AdaptiveFirFilter) updateFrequencyResponse(H2 *[][FftLengthBy2Plus1]float32) {
	n := f.currentSizePartitions
	if !f.tracked || f.skipDisabled || f.h2Stale.allBelow(n) {
		f.ComputeFrequencyResponse(H2)
		f.h2Stale.removeBelow(n)
		return
	}
	afirResize(H2, n)
	for p := 0; p < n; p++ {
		if f.h2Stale.has(p) {
			f.frequencyResponse(1, FirCoefficients{numChannels: f.H.numChannels, data: f.H.At(p)}, (*H2)[p:p+1])
		}
	}
	f.h2Stale.removeBelow(n)
}

// afirResize is std::vector::resize: new elements are zero.
func afirResize[T any](v *[]T, n int) {
	old := len(*v)
	if n > cap(*v) {
		nv := make([]T, n)
		copy(nv, *v)
		*v = nv
		return
	}
	*v = (*v)[:n]
	if n > old {
		clear((*v)[old:n])
	}
}

func (f AdaptiveFirFilter) adaptAndUpdateSize(renderBuffer *RenderBuffer, G *FftData) {
	f.adaptAndUpdateSizeKnownRender(renderBuffer, G, nil, 0)
}

// adaptAndUpdateSizeKnownRender is adaptAndUpdateSize for a caller that has
// X2, the sum of the render spectra of the first x2Partitions partitions
// (RenderBuffer.SpectralSum), or nil.
//
// It skips an adaptation that leaves every coefficient as it is: one with a
// gain of zeros and finite render values, on partitions without a -0.
//
// Every lane adds to a coefficient h, fused or not, products of render and
// gain values. With zero gains and finite render values the products and their
// sums are zeros, and h plus a zero is h bit for bit unless h is -0. A NaN h
// keeps its payload, since coefficients are quiet NaNs.
//
// The render values are finite when X2 is: a bin of a spectrum is re*re +
// im*im, which is Inf or NaN when re or im is, and a sum of such terms stays
// so. X2 covers the adaptation when the size after updateSize is at most
// x2Partitions and RenderBuffer.SpectraMatchFfts holds.
//
// An addition gives -0 only from a -0, so a partition gets one only from the
// writes that put it in negZero. The partitions that hold a -0 are adapted.
func (f AdaptiveFirFilter) adaptAndUpdateSizeKnownRender(renderBuffer *RenderBuffer, G *FftData,
	X2 *[FftLengthBy2Plus1]float32, x2Partitions int) {
	f.updateSize()
	n := f.currentSizePartitions
	if f.tracked && !f.skipDisabled && X2 != nil && n <= x2Partitions && afirGainIsZero(G) &&
		renderBuffer.SpectraMatchFfts() && afirAllFinite(X2) {
		f.zeroGainSkips++
		if f.negZero.anyBelow(n) {
			f.adaptNegZeroPartitions(renderBuffer, G, n)
		}
		return
	}
	if f.tracked {
		f.h2Stale.addBelow(n)
	}
	switch f.optimization {
	case OptSse2:
		AdaptPartitionsSse2(renderBuffer, G, f.currentSizePartitions, f.H)
	case OptAvx2:
		AdaptPartitionsAvx2(renderBuffer, G, f.currentSizePartitions, f.H)
	case OptNeon:
		AdaptPartitionsNeon(renderBuffer, G, f.currentSizePartitions, f.H)
	default:
		AdaptPartitions(renderBuffer, G, f.currentSizePartitions, f.H)
	}
}

// adaptNegZeroPartitions adapts the partitions below n that hold a -0 and
// removes the others from negZero.
func (f AdaptiveFirFilter) adaptNegZeroPartitions(renderBuffer *RenderBuffer, G *FftData, n int) {
	data := renderBuffer.GetFftBuffer()
	for p := 0; p < n; p++ {
		if !f.negZero.has(p) {
			continue
		}
		h := f.H.At(p)
		if afirHoldsNegZero(h) {
			// Partition p reads the render FFT p slots after the read
			// position.
			x := data.At((renderBuffer.Position() + p) % data.Size)
			for ch := range h {
				if f.optimization == OptNone {
					afirAdaptAllGo(&x[ch], G, &h[ch])
				} else {
					afirAdaptBlock(f.optimization, &x[ch], G, &h[ch])
					afirAdaptBinsGo(&x[ch], G, &h[ch], FftLengthBy2, FftLengthBy2Plus1)
				}
			}
			f.h2Stale.add(p)
			if afirHoldsNegZero(h) {
				continue
			}
		}
		f.negZero.remove(p)
	}
}

func (f *AdaptiveFirFilter) constrainAndUpdateImpulseResponse(impulseResponse *[]float32) {
	afirResize(impulseResponse, GetTimeDomainLength(f.currentSizePartitions))
	ir := *impulseResponse
	var h [FftLength]float32
	base := f.partitionToConstrain * FftLengthBy2
	// The partition of the impulse response takes the first channel as is.
	irp := (*[FftLengthBy2]float32)(ir[base : base+FftLengthBy2])
	if f.numRenderChannels == 0 {
		clear(irp[:])
	}

	for ch := 0; ch < f.numRenderChannels; ch++ {
		hp := &f.H.At(f.partitionToConstrain)[ch]
		f.fft.Ifft(hp, &h)
		const scale = 1.0 / FftLengthBy2
		kernScale(h[:FftLengthBy2], scale, h[:FftLengthBy2])
		clear(h[FftLengthBy2:])

		if ch == 0 {
			*irp = [FftLengthBy2]float32(h[:FftLengthBy2])
		} else {
			afirMaxAbsUpdate(irp, (*[FftLengthBy2]float32)(h[:FftLengthBy2]))
		}
		f.fft.Fft(&h, hp)
	}
	f.partitionWritten(f.partitionToConstrain)
	f.partitionToConstrain = afirNextPartition(f.partitionToConstrain, f.currentSizePartitions)
}

// afirConstrainPair is refined.constrainAndUpdateImpulseResponse followed by
// coarse.constrain, with the transforms of the two filters run as pairs. The
// filters have the same number of render channels and share no state.
func afirConstrainPair(fft *aec3FftPair, refined *AdaptiveFirFilter, impulseResponse *[]float32, coarse *AdaptiveFirFilter) {
	afirResize(impulseResponse, GetTimeDomainLength(refined.currentSizePartitions))
	ir := *impulseResponse
	base := refined.partitionToConstrain * FftLengthBy2
	irp := (*[FftLengthBy2]float32)(ir[base : base+FftLengthBy2])
	if refined.numRenderChannels == 0 {
		clear(irp[:])
	}
	hr, hc := &fft.x0, &fft.x1
	const scale = 1.0 / FftLengthBy2
	for ch := 0; ch < refined.numRenderChannels; ch++ {
		hpr := &refined.H.At(refined.partitionToConstrain)[ch]
		hpc := &coarse.H.At(coarse.partitionToConstrain)[ch]
		fft.Ifft(hpr, hpc, hr, hc)
		kernScale(hr[:FftLengthBy2], scale, hr[:FftLengthBy2])
		clear(hr[FftLengthBy2:])
		kernScale(hc[:FftLengthBy2], scale, hc[:FftLengthBy2])
		clear(hc[FftLengthBy2:])
		if ch == 0 {
			*irp = [FftLengthBy2]float32(hr[:FftLengthBy2])
		} else {
			afirMaxAbsUpdate(irp, (*[FftLengthBy2]float32)(hr[:FftLengthBy2]))
		}
		fft.Fft(hr, hc, hpr, hpc)
	}
	refined.partitionWritten(refined.partitionToConstrain)
	coarse.partitionWritten(coarse.partitionToConstrain)
	refined.partitionToConstrain = afirNextPartition(refined.partitionToConstrain, refined.currentSizePartitions)
	coarse.partitionToConstrain = afirNextPartition(coarse.partitionToConstrain, coarse.currentSizePartitions)
}

// afirMaxAbsUpdateGo sets ir[k] to h[k] where fabsf(ir[k]) < fabsf(h[k]).
func afirMaxAbsUpdateGo(ir, h *[FftLengthBy2]float32) {
	for k := range ir {
		if fabsf(ir[k]) < fabsf(h[k]) {
			ir[k] = h[k]
		}
	}
}

func (f *AdaptiveFirFilter) constrain() {
	var h [FftLength]float32
	for ch := 0; ch < f.numRenderChannels; ch++ {
		hp := &f.H.At(f.partitionToConstrain)[ch]
		f.fft.Ifft(hp, &h)
		const scale = 1.0 / FftLengthBy2
		kernScale(h[:FftLengthBy2], scale, h[:FftLengthBy2])
		clear(h[FftLengthBy2:])
		f.fft.Fft(&h, hp)
	}
	f.partitionWritten(f.partitionToConstrain)
	f.partitionToConstrain = afirNextPartition(f.partitionToConstrain, f.currentSizePartitions)
}

// ScaleFilter multiplies every coefficient by factor.
func (f *AdaptiveFirFilter) ScaleFilter(factor float32) {
	for i := range f.H.data {
		hp := &f.H.data[i]
		for i := range hp.Re {
			hp.Re[i] *= factor
		}
		for i := range hp.Im {
			hp.Im[i] *= factor
		}
	}
	if f.tracked {
		f.negZero.addBelow(f.maxSizePartitions)
		f.h2Stale.addBelow(f.maxSizePartitions)
	}
}

// SetFilter copies the first min(SizePartitions(), numPartitions) partitions
// of H into the filter.
func (f AdaptiveFirFilter) SetFilter(numPartitions int, H FirCoefficients) {
	n := min(f.currentSizePartitions, numPartitions)
	for p := 0; p < n; p++ {
		dst, src := f.H.At(p), H.At(p)
		for ch := 0; ch < f.numRenderChannels; ch++ {
			dst[ch].Re = src[ch].Re
			dst[ch].Im = src[ch].Im
		}
		f.partitionWritten(p)
	}
}
