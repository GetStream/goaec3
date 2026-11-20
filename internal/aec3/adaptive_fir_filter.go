package aec3

// Port of adaptive_fir_filter.{h,cc}.
//
// The aec3:: kernels exist in one scalar and three SIMD lanes. With
// -ffp-contract=off the NEON (vmlaq/vmlsq = separate multiply and add),
// SSE2 and AVX2 ApplyFilter/AdaptPartitions variants compute every bin with
// the same rounded products and the same accumulation order across
// partitions and channels as the scalar code; they only differ in how the
// partition loop is split around the ring-buffer wrap and in processing bin
// kFftLengthBy2 in a second pass. The lane functions below keep that
// structure; the per-block 64-bin work goes through afirApplyBlock /
// afirAdaptBlock / afirFreqRespBlock, which the per-arch files back with
// assembly. ComputeFrequencyResponse_Avx2 is the one lane that differs
// numerically (it uses _mm256_fmadd_ps).

import "math"

// ComputeFrequencyResponse is aec3::ComputeFrequencyResponse (scalar).
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

// ComputeFrequencyResponseNeon is aec3::ComputeFrequencyResponse_Neon.
func ComputeFrequencyResponseNeon(numPartitions int, H FirCoefficients, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptNeon, numPartitions, H, H2)
}

// ComputeFrequencyResponseSse2 is aec3::ComputeFrequencyResponse_Sse2.
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
		// The C++ clears H2 and takes the maximum with every channel; the
		// first channel is the maximum with +0.
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

// afirFreqRespFirstGo is *h2 = 0 followed by afirFreqRespBlock(opt, h, h2):
// the 64-bin frequency-response block of the first channel, whose stored
// maximum is +0. The power of a bin is +0, positive, +Inf or NaN, and the
// maximum of +0 and such a value is the value in the three SIMD lanes
// (vmaxq_f32 returns a NaN operand, _mm_max_ps its second operand unless
// the first is greater), so the block stores the powers. It leaves
// h2[FftLengthBy2] at +0.
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

// afirFreqRespBinsGo is the scalar max-accumulation of |H|^2 over [from, to):
// H2 = std::max(H2, tmp).
func afirFreqRespBinsGo(h *FftData, h2 *[FftLengthBy2Plus1]float32, from, to int) {
	for j := from; j < to; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		if h2[j] < tmp {
			h2[j] = tmp
		}
	}
}

// afirFreqRespBlockNeonGo is the pure-Go reference of the NEON 64-bin
// frequency-response block: H2 = vmaxq_f32(H2, vmlaq_f32(re*re, im, im)).
// vmaxq_f32 is Go's max (a NaN in either operand gives NaN).
func afirFreqRespBlockNeonGo(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		h2[j] = max(h2[j], tmp)
	}
}

// afirFreqRespBlockSse2Go is the pure-Go reference of the SSE2 64-bin
// frequency-response block: H2 = _mm_max_ps(H2, re*re + im*im), which keeps
// H2 only when H2 > tmp (a NaN in either operand gives tmp).
func afirFreqRespBlockSse2Go(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j++ {
		tmp := float32(h.Re[j]*h.Re[j]) + float32(h.Im[j]*h.Im[j])
		h2[j] = sseMax(h2[j], tmp)
	}
}

// afirInfBits is the bit pattern of +Inf. The float32 values whose pattern
// is not above it as an unsigned integer are +0, the positive numbers and
// +Inf, and on them the order of the patterns is the order of the values.
const afirInfBits = 0x7F800000

// afirFreqRespBlockSse2Fast is afirFreqRespBlockSse2Go without a branch on
// the data. With more than one render channel the comparison of the stored
// maximum with a channel's power is as good as random, and a float
// comparison that selects between two floats compiles to a branch.
//
// For a and b that are +0, positive or +Inf, _mm_max_ps(a, b) = a > b ? a : b
// has the larger bit pattern: a > b holds exactly when the pattern of a is
// the larger one, and patterns that are equal are the same value. So the
// maximum of the two patterns as unsigned integers (a conditional move) is
// the result, and it exceeds afirInfBits exactly when a or b is a NaN, a
// negative number or -0. The first group of four bins with such a value
// hands the rest of the block to afirFreqRespBlockSse2From and returns.
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

// AdaptPartitions is aec3::AdaptPartitions (scalar): H(t+1)=H(t)+G(t)*conj(X(t)).
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

// AdaptPartitionsNeon is aec3::AdaptPartitions_Neon.
func AdaptPartitionsNeon(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptNeon, renderBuffer, G, numPartitions, H)
}

// AdaptPartitionsSse2 is aec3::AdaptPartitions_Sse2.
func AdaptPartitionsSse2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	afirAdaptPartitionsSimd(OptSse2, renderBuffer, G, numPartitions, H)
}

func afirAdaptPartitionsSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, G *FftData, numPartitions int, H FirCoefficients) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if n := numPartitions * numRenderChannels; afirHasTermKernels && n <= afirMaxTerms {
		// The pair lists live on the stack and are zeroed per call, so
		// short filters take short lists.
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

// afirAdaptBinsGo is the scalar H += conj(X)*G over bins [from, to).
func afirAdaptBinsGo(x, g, h *FftData, from, to int) {
	for k := from; k < to; k++ {
		h.Re[k] += float32(x.Re[k]*g.Re[k]) + float32(x.Im[k]*g.Im[k])
		h.Im[k] += float32(x.Re[k]*g.Im[k]) - float32(x.Im[k]*g.Re[k])
	}
}

// ApplyFilter is aec3::ApplyFilter (scalar).
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

// ApplyFilterNeon is aec3::ApplyFilter_Neon.
func ApplyFilterNeon(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptNeon, renderBuffer, numPartitions, H, S)
}

// ApplyFilterSse2 is aec3::ApplyFilter_Sse2.
func ApplyFilterSse2(renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	afirApplyFilterSimd(OptSse2, renderBuffer, numPartitions, H, S)
}

func afirApplyFilterSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, numPartitions int, H FirCoefficients, S *FftData) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if n := numPartitions * numRenderChannels; afirHasTermKernels && n <= afirMaxTerms {
		// The pair lists live on the stack and are zeroed per call, so
		// short filters take short lists.
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

// afirMaxTerms bounds the (partition, channel) pairs that the term kernels
// take from stack arrays; larger filters run the per-partition loops.
const afirMaxTerms = 64

// afirFewTerms bounds the short pair lists: the default filters of one
// render channel have 13 partitions.
const afirFewTerms = 16

// afirTerms lists the (X, H) operand pairs of ApplyFilter/AdaptPartitions in
// the C++ accumulation order: partition-major, channel-minor, with the render
// ring index running from the read position and wrapping to 0 exactly as
// the lim1/lim2 loops do. It fills xs and hs, one pair per element of xs.
func afirTerms(data *FftBuffer, position int, H FirCoefficients, xs, hs []*FftData) {
	nc := data.numChannels
	x := data.data
	h := H.data[:len(xs)]
	hs = hs[:len(xs)]
	if H.numChannels != nc {
		panic("aec3: filter and render buffer shapes differ")
	}
	// Both arrays are partition-major, so the pairs are consecutive elements;
	// the render ring wraps to its start.
	xi := position * nc
	for t := range h {
		if xi == len(x) {
			xi = 0
		}
		xs[t], hs[t] = &x[xi], &h[t]
		xi++
	}
}

// afirApplyTermsGo is the term kernel reference: S = sum over the pairs of
// X*H, accumulated per bin in pair order starting from +0 (as S.Clear()
// followed by the per-partition S += X*H).
func afirApplyTermsGo(xs, hs []*FftData, s *FftData) {
	s.Clear()
	hs = hs[:len(xs)]
	for t, x := range xs {
		afirApplyBinsGo(x, hs[t], s, 0, FftLengthBy2Plus1)
	}
}

// afirAdaptTermsGo is the term kernel reference: H += conj(X)*G per pair.
func afirAdaptTermsGo(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	for t, x := range xs {
		afirAdaptBinsGo(x, g, hs[t], 0, FftLengthBy2Plus1)
	}
}

// afirApplyBinsGo is the scalar S += X*H over bins [from, to).
func afirApplyBinsGo(x, h, s *FftData, from, to int) {
	for k := from; k < to; k++ {
		s.Re[k] += float32(x.Re[k]*h.Re[k]) - float32(x.Im[k]*h.Im[k])
		s.Im[k] += float32(x.Re[k]*h.Im[k]) + float32(x.Im[k]*h.Re[k])
	}
}

// afirApplyAllGo is afirApplyBinsGo over all bins. The arrays have one
// length, so the loop runs without bounds checks.
func afirApplyAllGo(x, h, s *FftData) {
	xr, xi, hr, hi := &x.Re, &x.Im, &h.Re, &h.Im
	sr, si := &s.Re, &s.Im
	for k := range sr {
		a, b, c, d := xr[k], xi[k], hr[k], hi[k]
		sr[k] += float32(a*c) - float32(b*d)
		si[k] += float32(a*d) + float32(b*c)
	}
}

// afirAdaptAllGo is afirAdaptBinsGo over all bins, several bins per step.
// The loop is bound by its instruction count. Where the float registers hold
// the operands of eight bins a step loads them before it stores its results,
// which leaves every result as it is (a store to h reaches no operand of
// another bin, whatever h aliases); on amd64 a step is four bins, one after
// the other.
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
		// Bin by bin: the stores of a bin separate its operands from those
		// of the next, so four values are live at a time.
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

// afirAdaptLastBin is bin FftLengthBy2 of afirAdaptBinsGo.
func afirAdaptLastBin(x, g, h *FftData) {
	const n = FftLengthBy2
	a, b, c, d := x.Re[n], x.Im[n], g.Re[n], g.Im[n]
	h.Re[n] += float32(a*c) + float32(b*d)
	h.Im[n] += float32(a*d) - float32(b*c)
}

// afirApplyTermsFastGo is afirApplyTermsGo with the bin loop outside the
// pair loop: the sums of a group of bins stay in locals while all pairs
// accumulate into them, in pair order, starting from +0. The loop is bound
// by its instruction count, so the form follows the registers of the
// architecture.
func afirApplyTermsFastGo(xs, hs []*FftData, s *FftData) {
	if isAMD64 {
		afirApplyTermsQuadGo(xs, hs, s)
	} else {
		afirApplyTermsWideGo(xs, hs, s)
	}
}

// afirApplyTermsWideGo keeps eight bins (sixteen sums) in locals, for
// architectures with thirty-two float registers: the bookkeeping of a pair
// (two pointers, their nil checks, the window addresses) is paid once per
// eight bins.
func afirApplyTermsWideGo(xs, hs []*FftData, s *FftData) {
	afirApplyTermsWide(xs, hs[:len(xs)], s, len(xs))
	afirApplyTermsLastBin(xs, hs, s)
}

// afirApplyTermsWide is afirApplyTermsWideGo for the bins below
// FftLengthBy2. n is len(xs). A pair is two basic blocks of four bins: in
// one block the compiler computes the sixteen terms before it adds any of
// them to its sum, and sixteen sums with sixteen terms exceed the
// registers. The test between the blocks never holds (t < len(xs) = n); the
// function does not inline, so the compiler does not know n and keeps the
// test.
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

// afirApplyTermsQuadGo keeps four bins (eight sums) in locals, for amd64.
// The eight sums and the eight terms of a pair exceed its fifteen float
// registers, so the compiler spills some of them; the loop is bound by its
// loads all the same, and two passes that keep fewer values in registers
// but load every operand twice (the real sums, then the imaginary ones) are
// a third slower.
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

// afirApplyTermsLastBin is bin FftLengthBy2 of afirApplyTermsGo.
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

// afirAdaptTermsFastGo is afirAdaptTermsGo with bounds-check-free loops.
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

// afirMinSizeMinus1 returns std::min(a, n - 1) with size_t semantics
// (n - 1 wraps to SIZE_MAX for n == 0).
func afirMinSizeMinus1(a, n int) int {
	if n == 0 {
		return a
	}
	return min(a, n-1)
}

// afirNextPartition returns a < n-1 ? a+1 : 0 with size_t semantics.
func afirNextPartition(a, n int) int {
	if uint(a) < uint(n)-1 {
		return a + 1
	}
	return 0
}

// FirCoefficients holds the coefficients of a partitioned filter
// ([partition][channel]) partition-major in one pointer-free backing array;
// At returns the channels of one partition.
type FirCoefficients struct {
	numChannels int
	data        []FftData
}

func NewFirCoefficients(numPartitions, numChannels int) FirCoefficients {
	return FirCoefficients{numChannels: numChannels, data: make([]FftData, numPartitions*numChannels)}
}

// At returns the per-channel coefficients of partition p.
func (h FirCoefficients) At(p int) []FftData {
	n := h.numChannels
	return h.data[p*n : (p+1)*n : (p+1)*n]
}

func (h FirCoefficients) NumChannels() int { return h.numChannels }

// NumPartitions returns the number of partitions held.
func (h FirCoefficients) NumPartitions() int {
	if h.numChannels == 0 {
		return 0
	}
	return len(h.data) / h.numChannels
}

// Flat returns the coefficients partition-major, aliasing the storage.
func (h FirCoefficients) Flat() []FftData { return h.data }

// Clone returns a copy with its own storage.
func (h FirCoefficients) Clone() FirCoefficients {
	return FirCoefficients{numChannels: h.numChannels, data: append([]FftData(nil), h.data...)}
}

// AdaptiveFirFilter is a frequency-domain partitioned adaptive filter.
type AdaptiveFirFilter struct {
	*afirState
	H FirCoefficients
}

// afirState is the state of an AdaptiveFirFilter but its coefficients.
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

	// Bookkeeping of the zero-gain skip (adaptAndUpdateSizeKnownRender) and
	// of updateFrequencyResponse, for filters of at most
	// afirTrackedPartitions partitions (tracked).
	//
	// negZero holds every partition that may hold a coefficient equal to
	// -0: a partition outside it holds none. h2Stale holds every partition
	// written since updateFrequencyResponse last computed its response.
	tracked       bool
	skipDisabled  bool // tests: always adapt, always compute every response
	negZero       afirPartitionSet
	h2Stale       afirPartitionSet
	zeroGainSkips uint64 // adaptations skipped, for tests
}

// afirTrackedPartitions bounds the filter sizes the zero-gain skip handles.
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

// init sets up f in place; the coefficients live in storage, which holds
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

// zeroPartitions clears the partitions [from, to): they hold +0 only.
func (f AdaptiveFirFilter) zeroPartitions(from, to int) {
	afirZeroFilter(from, to, f.H)
	if f.tracked {
		for p := from; p < to; p++ {
			f.negZero.remove(p)
			f.h2Stale.add(p)
		}
	}
}

// partitionWritten records that partition p was written with values that
// may include -0.
func (f AdaptiveFirFilter) partitionWritten(p int) {
	if f.tracked {
		f.negZero.add(p)
		f.h2Stale.add(p)
	}
}

// ZeroGainSkipsForTesting returns the number of adaptations skipped because
// the gain was zero.
func (f AdaptiveFirFilter) ZeroGainSkipsForTesting() uint64 { return f.zeroGainSkips }

// SizePartitions returns the current filter size.
func (f AdaptiveFirFilter) SizePartitions() int { return f.currentSizePartitions }

// MaxFilterSizePartitions returns the maximum filter size.
func (f AdaptiveFirFilter) MaxFilterSizePartitions() int { return f.maxSizePartitions }

// GetFilter returns the filter coefficients ([partition][channel]).
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

// Adapt adapts the filter (C++ Adapt(render_buffer, G)).
func (f *AdaptiveFirFilter) Adapt(renderBuffer *RenderBuffer, G *FftData) {
	f.adaptAndUpdateSize(renderBuffer, G)
	f.constrain()
}

// AdaptWithImpulseResponse adapts the filter and updates the externally
// stored impulse response estimate (C++ Adapt(render_buffer, G,
// impulse_response)). *impulseResponse must have capacity
// GetTimeDomainLength(MaxFilterSizePartitions()); it is resized like the
// std::vector (new elements are zero).
func (f *AdaptiveFirFilter) AdaptWithImpulseResponse(renderBuffer *RenderBuffer, G *FftData, impulseResponse *[]float32) {
	f.adaptAndUpdateSize(renderBuffer, G)
	f.constrainAndUpdateImpulseResponse(impulseResponse)
}

// ComputeFrequencyResponse resizes *H2 to SizePartitions() entries (cap must
// be at least MaxFilterSizePartitions()) and fills it.
func (f AdaptiveFirFilter) ComputeFrequencyResponse(H2 *[][FftLengthBy2Plus1]float32) {
	afirResize(H2, f.currentSizePartitions)
	f.frequencyResponse(f.currentSizePartitions, f.H, *H2)
}

// frequencyResponse fills H2 with the response of the first numPartitions
// partitions of H in the filter's lane.
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

// updateFrequencyResponse is ComputeFrequencyResponse for a response array
// that this method alone writes, call after call: the response of a
// partition depends on that partition only, so the partitions that were not
// written since the last call keep their response.
//
// h2Stale covers every write: adaptation that ran (all partitions),
// constraining, scaling, SetFilter, and clearing. A partition that enters
// the filter when it grows is cleared, so the element afirResize zeroes for
// it is stale as well.
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

// afirResize mirrors std::vector::resize: grown elements are zero.
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
// (RenderBuffer.SpectralSum; nil when it has none). It skips the adaptation
// when that provably leaves every coefficient as it is:
//
//   - Every lane adapts a coefficient h with products of render and gain
//     values that it adds to or subtracts from h (fused or not). With a
//     gain that is +0 or -0 everywhere and finite render values, every such
//     product is a zero, and so is every sum or difference of them.
//   - h plus or minus a zero is h, bit for bit, for every h except -0
//     (-0 + +0 = +0). A NaN h keeps its payload: coefficients come out of
//     arithmetic, so they are quiet NaNs.
//   - The render values are finite when X2 is: a spectrum bin is
//     re*re + im*im (or the fused form), which is an infinity or a NaN when
//     re or im is, and the sum of such non-negative terms then stays one.
//     X2 covers the partitions of the adaptation when the size after
//     updateSize is at most x2Partitions and the spectrum ring holds the
//     spectra of the FFT ring (RenderBuffer.SpectraMatchFfts).
//   - Additions never produce -0 from an h that is not -0, so a partition
//     gets a -0 only where it is written otherwise; those writes put it in
//     negZero. The partitions in negZero that do hold a -0 are adapted as
//     usual.
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
// takes the ones that hold none out of negZero.
func (f AdaptiveFirFilter) adaptNegZeroPartitions(renderBuffer *RenderBuffer, G *FftData, n int) {
	data := renderBuffer.GetFftBuffer()
	for p := 0; p < n; p++ {
		if !f.negZero.has(p) {
			continue
		}
		h := f.H.At(p)
		if afirHoldsNegZero(h) {
			// Partition p meets the render FFT p slots after the read
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
	// The partition of the impulse response is zero-filled and then takes
	// the first channel as is.
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

// afirConstrainPair is refined.constrainAndUpdateImpulseResponse(
// impulseResponse) followed by coarse.constrain(), with the transforms of
// the two filters run as pairs. The filters have the same number of render
// channels and share no state.
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

// afirMaxAbsUpdateGo replaces ir[k] by h[k] where fabsf(ir[k]) < fabsf(h[k])
// (false when either is a NaN).
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

// ScaleFilter scales all filter coefficients by factor.
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
