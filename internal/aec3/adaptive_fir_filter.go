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

// ComputeFrequencyResponse is aec3::ComputeFrequencyResponse (scalar).
func ComputeFrequencyResponse(numPartitions int, H [][]FftData, H2 [][FftLengthBy2Plus1]float32) {
	for i := range H2 {
		H2[i] = [FftLengthBy2Plus1]float32{}
	}
	numRenderChannels := len(H[0])
	for p := 0; p < numPartitions; p++ {
		h2 := &H2[p]
		for ch := 0; ch < numRenderChannels; ch++ {
			afirFreqRespBinsGo(&H[p][ch], h2, 0, FftLengthBy2Plus1)
		}
	}
}

// ComputeFrequencyResponseNeon is aec3::ComputeFrequencyResponse_Neon.
func ComputeFrequencyResponseNeon(numPartitions int, H [][]FftData, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptNeon, numPartitions, H, H2)
}

// ComputeFrequencyResponseSse2 is aec3::ComputeFrequencyResponse_Sse2.
func ComputeFrequencyResponseSse2(numPartitions int, H [][]FftData, H2 [][FftLengthBy2Plus1]float32) {
	afirComputeFrequencyResponseSimd(OptSse2, numPartitions, H, H2)
}

func afirComputeFrequencyResponseSimd(opt Aec3Optimization, numPartitions int, H [][]FftData, H2 [][FftLengthBy2Plus1]float32) {
	if opt == OptNeon && afirHasNeonFreqResp {
		for p := 0; p < numPartitions; p++ {
			afirFreqRespPartitionNEON(H[p], &H2[p])
		}
		for i := numPartitions; i < len(H2); i++ {
			H2[i] = [FftLengthBy2Plus1]float32{}
		}
		return
	}
	for i := range H2 {
		H2[i] = [FftLengthBy2Plus1]float32{}
	}
	numRenderChannels := len(H[0])
	for p := 0; p < numPartitions; p++ {
		h2 := &H2[p]
		for ch := 0; ch < numRenderChannels; ch++ {
			hp := &H[p][ch]
			afirFreqRespBlock(opt, hp, h2)
			afirFreqRespBinsGo(hp, h2, FftLengthBy2, FftLengthBy2Plus1)
		}
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

// AdaptPartitions is aec3::AdaptPartitions (scalar): H(t+1)=H(t)+G(t)*conj(X(t)).
func AdaptPartitions(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H [][]FftData) {
	data := renderBuffer.GetFftBuffer()
	index := renderBuffer.Position()
	numRenderChannels := data.NumChannels()
	for p := 0; p < numPartitions; p++ {
		x := data.At(index)
		for ch := 0; ch < numRenderChannels; ch++ {
			afirAdaptAllGo(&x[ch], G, &H[p][ch])
		}
		if index < data.Size-1 {
			index++
		} else {
			index = 0
		}
	}
}

// AdaptPartitionsNeon is aec3::AdaptPartitions_Neon.
func AdaptPartitionsNeon(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H [][]FftData) {
	afirAdaptPartitionsSimd(OptNeon, renderBuffer, G, numPartitions, H)
}

// AdaptPartitionsSse2 is aec3::AdaptPartitions_Sse2.
func AdaptPartitionsSse2(renderBuffer *RenderBuffer, G *FftData, numPartitions int, H [][]FftData) {
	afirAdaptPartitionsSimd(OptSse2, renderBuffer, G, numPartitions, H)
}

func afirAdaptPartitionsSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, G *FftData, numPartitions int, H [][]FftData) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if afirHasTermKernels && numPartitions*numRenderChannels <= afirMaxTerms {
		var xs, hs [afirMaxTerms]*FftData
		n := afirTerms(data, renderBuffer.Position(), numPartitions, H, &xs, &hs)
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
				afirAdaptBlock(opt, &data.At(xPartition)[ch], G, &H[p][ch])
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
				afirAdaptBinsGo(&data.At(xPartition)[ch], G, &H[p][ch], FftLengthBy2, FftLengthBy2Plus1)
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
func ApplyFilter(renderBuffer *RenderBuffer, numPartitions int, H [][]FftData, S *FftData) {
	S.Re = [FftLengthBy2Plus1]float32{}
	S.Im = [FftLengthBy2Plus1]float32{}
	data := renderBuffer.GetFftBuffer()
	index := renderBuffer.Position()
	numRenderChannels := data.NumChannels()
	for p := 0; p < numPartitions; p++ {
		x := data.At(index)
		for ch := 0; ch < numRenderChannels; ch++ {
			afirApplyAllGo(&x[ch], &H[p][ch], S)
		}
		if index < data.Size-1 {
			index++
		} else {
			index = 0
		}
	}
}

// ApplyFilterNeon is aec3::ApplyFilter_Neon.
func ApplyFilterNeon(renderBuffer *RenderBuffer, numPartitions int, H [][]FftData, S *FftData) {
	afirApplyFilterSimd(OptNeon, renderBuffer, numPartitions, H, S)
}

// ApplyFilterSse2 is aec3::ApplyFilter_Sse2.
func ApplyFilterSse2(renderBuffer *RenderBuffer, numPartitions int, H [][]FftData, S *FftData) {
	afirApplyFilterSimd(OptSse2, renderBuffer, numPartitions, H, S)
}

func afirApplyFilterSimd(opt Aec3Optimization, renderBuffer *RenderBuffer, numPartitions int, H [][]FftData, S *FftData) {
	data := renderBuffer.GetFftBuffer()
	numRenderChannels := data.NumChannels()
	if afirHasTermKernels && numPartitions*numRenderChannels <= afirMaxTerms {
		var xs, hs [afirMaxTerms]*FftData
		n := afirTerms(data, renderBuffer.Position(), numPartitions, H, &xs, &hs)
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
				afirApplyBlock(opt, &data.At(xPartition)[ch], &H[p][ch], S)
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
				afirApplyBinsGo(&data.At(xPartition)[ch], &H[p][ch], S, FftLengthBy2, FftLengthBy2Plus1)
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

// afirTerms lists the (X, H) operand pairs of ApplyFilter/AdaptPartitions in
// the C++ accumulation order: partition-major, channel-minor, with the render
// ring index running from the read position and wrapping to 0 exactly as
// the lim1/lim2 loops do. It returns the number of pairs.
func afirTerms(data *FftBuffer, position, numPartitions int, H [][]FftData, xs, hs *[afirMaxTerms]*FftData) int {
	n := 0
	xPartition := position
	for p := 0; p < numPartitions; p++ {
		if xPartition == data.Size {
			xPartition = 0
		}
		xp, hp := data.At(xPartition), H[p]
		hp = hp[:len(xp)]
		for ch := range xp {
			xs[n], hs[n] = &xp[ch], &hp[ch]
			n++
		}
		xPartition++
	}
	return n
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

// afirAdaptAllGo is afirAdaptBinsGo over all bins.
func afirAdaptAllGo(x, g, h *FftData) {
	xr, xi, gr, gi := &x.Re, &x.Im, &g.Re, &g.Im
	hr, hi := &h.Re, &h.Im
	for k := range hr {
		a, b, c, d := xr[k], xi[k], gr[k], gi[k]
		hr[k] += float32(a*c) + float32(b*d)
		hi[k] += float32(a*d) - float32(b*c)
	}
}

// afirApplyTermsFastGo is afirApplyTermsGo with the bin loop outside the
// pair loop: four bins of S stay in locals while all pairs accumulate into
// them, in pair order, starting from +0.
func afirApplyTermsFastGo(xs, hs []*FftData, s *FftData) {
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
func afirZeroFilter(oldSize, newSize int, H [][]FftData) {
	for p := oldSize; p < newSize; p++ {
		for ch := range H[0] {
			H[p][ch].Clear()
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

// AdaptiveFirFilter is a frequency-domain partitioned adaptive filter.
type AdaptiveFirFilter struct {
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
	H                             [][]FftData
	partitionToConstrain          int
}

func NewAdaptiveFirFilter(maxSizePartitions, initialSizePartitions, sizeChangeDurationBlocks, numRenderChannels int, optimization Aec3Optimization) *AdaptiveFirFilter {
	f := &AdaptiveFirFilter{
		optimization:             optimization,
		numRenderChannels:        numRenderChannels,
		maxSizePartitions:        maxSizePartitions,
		sizeChangeDurationBlocks: sizeChangeDurationBlocks,
		currentSizePartitions:    initialSizePartitions,
		targetSizePartitions:     initialSizePartitions,
		oldTargetSizePartitions:  initialSizePartitions,
		H:                        make([][]FftData, maxSizePartitions),
	}
	h := make([]FftData, maxSizePartitions*numRenderChannels)
	for p := range f.H {
		f.H[p] = h[p*numRenderChannels : (p+1)*numRenderChannels : (p+1)*numRenderChannels]
	}
	f.oneBySizeChangeDurationBlocks = 1 / float32(f.sizeChangeDurationBlocks)
	afirZeroFilter(0, f.maxSizePartitions, f.H)
	f.SetSizePartitions(f.currentSizePartitions, true)
	return f
}

// HandleEchoPathChange clears the partitions beyond the current size.
func (f *AdaptiveFirFilter) HandleEchoPathChange() {
	afirZeroFilter(f.currentSizePartitions, f.maxSizePartitions, f.H)
}

// SizePartitions returns the current filter size.
func (f *AdaptiveFirFilter) SizePartitions() int { return f.currentSizePartitions }

// MaxFilterSizePartitions returns the maximum filter size.
func (f *AdaptiveFirFilter) MaxFilterSizePartitions() int { return f.maxSizePartitions }

// GetFilter returns the filter coefficients ([partition][channel]).
func (f *AdaptiveFirFilter) GetFilter() [][]FftData { return f.H }

// SetSizePartitions sets the target filter size, immediately or gradually.
func (f *AdaptiveFirFilter) SetSizePartitions(size int, immediateEffect bool) {
	f.targetSizePartitions = min(f.maxSizePartitions, size)
	if immediateEffect {
		oldSizePartitions := f.currentSizePartitions
		f.currentSizePartitions = f.targetSizePartitions
		f.oldTargetSizePartitions = f.targetSizePartitions
		afirZeroFilter(oldSizePartitions, f.currentSizePartitions, f.H)
		f.partitionToConstrain = afirMinSizeMinus1(f.partitionToConstrain, f.currentSizePartitions)
		f.sizeChangeCounter = 0
	} else {
		f.sizeChangeCounter = f.sizeChangeDurationBlocks
	}
}

func afirAverage(from, to, fromWeight float32) float32 {
	return float32(from*fromWeight) + float32(to*(1-fromWeight))
}

func (f *AdaptiveFirFilter) updateSize() {
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
	afirZeroFilter(oldSizePartitions, f.currentSizePartitions, f.H)
}

// Filter produces the filter output S.
func (f *AdaptiveFirFilter) Filter(renderBuffer *RenderBuffer, S *FftData) {
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
func (f *AdaptiveFirFilter) ComputeFrequencyResponse(H2 *[][FftLengthBy2Plus1]float32) {
	afirResize(H2, f.currentSizePartitions)
	switch f.optimization {
	case OptSse2:
		ComputeFrequencyResponseSse2(f.currentSizePartitions, f.H, *H2)
	case OptAvx2:
		ComputeFrequencyResponseAvx2(f.currentSizePartitions, f.H, *H2)
	case OptNeon:
		ComputeFrequencyResponseNeon(f.currentSizePartitions, f.H, *H2)
	default:
		ComputeFrequencyResponse(f.currentSizePartitions, f.H, *H2)
	}
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

func (f *AdaptiveFirFilter) adaptAndUpdateSize(renderBuffer *RenderBuffer, G *FftData) {
	f.updateSize()
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
		hp := &f.H[f.partitionToConstrain][ch]
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
	f.partitionToConstrain = afirNextPartition(f.partitionToConstrain, f.currentSizePartitions)
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
		hp := &f.H[f.partitionToConstrain][ch]
		f.fft.Ifft(hp, &h)
		const scale = 1.0 / FftLengthBy2
		kernScale(h[:FftLengthBy2], scale, h[:FftLengthBy2])
		clear(h[FftLengthBy2:])
		f.fft.Fft(&h, hp)
	}
	f.partitionToConstrain = afirNextPartition(f.partitionToConstrain, f.currentSizePartitions)
}

// ScaleFilter scales all filter coefficients by factor.
func (f *AdaptiveFirFilter) ScaleFilter(factor float32) {
	for p := range f.H {
		for ch := range f.H[p] {
			hp := &f.H[p][ch]
			for i := range hp.Re {
				hp.Re[i] *= factor
			}
			for i := range hp.Im {
				hp.Im[i] *= factor
			}
		}
	}
}

// SetFilter copies the first min(SizePartitions(), numPartitions) partitions
// of H into the filter.
func (f *AdaptiveFirFilter) SetFilter(numPartitions int, H [][]FftData) {
	n := min(f.currentSizePartitions, numPartitions)
	for p := 0; p < n; p++ {
		for ch := 0; ch < f.numRenderChannels; ch++ {
			f.H[p][ch].Re = H[p][ch].Re
			f.H[p][ch].Im = H[p][ch].Im
		}
	}
}
