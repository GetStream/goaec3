package aec3

// Subsample rate used for computing the accumulated error. The implementation
// of some core functions depends on this constant being equal to 4.
const dlyAccumulatedErrorSubSampleRate = 4

func dlyUpdateAccumulatedError(instantaneousAccumulatedError, accumulatedError []float32, oneOverErrorSumAnchor float32) {
	const smoothConstantIncreases = 0.015
	accumulatedError = accumulatedError[:len(instantaneousAccumulatedError)]
	for k, v := range instantaneousAccumulatedError {
		errorNorm := float32(v * oneOverErrorSumAnchor)
		if errorNorm < accumulatedError[k] {
			accumulatedError[k] = errorNorm
		} else {
			accumulatedError[k] += float32(smoothConstantIncreases * (errorNorm - accumulatedError[k]))
		}
	}
}

func dlyComputePreEchoLag(accumulatedError []float32, lag, alignmentShiftWinner int) int {
	const preEchoThreshold = 0.5
	preEchoLagEstimate := lag - alignmentShiftWinner
	maximumPreEchoLag := min(preEchoLagEstimate/dlyAccumulatedErrorSubSampleRate, len(accumulatedError))
	for k := maximumPreEchoLag - 1; k >= 0; k-- {
		if accumulatedError[k] > preEchoThreshold {
			break
		}
		preEchoLagEstimate = (k+1)*dlyAccumulatedErrorSubSampleRate - 1
	}
	return preEchoLagEstimate + alignmentShiftWinner
}

func dlySaturated(y float32) bool { return y >= 32000 || y <= -32000 }

// MatchedFilterCoreWithAccumulatedErrorNeon mirrors
// aec3::MatchedFilterCoreWithAccumulatedError_NEON.
func MatchedFilterCoreWithAccumulatedErrorNeon(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var x2v [4]float32
		s := dlyAccErrNeon(h, xp, y[i], accumulatedError, &x2v)
		var x2Sum float32
		x2Sum += (x2v[0] + x2v[1]) + (x2v[2] + x2v[3])
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:hSize&^3], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCoreNeon mirrors aec3::MatchedFilterCore_NEON.
func MatchedFilterCoreNeon(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreWithAccumulatedErrorNeon(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreFused(4, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreAccumulatedErrorSse2 mirrors
// aec3::MatchedFilterCore_AccumulatedError_SSE2.
func MatchedFilterCoreAccumulatedErrorSse2(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var x2v [8]float32
		s := dlyAccErrSse2(h, xp, y[i], accumulatedError, &x2v)
		var v [4]float32
		for l := range v {
			v[l] = x2v[l] + x2v[4+l]
		}
		var x2Sum float32
		x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:hSize&^3], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCoreSse2 mirrors aec3::MatchedFilterCore_SSE2.
func MatchedFilterCoreSse2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreAccumulatedErrorSse2(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreFused(8, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreAccumulatedErrorAvx2 mirrors
// aec3::MatchedFilterCore_AccumulatedError_AVX2.
func MatchedFilterCoreAccumulatedErrorAvx2(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var x2v [16]float32
		s := dlyAccErrAvx2(h, xp, y[i], accumulatedError, &x2v)
		var a [8]float32
		for l := range a {
			a[l] = x2v[l] + x2v[8+l]
		}
		var v [4]float32
		for l := range v {
			v[l] = a[l] + a[4+l]
		}
		var x2Sum float32
		x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpyFma(h[:hSize&^7], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCoreAvx2 mirrors aec3::MatchedFilterCore_AVX2.
func MatchedFilterCoreAvx2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreAccumulatedErrorAvx2(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreFused(16, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// dlyReduceNeon adds the NEON lane sums to the scalar tail sums.
func dlyReduceNeon(acc *[8]float32, s, x2Sum float32) (float32, float32) {
	s += (acc[0] + acc[1]) + (acc[2] + acc[3])
	x2Sum += (acc[4] + acc[5]) + (acc[6] + acc[7])
	return s, x2Sum
}

// dlyReduceSse2 adds the SSE2 lane sums to the scalar tail sums.
func dlyReduceSse2(acc *[16]float32, s, x2Sum float32) (float32, float32) {
	var v [4]float32
	for l := range v {
		v[l] = acc[8+l] + acc[12+l]
	}
	x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
	for l := range v {
		v[l] = acc[l] + acc[4+l]
	}
	s += ((v[0] + v[1]) + v[2]) + v[3]
	return s, x2Sum
}

// dlyReduceAvx2 adds the AVX2 lane sums to the scalar tail sums:
// hsum_ab(x2_sum_256 + x2_sum_256_8, s_256 + s_256_8).
func dlyReduceAvx2(acc *[32]float32, s, x2Sum float32) (float32, float32) {
	var a, b [8]float32
	for l := range a {
		a[l] = acc[16+l] + acc[24+l]
		b[l] = acc[l] + acc[8+l]
	}
	x2Sum += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
	s += ((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7]))
	return s, x2Sum
}

// dlyAxpyRun is a stretch [start, end) of the filter whose update reads
// x[k+off] and uses a fused (fma) or unfused multiply-add.
type dlyAxpyRun struct {
	start, end, off int
	fma             bool
}

// dlyPendingAxpy is the filter update of one capture sample, h[k] +=
// alpha*x[(xStartIndex+k) mod len(x)], split into the stretches the C++ lane
// loop uses. The next sample's dot product applies it to each element just
// before reading that element; every element sees the same operations as
// with a separate update pass.
type dlyPendingAxpy struct {
	runs   [4]dlyAxpyRun
	n      int
	alpha  float32
	active bool
}

func (p *dlyPendingAxpy) set(xStartIndex, chunk1, hSize int, alpha float32, fma bool) {
	p.n = 0
	p.alpha = alpha
	p.active = true
	add := func(start, end, off int, fma bool) {
		if start < end {
			p.runs[p.n] = dlyAxpyRun{start, end, off, fma}
			p.n++
		}
	}
	if fma {
		// AVX2: fused over the multiple-of-8 part of each chunk, unfused tail.
		chunk2 := hSize - chunk1
		add(0, chunk1&^7, xStartIndex, true)
		add(chunk1&^7, chunk1, xStartIndex, false)
		add(chunk1, chunk1+chunk2&^7, -chunk1, true)
		add(chunk1+chunk2&^7, hSize, -chunk1, false)
	} else {
		add(0, chunk1, xStartIndex, false)
		add(chunk1, hSize, -chunk1, false)
	}
}

// apply runs the update on h[a:b].
func (p *dlyPendingAxpy) apply(h, x []float32, a, b int) {
	for i := 0; i < p.n; i++ {
		r := &p.runs[i]
		lo, hi := max(a, r.start), min(b, r.end)
		if lo >= hi {
			continue
		}
		if r.fma {
			dlyAxpyFma(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		} else {
			dlyAxpy(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		}
	}
}

// dlyLaneDot runs the dot kernel of the lane with group size g on the lane's
// accumulators, which occupy the front of acc.
func dlyLaneDot(g int, h, x []float32, acc *[32]float32) {
	switch g {
	case 4:
		dlyDotNeon(h, x, (*[8]float32)(acc[:8]))
	case 8:
		dlyDotSse2(h, x, (*[16]float32)(acc[:16]))
	default:
		dlyDotAvx2(h, x, acc)
	}
}

// dlyDotSegment accumulates h[p0:q]·x[p0+off:q+off] into the lane
// accumulators in groups of g, first applying the pending update to every
// group. Stretches of whole groups inside one update run of the lane's main
// update kind go through the fused update-and-dot kernel.
func dlyDotSegment(g int, pend *dlyPendingAxpy, h, x []float32, p0, q, off int, acc *[32]float32) {
	if !pend.active {
		dlyLaneDot(g, h[p0:q], x[p0+off:q+off], acc)
		return
	}
	fusedFma := g == 16
	k := p0
	for k < q {
		r := &pend.runs[0]
		for i := 1; i < pend.n && pend.runs[i].start <= k; i++ {
			r = &pend.runs[i]
		}
		n := (min(r.end, q) - k) &^ (g - 1)
		if n > 0 {
			if r.fma == fusedFma {
				hk, xa, xb := h[k:k+n], x[k+r.off:k+n+r.off], x[k+off:k+n+off]
				switch g {
				case 4:
					dlyAxpyDotNeon(hk, xa, xb, pend.alpha, (*[8]float32)(acc[:8]))
				case 8:
					dlyAxpyDotSse2(hk, xa, xb, pend.alpha, (*[16]float32)(acc[:16]))
				default:
					dlyAxpyFmaDotAvx2(hk, xa, xb, pend.alpha, acc)
				}
			} else {
				pend.apply(h, x, k, k+n)
				dlyLaneDot(g, h[k:k+n], x[k+off:k+n+off], acc)
			}
			k += n
			continue
		}
		// The group straddles two update runs.
		pend.apply(h, x, k, k+g)
		dlyLaneDot(g, h[k:k+g], x[k+off:k+g+off], acc)
		k += g
	}
}

// dlyCoreFused runs the MatchedFilterCore loop of the SIMD lane with group
// size g (4 NEON, 8 SSE2, 16 AVX2). Per capture sample the lane accumulates
// the multiple-of-g part of each of the two circular-buffer chunks in its
// vector accumulators and the rest in scalars, then reduces. The filter
// update of a sample is deferred into the next sample's dot product (or a
// final pass), which updates each h element right before reading it.
func dlyCoreFused(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	hSize := len(h)
	xSize := len(x)
	var pend dlyPendingAxpy
	for i := range y {
		var acc [32]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1

		v1 := chunk1 &^ (g - 1)
		dlyDotSegment(g, &pend, h, x, 0, v1, xStartIndex, &acc)
		if pend.active {
			pend.apply(h, x, v1, chunk1)
		}
		for k := v1; k < chunk1; k++ {
			xk := x[xStartIndex+k]
			x2Sum += float32(xk * xk)
			s += float32(h[k] * xk)
		}
		v2 := chunk2 &^ (g - 1)
		dlyDotSegment(g, &pend, h, x, chunk1, chunk1+v2, -chunk1, &acc)
		if pend.active {
			pend.apply(h, x, chunk1+v2, hSize)
		}
		for k := chunk1 + v2; k < hSize; k++ {
			xk := x[k-chunk1]
			x2Sum += float32(xk * xk)
			s += float32(h[k] * xk)
		}
		switch g {
		case 4:
			s, x2Sum = dlyReduceNeon((*[8]float32)(acc[:8]), s, x2Sum)
		case 8:
			s, x2Sum = dlyReduceSse2((*[16]float32)(acc[:16]), s, x2Sum)
		default:
			s, x2Sum = dlyReduceAvx2(&acc, s, x2Sum)
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		pend.active = false
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			pend.set(xStartIndex, chunk1, hSize, alpha, g == 16)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
	if pend.active {
		pend.apply(h, x, 0, hSize)
	}
}

// MatchedFilterCore mirrors the scalar aec3::MatchedFilterCore.
func MatchedFilterCore(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError []float32) {
	if computeAccumulatedError {
		clear(accumulatedError)
	}
	xSize := len(x)
	for i := range y {
		var x2Sum, s float32
		xIndex := xStartIndex
		if computeAccumulatedError {
			for k := range h {
				x2Sum += float32(x[xIndex] * x[xIndex])
				s += float32(h[k] * x[xIndex])
				if xIndex < xSize-1 {
					xIndex++
				} else {
					xIndex = 0
				}
				if (k+1)&0b11 == 0 {
					idx := k >> 2
					accumulatedError[idx] += float32((y[i] - s) * (y[i] - s))
				}
			}
		} else {
			for k := range h {
				x2Sum += float32(x[xIndex] * x[xIndex])
				s += float32(h[k] * x[xIndex])
				if xIndex < xSize-1 {
					xIndex++
				} else {
					xIndex = 0
				}
			}
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			xIndex2 := xStartIndex
			for k := range h {
				h[k] += float32(alpha * x[xIndex2])
				if xIndex2 < xSize-1 {
					xIndex2++
				} else {
					xIndex2 = 0
				}
			}
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MaxSquarePeakIndex finds the largest peak of squared values in h.
func MaxSquarePeakIndex(h []float32) int { return dlyMaxSquarePeakIndex(h) }

// dlyMaxSquarePeakIndexRef is the scalar MaxSquarePeakIndex of the C++: it
// tracks the even and odd indices separately with strict comparisons, so
// each keeps its earliest maximum.
func dlyMaxSquarePeakIndexRef(h []float32) int {
	if len(h) < 2 {
		return 0
	}
	maxElement1 := h[0] * h[0]
	maxElement2 := h[1] * h[1]
	lagEstimate1 := 0
	lagEstimate2 := 1
	lastIndex := len(h) - 1
	// Keeping track of even & odd max elements separately typically allows the
	// compiler to produce more efficient code.
	for k := 2; k < lastIndex; k += 2 {
		element1 := h[k] * h[k]
		element2 := h[k+1] * h[k+1]
		if element1 > maxElement1 {
			maxElement1 = element1
			lagEstimate1 = k
		}
		if element2 > maxElement2 {
			maxElement2 = element2
			lagEstimate2 = k + 1
		}
	}
	if maxElement2 > maxElement1 {
		maxElement1 = maxElement2
		lagEstimate1 = lagEstimate2
	}
	// In case of odd h size, we have not yet checked the last element.
	lastElement := h[lastIndex] * h[lastIndex]
	if lastElement > maxElement1 {
		return lastIndex
	}
	return lagEstimate1
}

// MatchedFilterLagEstimate mirrors MatchedFilter::LagEstimate.
type MatchedFilterLagEstimate struct {
	Lag        int
	PreEchoLag int
}

// MatchedFilter produces recursively updated cross-correlation estimates for
// several signal shifts where the intra-shift spacing is uniform
// (matched_filter.h).
type MatchedFilter struct {
	optimization                  Aec3Optimization
	subBlockSize                  int
	filterIntraLagShift           int
	filters                       [][]float32
	accumulatedError              [][]float32
	instantaneousAccumulatedError []float32
	scratchMemory                 []float32
	reportedLagEstimate           MatchedFilterLagEstimate
	hasReportedLagEstimate        bool
	winnerLag                     int
	hasWinnerLag                  bool
	lastDetectedBestLagFilter     int
	numberPreEchoUpdates          int
	excitationLimit               float32
	smoothingFast                 float32
	smoothingSlow                 float32
	matchingFilterThreshold       float32
	detectPreEcho                 bool
}

// NewMatchedFilter mirrors the MatchedFilter constructor (without the data
// dumper).
func NewMatchedFilter(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) *MatchedFilter {
	m := &MatchedFilter{}
	m.init(optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters, alignmentShiftSubBlocks,
		excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold, detectPreEcho)
	return m
}

func (m *MatchedFilter) init(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) {
	if windowSizeSubBlocks <= 0 || BlockSize%subBlockSize != 0 || subBlockSize%4 != 0 {
		panic("aec3: invalid MatchedFilter dimensions")
	}
	*m = MatchedFilter{
		optimization:              optimization,
		subBlockSize:              subBlockSize,
		filterIntraLagShift:       alignmentShiftSubBlocks * subBlockSize,
		filters:                   make([][]float32, numMatchedFilters),
		lastDetectedBestLagFilter: -1,
		excitationLimit:           excitationLimit,
		smoothingFast:             smoothingFast,
		smoothingSlow:             smoothingSlow,
		matchingFilterThreshold:   matchingFilterThreshold,
		detectPreEcho:             detectPreEcho,
	}
	filterSize := windowSizeSubBlocks * subBlockSize
	for n := range m.filters {
		m.filters[n] = make([]float32, filterSize)
	}
	if detectPreEcho {
		m.accumulatedError = make([][]float32, numMatchedFilters)
		for n := range m.accumulatedError {
			e := make([]float32, filterSize/dlyAccumulatedErrorSubSampleRate)
			for k := range e {
				e[k] = 1
			}
			m.accumulatedError[n] = e
		}
		m.instantaneousAccumulatedError = make([]float32, filterSize/dlyAccumulatedErrorSubSampleRate)
		m.scratchMemory = make([]float32, filterSize)
	}
}

// Reset resets the matched filter.
func (m *MatchedFilter) Reset(fullReset bool) {
	for _, f := range m.filters {
		clear(f)
	}
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	if fullReset {
		for _, e := range m.accumulatedError {
			for k := range e {
				e[k] = 1
			}
		}
		m.numberPreEchoUpdates = 0
	}
}

// GetBestLagEstimate returns the current lag estimate, if any.
func (m *MatchedFilter) GetBestLagEstimate() (MatchedFilterLagEstimate, bool) {
	if !m.hasReportedLagEstimate {
		return MatchedFilterLagEstimate{}, false
	}
	return m.reportedLagEstimate, true
}

// GetMaxFilterLag returns the maximum filter lag.
func (m *MatchedFilter) GetMaxFilterLag() int {
	return len(m.filters)*m.filterIntraLagShift + len(m.filters[0])
}

func (m *MatchedFilter) runCore(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computePreEcho bool) {
	switch m.optimization {
	case OptSse2:
		MatchedFilterCoreSse2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	case OptAvx2:
		MatchedFilterCoreAvx2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	case OptNeon:
		MatchedFilterCoreNeon(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	default:
		MatchedFilterCore(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError)
	}
}

// Update updates the correlation with the values in the capture buffer.
func (m *MatchedFilter) Update(renderBuffer *DownsampledRenderBuffer, capture []float32, useSlowSmoothing bool) {
	y := capture[:m.subBlockSize]

	smoothing := m.smoothingFast
	if useSlowSmoothing {
		smoothing = m.smoothingSlow
	}

	x2SumThreshold := float32(float32(len(m.filters[0]))*m.excitationLimit) * m.excitationLimit

	// Compute anchor for the matched filter error.
	var errorSumAnchor float32
	for _, v := range y {
		errorSumAnchor += float32(v * v)
	}

	// Apply all matched filters.
	winnerErrorSum := errorSumAnchor
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	alignmentShift := 0
	previousLagEstimate, hasPreviousLagEstimate := 0, false
	winnerIndex := -1
	for n := range m.filters {
		var errorSum float32
		filtersUpdated := false
		computePreEcho := m.detectPreEcho && n == m.lastDetectedBestLagFilter

		xStartIndex := (renderBuffer.Read + alignmentShift + m.subBlockSize - 1) % len(renderBuffer.Buffer)

		m.runCore(xStartIndex, x2SumThreshold, smoothing, renderBuffer.Buffer, y, m.filters[n],
			&filtersUpdated, &errorSum, computePreEcho)

		// Estimate the lag in the matched filter as the distance to the portion
		// in the filter that contributes the most to the matched filter output.
		// This is detected as the peak of the matched filter.
		lagEstimate := MaxSquarePeakIndex(m.filters[n])
		reliable := lagEstimate > 2 && lagEstimate < len(m.filters[n])-10 &&
			errorSum < float32(m.matchingFilterThreshold*errorSumAnchor)

		// Find the best estimate.
		lag := lagEstimate + alignmentShift
		if filtersUpdated && reliable && errorSum < winnerErrorSum {
			winnerErrorSum = errorSum
			winnerIndex = n
			// In case that 2 matched filters return the same winner candidate
			// (overlap region), the one with the smaller index is chosen in
			// order to search for pre-echoes.
			if hasPreviousLagEstimate && previousLagEstimate == lag {
				m.winnerLag = previousLagEstimate
				winnerIndex = n - 1
			} else {
				m.winnerLag = lag
			}
			m.hasWinnerLag = true
		}
		previousLagEstimate, hasPreviousLagEstimate = lag, true
		alignmentShift += m.filterIntraLagShift
	}

	if winnerIndex != -1 {
		m.reportedLagEstimate = MatchedFilterLagEstimate{Lag: m.winnerLag, PreEchoLag: m.winnerLag}
		m.hasReportedLagEstimate = true
		if m.detectPreEcho && m.lastDetectedBestLagFilter == winnerIndex {
			const energyThreshold = 1.0
			if errorSumAnchor > energyThreshold {
				dlyUpdateAccumulatedError(m.instantaneousAccumulatedError, m.accumulatedError[winnerIndex],
					1/errorSumAnchor)
				m.numberPreEchoUpdates++
			}
			if m.numberPreEchoUpdates >= 50 {
				m.reportedLagEstimate.PreEchoLag = dlyComputePreEchoLag(m.accumulatedError[winnerIndex],
					m.winnerLag, winnerIndex*m.filterIntraLagShift)
			} else {
				m.reportedLagEstimate.PreEchoLag = m.winnerLag
			}
		}
		m.lastDetectedBestLagFilter = winnerIndex
	}
}
