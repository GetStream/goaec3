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
	hSize := len(h)
	xSize := len(x)
	for i := range y {
		var acc [8]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		hp, xp := 0, xStartIndex
		for _, limit := range [2]int{chunk1, chunk2} {
			v := limit &^ 3
			dlyDotNeon(h[hp:hp+v], x[xp:xp+v], &acc)
			hp += v
			xp += v
			for k := limit - v; k > 0; k-- {
				xk := x[xp]
				x2Sum += float32(xk * xk)
				s += float32(h[hp] * xk)
				hp++
				xp++
			}
			xp = 0
		}
		s += (acc[0] + acc[1]) + (acc[2] + acc[3])
		x2Sum += (acc[4] + acc[5]) + (acc[6] + acc[7])

		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:chunk1], x[xStartIndex:xStartIndex+chunk1], alpha)
			dlyAxpy(h[chunk1:], x[:chunk2], alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
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
	hSize := len(h)
	xSize := len(x)
	for i := range y {
		var acc [16]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		hp, xp := 0, xStartIndex
		for _, limit := range [2]int{chunk1, chunk2} {
			v := limit &^ 7
			dlyDotSse2(h[hp:hp+v], x[xp:xp+v], &acc)
			hp += v
			xp += v
			for k := limit - v; k > 0; k-- {
				xk := x[xp]
				x2Sum += float32(xk * xk)
				s += float32(h[hp] * xk)
				hp++
				xp++
			}
			xp = 0
		}
		var v [4]float32
		for l := range v {
			v[l] = acc[8+l] + acc[12+l]
		}
		x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
		for l := range v {
			v[l] = acc[l] + acc[4+l]
		}
		s += ((v[0] + v[1]) + v[2]) + v[3]

		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:chunk1], x[xStartIndex:xStartIndex+chunk1], alpha)
			dlyAxpy(h[chunk1:], x[:chunk2], alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
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
	hSize := len(h)
	xSize := len(x)
	for i := range y {
		var acc [32]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		hp, xp := 0, xStartIndex
		for _, limit := range [2]int{chunk1, chunk2} {
			v := limit &^ 15
			dlyDotAvx2(h[hp:hp+v], x[xp:xp+v], &acc)
			hp += v
			xp += v
			for k := limit - v; k > 0; k-- {
				xk := x[xp]
				x2Sum += float32(xk * xk)
				s += float32(h[hp] * xk)
				hp++
				xp++
			}
			xp = 0
		}
		// hsum_ab(x2_sum_256 + x2_sum_256_8, s_256 + s_256_8).
		var a, b [8]float32
		for l := range a {
			a[l] = acc[16+l] + acc[24+l]
			b[l] = acc[l] + acc[8+l]
		}
		x2Sum += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
		s += ((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7]))

		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			hp, xp := 0, xStartIndex
			for _, limit := range [2]int{chunk1, chunk2} {
				v := limit &^ 7
				dlyAxpyFma(h[hp:hp+v], x[xp:xp+v], alpha)
				dlyAxpyRef(h[hp+v:hp+limit], x[xp+v:xp+limit], alpha)
				hp += limit
				xp = 0
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
func MaxSquarePeakIndex(h []float32) int {
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
