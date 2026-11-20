package aec3

const dlyPreEchoHistogramDataNotUpdated = -1

func dlyGetDownSamplingBlockSizeLog2(downSamplingFactor int) int {
	downSamplingFactorLog2 := 0
	downSamplingFactor >>= 1
	for downSamplingFactor > 0 {
		downSamplingFactorLog2++
		downSamplingFactor >>= 1
	}
	if BlockSizeLog2 > downSamplingFactorLog2 {
		return BlockSizeLog2 - downSamplingFactorLog2
	}
	return 0
}

// dlyHistogramMax tracks the first maximum (std::max_element) of a lag
// histogram whose bins change by one count at a time: a bin that grows can
// only take over the maximum, and the histogram is rescanned only when the
// maximum bin shrinks.
type dlyHistogramMax struct {
	best int
}

// update records histogram[removed]-- (unless removed < 0) followed by
// histogram[added]++, both already applied to histogram.
func (m *dlyHistogramMax) update(histogram []int32, removed, added int) {
	if removed == added {
		return
	}
	if removed == m.best {
		m.best = dlyArgMax(histogram)
		return
	}
	if v, b := histogram[added], histogram[m.best]; v > b || (v == b && added < m.best) {
		m.best = added
	}
}

// dlyArgMax returns the index of the first maximum of v (std::max_element).
func dlyArgMax(v []int32) int {
	best := 0
	if len(v) == 0 {
		return best
	}
	bestValue := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestValue {
			best, bestValue = i, v[i]
		}
	}
	return best
}

type dlyPreEchoLagAggregator struct {
	*dlyPreEchoLagState
	histogram []int32
}

// dlyPreEchoLagState is the state of a dlyPreEchoLagAggregator but its
// histogram.
type dlyPreEchoLagState struct {
	histogramSpan      span
	blockSizeLog2      int
	histogramData      [250]int32
	histogramMax       dlyHistogramMax
	histogramDataIndex int
	preEchoCandidate   int
	numberUpdates      int
}

// dlyPreEchoHistogramSize returns the number of bins of the pre-echo
// histogram.
func dlyPreEchoHistogramSize(maxFilterLag, downSamplingFactor int) int {
	return ((maxFilterLag + 1) * downSamplingFactor) >> BlockSizeLog2
}

func newDlyPreEchoLagAggregator(maxFilterLag, downSamplingFactor int) *dlyPreEchoLagAggregator {
	a := &dlyPreEchoLagAggregator{}
	a.init(downSamplingFactor, make([]int32, dlyPreEchoHistogramSize(maxFilterLag, downSamplingFactor)))
	return a
}

// init sets up the aggregator over histogram, which holds
// dlyPreEchoHistogramSize zeros.
func (a *dlyPreEchoLagAggregator) init(downSamplingFactor int, histogram []int32) {
	if a.dlyPreEchoLagState == nil {
		a.dlyPreEchoLagState = &dlyPreEchoLagState{}
	}
	a.histogram = histogram
	*a.dlyPreEchoLagState = dlyPreEchoLagState{
		blockSizeLog2: dlyGetDownSamplingBlockSizeLog2(downSamplingFactor),
	}
	a.Reset()
}

func (a *dlyPreEchoLagAggregator) Reset() {
	clear(a.histogram)
	for i := range a.histogramData {
		a.histogramData[i] = dlyPreEchoHistogramDataNotUpdated
	}
	a.histogramMax.best = 0
	a.histogramDataIndex = 0
	a.preEchoCandidate = 0
}

func (a *dlyPreEchoLagAggregator) Aggregate(preEchoLag int) {
	preEchoBlockSize := preEchoLag >> a.blockSizeLog2
	preEchoBlockSize = min(max(preEchoBlockSize, 0), len(a.histogram)-1)
	// Remove the oldest point from the histogram, it ignores the initial
	// points where no updates have been done to the histogramData array.
	removed := int(a.histogramData[a.histogramDataIndex])
	if removed != dlyPreEchoHistogramDataNotUpdated {
		a.histogram[removed]--
	}
	a.histogramData[a.histogramDataIndex] = int32(preEchoBlockSize)
	a.histogram[preEchoBlockSize]++
	a.histogramMax.update(a.histogram, removed, preEchoBlockSize)
	a.histogramDataIndex = (a.histogramDataIndex + 1) % len(a.histogramData)
	preEchoCandidateBlockSize := 0
	if a.numberUpdates < NumBlocksPerSecond*2 {
		a.numberUpdates++
		penalizationPerDelay := float32(1)
		maxHistogramValue := float32(-1)
		for it := 0; len(a.histogram)-it >= MatchedFilterWindowSizeSubBlocks; it += MatchedFilterWindowSizeSubBlocks {
			itMaxElement := it + dlyArgMax(a.histogram[it:it+MatchedFilterWindowSizeSubBlocks])
			weightedMaxValue := float32(a.histogram[itMaxElement]) * penalizationPerDelay
			if weightedMaxValue > maxHistogramValue {
				maxHistogramValue = weightedMaxValue
				preEchoCandidateBlockSize = itMaxElement
			}
			penalizationPerDelay *= 0.7
		}
	} else {
		preEchoCandidateBlockSize = a.histogramMax.best
	}
	a.preEchoCandidate = preEchoCandidateBlockSize << a.blockSizeLog2
}

type dlyHighestPeakAggregator struct {
	*dlyHighestPeakState
	histogram []int32
}

// dlyHighestPeakState is the state of a dlyHighestPeakAggregator but its
// histogram.
type dlyHighestPeakState struct {
	histogramSpan      span
	histogramData      [250]int32
	histogramMax       dlyHistogramMax
	histogramDataIndex int
	candidate          int
}

func (a *dlyHighestPeakAggregator) init(maxFilterLag int) {
	a.initOver(make([]int32, maxFilterLag+1))
}

// initOver sets up the aggregator over histogram, which holds maxFilterLag+1
// zeros.
func (a *dlyHighestPeakAggregator) initOver(histogram []int32) {
	if a.dlyHighestPeakState == nil {
		a.dlyHighestPeakState = &dlyHighestPeakState{}
	}
	a.histogram = histogram
	*a.dlyHighestPeakState = dlyHighestPeakState{candidate: -1}
}

// Reset clears the histogram; the candidate keeps its value until the next
// Aggregate, as in the C++.
func (a *dlyHighestPeakAggregator) Reset() {
	clear(a.histogram)
	a.histogramData = [250]int32{}
	a.histogramMax.best = 0
	a.histogramDataIndex = 0
}

func (a *dlyHighestPeakAggregator) Aggregate(lag int) {
	removed := int(a.histogramData[a.histogramDataIndex])
	a.histogram[removed]--
	a.histogramData[a.histogramDataIndex] = int32(lag)
	a.histogram[lag]++
	a.histogramMax.update(a.histogram, removed, lag)
	a.histogramDataIndex = (a.histogramDataIndex + 1) % len(a.histogramData)
	a.candidate = a.histogramMax.best
}

// MatchedFilterLagAggregator aggregates lag estimates produced by the
// MatchedFilter into a single reliable combined lag estimate
// (matched_filter_lag_aggregator.h).
type MatchedFilterLagAggregator struct {
	*lagAggregatorState
	highestPeakAggregator dlyHighestPeakAggregator
	preEchoLagAggregator  dlyPreEchoLagAggregator
}

// lagAggregatorState is the state of a MatchedFilterLagAggregator but its
// histograms.
type lagAggregatorState struct {
	significantCandidateFound bool
	thresholdsInitial         int
	thresholdsConverged       int
	headroom                  int
	highestPeakAggregator     dlyHighestPeakState
	hasPreEchoLagAggregator   bool
	preEchoLagAggregator      dlyPreEchoLagState
}

// bind makes a the view of st.
func (a *MatchedFilterLagAggregator) bind(st *lagAggregatorState, b *binder) {
	a.lagAggregatorState = st
	a.highestPeakAggregator.dlyHighestPeakState = &st.highestPeakAggregator
	b.i32(&a.highestPeakAggregator.histogram, &st.highestPeakAggregator.histogramSpan)
	a.preEchoLagAggregator.dlyPreEchoLagState = &st.preEchoLagAggregator
	b.i32(&a.preEchoLagAggregator.histogram, &st.preEchoLagAggregator.histogramSpan)
}

// NewMatchedFilterLagAggregator mirrors the C++ constructor.
func NewMatchedFilterLagAggregator(maxFilterLag int, config *DelayConfig) *MatchedFilterLagAggregator {
	a := &MatchedFilterLagAggregator{}
	a.bind(&lagAggregatorState{}, nil)
	a.init(maxFilterLag, config)
	return a
}

func (a *MatchedFilterLagAggregator) init(maxFilterLag int, d *DelayConfig) {
	st := a.lagAggregatorState
	*st = lagAggregatorState{
		thresholdsInitial:   d.DelaySelectionThresholds.Initial,
		thresholdsConverged: d.DelaySelectionThresholds.Converged,
		headroom:            d.DelayHeadroomSamples / d.DownSamplingFactor,
	}
	*a = MatchedFilterLagAggregator{}
	a.bind(st, nil)
	// The two histograms share one array.
	n, nPreEcho := maxFilterLag+1, 0
	if d.DetectPreEcho {
		nPreEcho = dlyPreEchoHistogramSize(maxFilterLag, d.DownSamplingFactor)
	}
	histograms := make([]int32, n+nPreEcho)
	a.highestPeakAggregator.initOver(histograms[:n:n])
	if d.DetectPreEcho {
		a.hasPreEchoLagAggregator = true
		a.preEchoLagAggregator.init(d.DownSamplingFactor, histograms[n:])
	}
}

// Reset resets the aggregator.
func (a *MatchedFilterLagAggregator) Reset(hardReset bool) {
	a.highestPeakAggregator.Reset()
	if a.hasPreEchoLagAggregator {
		a.preEchoLagAggregator.Reset()
	}
	if hardReset {
		a.significantCandidateFound = false
	}
}

// Aggregate aggregates the provided lag estimate (present when ok is true).
func (a *MatchedFilterLagAggregator) Aggregate(lagEstimate MatchedFilterLagEstimate, ok bool) (DelayEstimate, bool) {
	if ok && a.hasPreEchoLagAggregator {
		a.preEchoLagAggregator.Aggregate(max(0, lagEstimate.PreEchoLag-a.headroom))
	}

	if ok {
		a.highestPeakAggregator.Aggregate(max(0, lagEstimate.Lag-a.headroom))
		histogram := a.highestPeakAggregator.histogram
		candidate := a.highestPeakAggregator.candidate
		a.significantCandidateFound = a.significantCandidateFound || int(histogram[candidate]) > a.thresholdsConverged
		if int(histogram[candidate]) > a.thresholdsConverged ||
			(int(histogram[candidate]) > a.thresholdsInitial && !a.significantCandidateFound) {
			quality := DelayQualityCoarse
			if a.significantCandidateFound {
				quality = DelayQualityRefined
			}
			reportedDelay := candidate
			if a.hasPreEchoLagAggregator {
				reportedDelay = a.preEchoLagAggregator.preEchoCandidate
			}
			return DelayEstimate{Quality: quality, Delay: reportedDelay}, true
		}
	}
	return DelayEstimate{}, false
}

// ReliableDelayFound returns whether a reliable delay estimate has been found.
func (a *MatchedFilterLagAggregator) ReliableDelayFound() bool { return a.significantCandidateFound }

// GetDelayAtHighestPeak returns the delay candidate computed from the highest
// peak of the matched filters.
func (a *MatchedFilterLagAggregator) GetDelayAtHighestPeak() int {
	return a.highestPeakAggregator.candidate
}
