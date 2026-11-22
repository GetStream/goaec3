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

// dlyHistogramMax tracks the first maximum of a lag histogram, as
// std::max_element finds it, while bins change by one count: a bin that
// grows can only take over the maximum, and only a maximum bin that shrinks
// needs a scan.
type dlyHistogramMax struct {
	best int
}

// update records histogram[removed]--, unless removed < 0, and then
// histogram[added]++. Both are already applied to histogram.
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

// dlyArgMax returns the index of the first maximum of v.
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

// dlyPreEchoLagState is the state of a dlyPreEchoLagAggregator except its
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

// init sets up a. histogram holds dlyPreEchoHistogramSize zeros.
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
	// Remove the oldest point from the histogram, unless histogramData is
	// not full yet.
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

// dlyHighestPeakState is the state of a dlyHighestPeakAggregator except its
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

// initOver sets up a. histogram holds maxFilterLag+1 zeros.
func (a *dlyHighestPeakAggregator) initOver(histogram []int32) {
	if a.dlyHighestPeakState == nil {
		a.dlyHighestPeakState = &dlyHighestPeakState{}
	}
	a.histogram = histogram
	*a.dlyHighestPeakState = dlyHighestPeakState{candidate: -1}
}

// Reset clears the histogram. The candidate keeps its value until the next
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

// MatchedFilterLagAggregator combines the lag estimates of the MatchedFilter
// into one reliable estimate (matched_filter_lag_aggregator.h).
type MatchedFilterLagAggregator struct {
	*lagAggregatorState
	highestPeakAggregator dlyHighestPeakAggregator
	preEchoLagAggregator  dlyPreEchoLagAggregator
}

// lagAggregatorState is the state of a MatchedFilterLagAggregator except its
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
	a.bind(&lagAggregatorState{}, bindAlone)
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
	a.bind(st, bindAlone)
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

// Reset clears the histograms. With hardReset it also forgets that a
// significant candidate was found.
func (a *MatchedFilterLagAggregator) Reset(hardReset bool) {
	a.highestPeakAggregator.Reset()
	if a.hasPreEchoLagAggregator {
		a.preEchoLagAggregator.Reset()
	}
	if hardReset {
		a.significantCandidateFound = false
	}
}

// Aggregate takes the lag estimate of one block, which is present if ok, and
// returns the aggregated delay, if there is one.
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

// ReliableDelayFound reports whether a reliable delay estimate was found.
func (a *MatchedFilterLagAggregator) ReliableDelayFound() bool { return a.significantCandidateFound }

// GetDelayAtHighestPeak returns the delay candidate at the highest peak of
// the matched filters.
func (a *MatchedFilterLagAggregator) GetDelayAtHighestPeak() int {
	return a.highestPeakAggregator.candidate
}
