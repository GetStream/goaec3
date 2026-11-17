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

type dlyPreEchoLagAggregator struct {
	blockSizeLog2      int
	histogramData      [250]int
	histogram          []int
	histogramDataIndex int
	preEchoCandidate   int
	numberUpdates      int
}

func newDlyPreEchoLagAggregator(maxFilterLag, downSamplingFactor int) *dlyPreEchoLagAggregator {
	a := &dlyPreEchoLagAggregator{
		blockSizeLog2: dlyGetDownSamplingBlockSizeLog2(downSamplingFactor),
		histogram:     make([]int, ((maxFilterLag+1)*downSamplingFactor)>>BlockSizeLog2),
	}
	a.Reset()
	return a
}

func (a *dlyPreEchoLagAggregator) Reset() {
	clear(a.histogram)
	for i := range a.histogramData {
		a.histogramData[i] = dlyPreEchoHistogramDataNotUpdated
	}
	a.histogramDataIndex = 0
	a.preEchoCandidate = 0
}

func dlyArgMax(v []int) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func (a *dlyPreEchoLagAggregator) Aggregate(preEchoLag int) {
	preEchoBlockSize := preEchoLag >> a.blockSizeLog2
	preEchoBlockSize = min(max(preEchoBlockSize, 0), len(a.histogram)-1)
	// Remove the oldest point from the histogram, it ignores the initial
	// points where no updates have been done to the histogramData array.
	if a.histogramData[a.histogramDataIndex] != dlyPreEchoHistogramDataNotUpdated {
		a.histogram[a.histogramData[a.histogramDataIndex]]--
	}
	a.histogramData[a.histogramDataIndex] = preEchoBlockSize
	a.histogram[preEchoBlockSize]++
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
		preEchoCandidateBlockSize = dlyArgMax(a.histogram)
	}
	a.preEchoCandidate = preEchoCandidateBlockSize << a.blockSizeLog2
}

type dlyHighestPeakAggregator struct {
	histogram          []int
	histogramData      [250]int
	histogramDataIndex int
	candidate          int
}

func (a *dlyHighestPeakAggregator) init(maxFilterLag int) {
	a.histogram = make([]int, maxFilterLag+1)
	a.candidate = -1
}

func (a *dlyHighestPeakAggregator) Reset() {
	clear(a.histogram)
	a.histogramData = [250]int{}
	a.histogramDataIndex = 0
}

func (a *dlyHighestPeakAggregator) Aggregate(lag int) {
	a.histogram[a.histogramData[a.histogramDataIndex]]--
	a.histogramData[a.histogramDataIndex] = lag
	a.histogram[lag]++
	a.histogramDataIndex = (a.histogramDataIndex + 1) % len(a.histogramData)
	a.candidate = dlyArgMax(a.histogram)
}

// MatchedFilterLagAggregator aggregates lag estimates produced by the
// MatchedFilter into a single reliable combined lag estimate
// (matched_filter_lag_aggregator.h).
type MatchedFilterLagAggregator struct {
	significantCandidateFound bool
	thresholdsInitial         int
	thresholdsConverged       int
	headroom                  int
	highestPeakAggregator     dlyHighestPeakAggregator
	preEchoLagAggregator      *dlyPreEchoLagAggregator
}

// NewMatchedFilterLagAggregator mirrors the C++ constructor; it reads
// config.Delay (the C++ takes EchoCanceller3Config::Delay).
func NewMatchedFilterLagAggregator(maxFilterLag int, config *EchoCanceller3Config) *MatchedFilterLagAggregator {
	a := &MatchedFilterLagAggregator{}
	a.init(maxFilterLag, config)
	return a
}

func (a *MatchedFilterLagAggregator) init(maxFilterLag int, config *EchoCanceller3Config) {
	d := &config.Delay
	*a = MatchedFilterLagAggregator{
		thresholdsInitial:   d.DelaySelectionThresholds.Initial,
		thresholdsConverged: d.DelaySelectionThresholds.Converged,
		headroom:            d.DelayHeadroomSamples / d.DownSamplingFactor,
	}
	a.highestPeakAggregator.init(maxFilterLag)
	if d.DetectPreEcho {
		a.preEchoLagAggregator = newDlyPreEchoLagAggregator(maxFilterLag, d.DownSamplingFactor)
	}
}

// Reset resets the aggregator.
func (a *MatchedFilterLagAggregator) Reset(hardReset bool) {
	a.highestPeakAggregator.Reset()
	if a.preEchoLagAggregator != nil {
		a.preEchoLagAggregator.Reset()
	}
	if hardReset {
		a.significantCandidateFound = false
	}
}

// Aggregate aggregates the provided lag estimate (present when ok is true).
func (a *MatchedFilterLagAggregator) Aggregate(lagEstimate MatchedFilterLagEstimate, ok bool) (DelayEstimate, bool) {
	if ok && a.preEchoLagAggregator != nil {
		a.preEchoLagAggregator.Aggregate(max(0, lagEstimate.PreEchoLag-a.headroom))
	}

	if ok {
		a.highestPeakAggregator.Aggregate(max(0, lagEstimate.Lag-a.headroom))
		histogram := a.highestPeakAggregator.histogram
		candidate := a.highestPeakAggregator.candidate
		a.significantCandidateFound = a.significantCandidateFound || histogram[candidate] > a.thresholdsConverged
		if histogram[candidate] > a.thresholdsConverged ||
			(histogram[candidate] > a.thresholdsInitial && !a.significantCandidateFound) {
			quality := DelayQualityCoarse
			if a.significantCandidateFound {
				quality = DelayQualityRefined
			}
			reportedDelay := candidate
			if a.preEchoLagAggregator != nil {
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
