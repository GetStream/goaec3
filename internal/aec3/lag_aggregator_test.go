package aec3

import (
	"math/rand"
	"testing"
)

// The reference aggregators rescan the full histogram on every update, like
// the C++ std::max_element calls.

func refArgMax(v []int) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

type refHighestPeakAggregator struct {
	histogram          []int
	histogramData      [250]int
	histogramDataIndex int
	candidate          int
}

func newRefHighestPeakAggregator(maxFilterLag int) *refHighestPeakAggregator {
	return &refHighestPeakAggregator{histogram: make([]int, maxFilterLag+1), candidate: -1}
}

func (a *refHighestPeakAggregator) Reset() {
	clear(a.histogram)
	a.histogramData = [250]int{}
	a.histogramDataIndex = 0
}

func (a *refHighestPeakAggregator) Aggregate(lag int) {
	a.histogram[a.histogramData[a.histogramDataIndex]]--
	a.histogramData[a.histogramDataIndex] = lag
	a.histogram[lag]++
	a.histogramDataIndex = (a.histogramDataIndex + 1) % len(a.histogramData)
	a.candidate = refArgMax(a.histogram)
}

type refPreEchoLagAggregator struct {
	blockSizeLog2      int
	histogramData      [250]int
	histogram          []int
	histogramDataIndex int
	preEchoCandidate   int
	numberUpdates      int
}

func newRefPreEchoLagAggregator(maxFilterLag, downSamplingFactor int) *refPreEchoLagAggregator {
	a := &refPreEchoLagAggregator{
		blockSizeLog2: dlyGetDownSamplingBlockSizeLog2(downSamplingFactor),
		histogram:     make([]int, ((maxFilterLag+1)*downSamplingFactor)>>BlockSizeLog2),
	}
	a.Reset()
	return a
}

func (a *refPreEchoLagAggregator) Reset() {
	clear(a.histogram)
	for i := range a.histogramData {
		a.histogramData[i] = dlyPreEchoHistogramDataNotUpdated
	}
	a.histogramDataIndex = 0
	a.preEchoCandidate = 0
}

func (a *refPreEchoLagAggregator) Aggregate(preEchoLag int) {
	preEchoBlockSize := preEchoLag >> a.blockSizeLog2
	preEchoBlockSize = min(max(preEchoBlockSize, 0), len(a.histogram)-1)
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
			itMaxElement := it + refArgMax(a.histogram[it:it+MatchedFilterWindowSizeSubBlocks])
			weightedMaxValue := float32(a.histogram[itMaxElement]) * penalizationPerDelay
			if weightedMaxValue > maxHistogramValue {
				maxHistogramValue = weightedMaxValue
				preEchoCandidateBlockSize = itMaxElement
			}
			penalizationPerDelay *= 0.7
		}
	} else {
		preEchoCandidateBlockSize = refArgMax(a.histogram)
	}
	a.preEchoCandidate = preEchoCandidateBlockSize << a.blockSizeLog2
}

// lagSequence draws lags that mostly jitter around a few peaks, with bursts
// of uniformly random lags and peak moves, so that the maximum bin both
// grows and shrinks and ties are frequent.
type lagSequence struct {
	rng    *rand.Rand
	maxLag int
	peak   int
}

func (s *lagSequence) next(i int) int {
	if s.rng.Intn(400) == 0 {
		s.peak = s.rng.Intn(s.maxLag + 1)
	}
	switch {
	case i%20000 > 15000, s.rng.Intn(50) == 0:
		return s.rng.Intn(s.maxLag + 1)
	case s.rng.Intn(3) == 0:
		return min(s.maxLag, s.peak+s.rng.Intn(2)*64)
	default:
		return min(s.maxLag, s.peak+s.rng.Intn(3))
	}
}

func TestHighestPeakAggregatorIncrementalArgMax(t *testing.T) {
	for _, maxLag := range []int{1, 63, 2432} {
		rng := rand.New(rand.NewSource(int64(maxLag)))
		seq := &lagSequence{rng: rng, maxLag: maxLag, peak: maxLag / 3}
		ref := newRefHighestPeakAggregator(maxLag)
		var got dlyHighestPeakAggregator
		got.init(maxLag)
		if got.candidate != ref.candidate {
			t.Fatalf("maxLag %d: initial candidate %d, want %d", maxLag, got.candidate, ref.candidate)
		}
		for i := 0; i < 300000; i++ {
			if rng.Intn(5000) == 0 {
				ref.Reset()
				got.Reset()
			}
			lag := seq.next(i)
			ref.Aggregate(lag)
			got.Aggregate(lag)
			if got.candidate != ref.candidate {
				t.Fatalf("maxLag %d, update %d: candidate %d, want %d", maxLag, i, got.candidate, ref.candidate)
			}
		}
		for k := range ref.histogram {
			if int(got.histogram[k]) != ref.histogram[k] {
				t.Fatalf("maxLag %d: histogram[%d] = %d, want %d", maxLag, k, got.histogram[k], ref.histogram[k])
			}
		}
	}
}

func TestPreEchoLagAggregatorIncrementalArgMax(t *testing.T) {
	for _, c := range []struct{ maxLag, factor int }{{2432, 4}, {1215, 8}, {600, 4}} {
		rng := rand.New(rand.NewSource(int64(c.maxLag * c.factor)))
		maxPreEchoLag := (c.maxLag + 1) * c.factor
		seq := &lagSequence{rng: rng, maxLag: maxPreEchoLag + 100, peak: maxPreEchoLag / 2}
		ref := newRefPreEchoLagAggregator(c.maxLag, c.factor)
		got := newDlyPreEchoLagAggregator(c.maxLag, c.factor)
		for i := 0; i < 200000; i++ {
			if rng.Intn(3000) == 0 {
				ref.Reset()
				got.Reset()
			}
			lag := seq.next(i) - 50
			ref.Aggregate(lag)
			got.Aggregate(lag)
			if got.preEchoCandidate != ref.preEchoCandidate {
				t.Fatalf("%+v, update %d: candidate %d, want %d", c, i, got.preEchoCandidate, ref.preEchoCandidate)
			}
		}
	}
}

func TestMatchedFilterLagAggregatorAllocationFree(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	cfg.Delay.DetectPreEcho = true
	a := NewMatchedFilterLagAggregator(2432, &cfg.Delay)
	i := 0
	allocs := testing.AllocsPerRun(1000, func() {
		a.Aggregate(MatchedFilterLagEstimate{Lag: 700 + i%3, PreEchoLag: 690 + i%5}, true)
		i++
	})
	if allocs != 0 {
		t.Fatalf("Aggregate allocates %v times per call", allocs)
	}
}

func benchmarkLags() []int {
	rng := rand.New(rand.NewSource(1))
	seq := &lagSequence{rng: rng, maxLag: 2432, peak: 700}
	lags := make([]int, 4096)
	for i := range lags {
		lags[i] = seq.next(i)
	}
	return lags
}

func BenchmarkHighestPeakAggregator(b *testing.B) {
	lags := benchmarkLags()
	b.Run("rescan", func(b *testing.B) {
		a := newRefHighestPeakAggregator(2432)
		for i := 0; i < b.N; i++ {
			a.Aggregate(lags[i&4095])
		}
	})
	b.Run("incremental", func(b *testing.B) {
		var a dlyHighestPeakAggregator
		a.init(2432)
		for i := 0; i < b.N; i++ {
			a.Aggregate(lags[i&4095])
		}
	})
}
