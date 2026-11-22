package aec3

import "math"

// Port of render_signal_analyzer.{h,cc}.

const afirCounterThreshold = 5

// RenderSignalAnalyzer analyzes the properties of the render signal
// (render_signal_analyzer.h).
type RenderSignalAnalyzer struct {
	strongPeakFreezeDuration int
	narrowBandCounters       [FftLengthBy2 - 1]int
	narrowPeakBand           int
	hasNarrowPeakBand        bool
	narrowPeakCounter        int
	// Whether a counter of narrowBandCounters is above 10, and above
	// afirCounterThreshold. identifySmallNarrowBandRegions, the only writer
	// of the counters, keeps them current.
	poorSignalExcitation bool
	narrowBandsToMask    bool
}

func NewRenderSignalAnalyzer(config *EchoCanceller3Config) *RenderSignalAnalyzer {
	return &RenderSignalAnalyzer{strongPeakFreezeDuration: config.Filter.Refined.LengthBlocks}
}

// Update takes the render signal of one block. hasDelay is false for a
// std::nullopt delay_partitions.
func (r *RenderSignalAnalyzer) Update(renderBuffer *RenderBuffer, delayPartitions int, hasDelay bool) {
	r.identifySmallNarrowBandRegions(renderBuffer, delayPartitions, hasDelay)
	r.identifyStrongNarrowBandComponent(renderBuffer)
}

func (r *RenderSignalAnalyzer) identifySmallNarrowBandRegions(renderBuffer *RenderBuffer, delayPartitions int, hasDelay bool) {
	if !hasDelay {
		r.narrowBandCounters = [FftLengthBy2 - 1]int{}
		r.poorSignalExcitation = false
		r.narrowBandsToMask = false
		return
	}
	// narrow[k-1] is all ones when bin k is a narrow band in some channel.
	// The C++ counts the channels and tests the count.
	var narrow [FftLengthBy2]int32
	X2 := renderBuffer.Spectrum(delayPartitions)
	for ch := range X2 {
		kernNarrowBands(&X2[ch], &narrow)
	}
	// above10 and aboveThreshold turn negative when a counter exceeds 10 or
	// afirCounterThreshold. Counters are never negative.
	above10, aboveThreshold := 0, 0
	for k := range r.narrowBandCounters {
		// Increment, or reset to zero, without a branch.
		c := (r.narrowBandCounters[k] + 1) & int(narrow[k])
		r.narrowBandCounters[k] = c
		above10 |= 10 - c
		aboveThreshold |= afirCounterThreshold - c
	}
	r.poorSignalExcitation = above10 < 0
	r.narrowBandsToMask = aboveThreshold < 0
}

// rsaMax is std::max(a, b). It selects between the bit patterns with a mask,
// which compiles without a branch.
func rsaMax(a, b float32) float32 {
	mask := -uint32(rsaB2i(a < b))
	return math.Float32frombits(math.Float32bits(a)&^mask | math.Float32bits(b)&mask)
}

func rsaB2i(c bool) int {
	i := 0
	if c {
		i = 1
	}
	return i
}

func (r *RenderSignalAnalyzer) identifyStrongNarrowBandComponent(renderBuffer *RenderBuffer) {
	if r.hasNarrowPeakBand {
		r.narrowPeakCounter++
		if r.narrowPeakCounter > r.strongPeakFreezeDuration {
			r.hasNarrowPeakBand = false
			r.narrowPeakBand = 0
		}
	}

	xLatest := renderBuffer.GetBlock(0)
	var maxPeakLevel float32
	for channel := 0; channel < xLatest.NumChannels(); channel++ {
		X2Latest := &renderBuffer.Spectrum(0)[channel]

		peakBin := kernPeakBin(X2Latest)

		// The level around the peak. These maxima are chains whose branches
		// predict well, so they stay branches.
		var nonPeakPower float32
		for k := max(0, peakBin-14); k < peakBin-4; k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}
		for k := peakBin + 5; k < min(peakBin+15, FftLengthBy2Plus1); k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}

		maxAbs := afirMaxAbsMinMax(xLatest.View(0, channel))
		if xLatest.NumBands() > 1 {
			maxAbs = stdMax(maxAbs, afirMaxAbsMinMax(xLatest.View(1, channel)))
		}

		peakLevel := X2Latest[peakBin]
		if peakBin > 0 && maxAbs > 100 && peakLevel > 100*nonPeakPower {
			if peakLevel > maxPeakLevel {
				maxPeakLevel = peakLevel
				r.narrowPeakBand = peakBin
				r.hasNarrowPeakBand = true
				r.narrowPeakCounter = 0
			}
		}
	}
}

// afirMaxAbsMinMax is std::max(fabs(*r.first), fabs(*r.second)) with r =
// std::minmax_element(x).
//
// Without a NaN in x that is the largest magnitude in x (kernMaxAbsBits).
// With one it depends on the order of the comparisons, which
// afirMaxAbsMinMaxRef follows.
func afirMaxAbsMinMax(x []float32) float32 {
	if mn, mx, ok := kernMinMax(x); ok {
		return stdMax(fabsf(mn), fabsf(mx))
	}
	if m := kernMaxAbsBits(x); m <= kernInfBits {
		return math.Float32frombits(m)
	}
	return afirMaxAbsMinMaxRef(x)
}

// afirMaxAbsMinMaxRef is afirMaxAbsMinMax comparison for comparison.
func afirMaxAbsMinMaxRef(x []float32) float32 {
	mn, mx := stdMinmaxElement(x)
	return stdMax(fabsf(mn), fabsf(mx))
}

// PoorSignalExcitation reports whether the render signal excites the filter
// poorly.
func (r *RenderSignalAnalyzer) PoorSignalExcitation() bool {
	return r.poorSignalExcitation
}

// MaskRegionsAroundNarrowBands zeroes v around the narrow bands.
func (r *RenderSignalAnalyzer) MaskRegionsAroundNarrowBands(v *[FftLengthBy2Plus1]float32) {
	if !r.narrowBandsToMask {
		return
	}
	if r.narrowBandCounters[0] > afirCounterThreshold {
		v[1], v[0] = 0, 0
	}
	for k := 2; k < FftLengthBy2-1; k++ {
		if r.narrowBandCounters[k-1] > afirCounterThreshold {
			v[k-2], v[k-1], v[k], v[k+1], v[k+2] = 0, 0, 0, 0, 0
		}
	}
	if r.narrowBandCounters[FftLengthBy2-2] > afirCounterThreshold {
		v[FftLengthBy2], v[FftLengthBy2-1] = 0, 0
	}
}

// NarrowPeakBand returns the bin of the strong narrow-band peak, if any.
func (r *RenderSignalAnalyzer) NarrowPeakBand() (int, bool) {
	return r.narrowPeakBand, r.hasNarrowPeakBand
}
