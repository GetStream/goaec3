package aec3

import "math"

// Port of render_signal_analyzer.{h,cc}.

const afirCounterThreshold = 5

// RenderSignalAnalyzer analyzes the properties of the render signal.
type RenderSignalAnalyzer struct {
	strongPeakFreezeDuration int
	narrowBandCounters       [FftLengthBy2 - 1]int
	narrowPeakBand           int
	hasNarrowPeakBand        bool
	narrowPeakCounter        int
}

func NewRenderSignalAnalyzer(config *EchoCanceller3Config) *RenderSignalAnalyzer {
	return &RenderSignalAnalyzer{strongPeakFreezeDuration: config.Filter.Refined.LengthBlocks}
}

// Update updates the analysis with the most recent render signal.
// hasDelay=false corresponds to a std::nullopt delay_partitions.
func (r *RenderSignalAnalyzer) Update(renderBuffer *RenderBuffer, delayPartitions int, hasDelay bool) {
	r.identifySmallNarrowBandRegions(renderBuffer, delayPartitions, hasDelay)
	r.identifyStrongNarrowBandComponent(renderBuffer)
}

func (r *RenderSignalAnalyzer) identifySmallNarrowBandRegions(renderBuffer *RenderBuffer, delayPartitions int, hasDelay bool) {
	if !hasDelay {
		r.narrowBandCounters = [FftLengthBy2 - 1]int{}
		return
	}
	// narrow[k-1] is all ones when bin k is a narrow band in some channel
	// (the C++ counts the channels and tests the count against zero).
	var narrow [FftLengthBy2]int32
	X2 := renderBuffer.Spectrum(delayPartitions)
	for ch := range X2 {
		kernNarrowBands(&X2[ch], &narrow)
	}
	for k := range r.narrowBandCounters {
		// Increment, or reset to zero, without a branch.
		r.narrowBandCounters[k] = (r.narrowBandCounters[k] + 1) & int(narrow[k])
	}
}

// rsaMax is std::max(a, b): a unless a < b, so a NaN b yields a. It selects
// between the bit patterns with a mask, which compiles to straight-line code
// (a float select compiles to a branch).
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

		// Identify the spectral peak (first maximum).
		peakBin := kernPeakBin(X2Latest)

		// Compute the level around the peak.
		// The max folds below form serial dependency chains whose branches
		// predict well, so they stay branches.
		var nonPeakPower float32
		for k := max(0, peakBin-14); k < peakBin-4; k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}
		for k := peakBin + 5; k < min(peakBin+15, FftLengthBy2Plus1); k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}

		// Assess the render signal strength.
		maxAbs := afirMaxAbsMinMax(xLatest.View(0, channel))
		if xLatest.NumBands() > 1 {
			maxAbs = stdMax(maxAbs, afirMaxAbsMinMax(xLatest.View(1, channel)))
		}

		// Detect whether the spectral peak has as strong narrowband nature.
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

// afirMaxAbsMinMax is std::max(fabs(*r.first), fabs(*r.second)) with
// r = std::minmax_element(x).
func afirMaxAbsMinMax(x []float32) float32 {
	mn, mx, ok := kernMinMax(x)
	if !ok {
		mn, mx = stdMinmaxElement(x)
	}
	return stdMax(fabsf(mn), fabsf(mx))
}

// PoorSignalExcitation reports whether the render signal is poorly exciting.
func (r *RenderSignalAnalyzer) PoorSignalExcitation() bool {
	for _, a := range r.narrowBandCounters {
		if a > 10 {
			return true
		}
	}
	return false
}

// MaskRegionsAroundNarrowBands zeros v around narrow-band regions.
func (r *RenderSignalAnalyzer) MaskRegionsAroundNarrowBands(v *[FftLengthBy2Plus1]float32) {
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

// NarrowPeakBand returns the strong narrow-band peak bin, if any.
func (r *RenderSignalAnalyzer) NarrowPeakBand() (int, bool) {
	return r.narrowPeakBand, r.hasNarrowPeakBand
}
