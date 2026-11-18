package aec3

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
	var channelCounters [FftLengthBy2 - 1]int
	X2 := renderBuffer.Spectrum(delayPartitions)
	for ch := range X2 {
		x := &X2[ch]
		for k := 1; k < FftLengthBy2; k++ {
			if x[k] > 3*max(x[k-1], x[k+1]) {
				channelCounters[k-1]++
			}
		}
	}
	for k := 1; k < FftLengthBy2; k++ {
		if channelCounters[k-1] > 0 {
			r.narrowBandCounters[k-1]++
		} else {
			r.narrowBandCounters[k-1] = 0
		}
	}
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
		peakBin := 0
		for k := 1; k < FftLengthBy2Plus1; k++ {
			if X2Latest[k] > X2Latest[peakBin] {
				peakBin = k
			}
		}

		// Compute the level around the peak.
		var nonPeakPower float32
		for k := max(0, peakBin-14); k < peakBin-4; k++ {
			nonPeakPower = max(X2Latest[k], nonPeakPower)
		}
		for k := peakBin + 5; k < min(peakBin+15, FftLengthBy2Plus1); k++ {
			nonPeakPower = max(X2Latest[k], nonPeakPower)
		}

		// Assess the render signal strength.
		maxAbs := afirMaxAbsMinMax(xLatest.View(0, channel))
		if xLatest.NumBands() > 1 {
			maxAbs = max(maxAbs, afirMaxAbsMinMax(xLatest.View(1, channel)))
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

// afirMaxAbsMinMax is max(|min|, |max|) of x via std::minmax_element.
func afirMaxAbsMinMax(x []float32) float32 {
	if mn, mx, ok := kernMinMax(x); ok {
		return max(afirAbs(mn), afirAbs(mx))
	}
	mn, mx := x[0], x[0]
	for _, v := range x[1:] {
		if v < mn {
			mn = v
		}
		if !(v < mx) {
			mx = v
		}
	}
	return max(afirAbs(mn), afirAbs(mx))
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
