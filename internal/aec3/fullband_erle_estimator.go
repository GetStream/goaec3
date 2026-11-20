package aec3

const (
	stErleEpsilon            float32 = 1e-3
	stX2BandEnergyThreshold  float32 = 44015068.0
	stBlocksToHoldErle               = 100
	stErlePointsToAccumulate         = 6
)

// stErleInstantaneous mirrors FullBandErleEstimator::ErleInstantaneous.
type stErleInstantaneous struct {
	clampInstQualityToZero bool
	clampInstQualityToOne  bool
	erleLog2               LinearFilterQuality
	instQualityEstimate    float32
	maxErleLog2            float32
	minErleLog2            float32
	Y2Acum                 float32
	E2Acum                 float32
	numPoints              int
}

func (e *stErleInstantaneous) Update(Y2Sum, E2Sum float32) bool {
	updateEstimates := false
	e.E2Acum += E2Sum
	e.Y2Acum += Y2Sum
	e.numPoints++
	if e.numPoints == stErlePointsToAccumulate {
		if e.E2Acum > 0 {
			updateEstimates = true
			e.erleLog2 = LinearFilterQuality{Value: FastApproxLog2f(e.Y2Acum/e.E2Acum + stErleEpsilon), Ok: true}
		}
		e.numPoints = 0
		e.E2Acum = 0
		e.Y2Acum = 0
	}
	if updateEstimates {
		e.updateMaxMin()
		e.updateQualityEstimate()
	}
	return updateEstimates
}

func (e *stErleInstantaneous) Reset() {
	e.ResetAccumulators()
	e.maxErleLog2 = -10 // -30 dB.
	e.minErleLog2 = 33  // 100 dB.
	e.instQualityEstimate = 0
}

func (e *stErleInstantaneous) ResetAccumulators() {
	e.erleLog2 = LinearFilterQuality{}
	e.instQualityEstimate = 0
	e.numPoints = 0
	e.E2Acum = 0
	e.Y2Acum = 0
}

func (e *stErleInstantaneous) GetQualityEstimate() LinearFilterQuality {
	if e.erleLog2.Ok {
		value := e.instQualityEstimate
		if e.clampInstQualityToZero {
			value = stdMax(0, value)
		}
		if e.clampInstQualityToOne {
			value = stdMin(1, value)
		}
		return LinearFilterQuality{Value: value, Ok: true}
	}
	return LinearFilterQuality{}
}

func (e *stErleInstantaneous) updateMaxMin() {
	e.maxErleLog2 -= 0.0004
	e.maxErleLog2 = stdMax(e.maxErleLog2, e.erleLog2.Value)
	e.minErleLog2 += 0.0004
	e.minErleLog2 = stdMin(e.minErleLog2, e.erleLog2.Value)
}

func (e *stErleInstantaneous) updateQualityEstimate() {
	const alpha float32 = 0.07
	var qualityEstimate float32
	if e.maxErleLog2 > e.minErleLog2 {
		qualityEstimate = (e.erleLog2.Value - e.minErleLog2) / (e.maxErleLog2 - e.minErleLog2)
	}
	if qualityEstimate > e.instQualityEstimate {
		e.instQualityEstimate = qualityEstimate
	} else {
		e.instQualityEstimate += float32(alpha * (qualityEstimate - e.instQualityEstimate))
	}
}

// FullBandErleEstimator mirrors fullband_erle_estimator.h: estimates the
// fullband ERLE and the linear filter quality.
type FullBandErleEstimator struct {
	more               *fbErleChannels // with more than inlineChannels channels
	numCaptureChannels int
	minErleLog2        float32
	maxErleLfLog2      float32

	// The per-channel state for up to inlineChannels channels.
	inline struct {
		holdCountersInstantaneousErle [inlineChannels]int
		erleTimeDomainLog2            [inlineChannels]float32
		instantaneousErle             [inlineChannels]stErleInstantaneous
		linearFiltersQualities        [inlineChannels]LinearFilterQuality
	}
}

// fbErleChannels is the per-channel state of a FullBandErleEstimator.
type fbErleChannels struct {
	holdCountersInstantaneousErle []int
	erleTimeDomainLog2            []float32
	instantaneousErle             []stErleInstantaneous
	linearFiltersQualities        []LinearFilterQuality
}

// channels returns the per-channel state: views of f for up to
// inlineChannels channels.
func (f *FullBandErleEstimator) channels() fbErleChannels {
	if f.more != nil {
		return *f.more
	}
	n := f.numCaptureChannels
	return fbErleChannels{
		holdCountersInstantaneousErle: f.inline.holdCountersInstantaneousErle[:n:n],
		erleTimeDomainLog2:            f.inline.erleTimeDomainLog2[:n:n],
		instantaneousErle:             f.inline.instantaneousErle[:n:n],
		linearFiltersQualities:        f.inline.linearFiltersQualities[:n:n],
	}
}

func NewFullBandErleEstimator(config *EchoCanceller3Config, numCaptureChannels int) *FullBandErleEstimator {
	f := &FullBandErleEstimator{}
	f.init(config, numCaptureChannels)
	return f
}

// init sets up f in place.
func (f *FullBandErleEstimator) init(config *EchoCanceller3Config, numCaptureChannels int) {
	*f = FullBandErleEstimator{
		numCaptureChannels: numCaptureChannels,
		minErleLog2:        FastApproxLog2f(config.Erle.Min + stErleEpsilon),
		maxErleLfLog2:      FastApproxLog2f(config.Erle.MaxL + stErleEpsilon),
	}
	if n := numCaptureChannels; n > inlineChannelLimit {
		f.more = &fbErleChannels{
			holdCountersInstantaneousErle: make([]int, n),
			erleTimeDomainLog2:            make([]float32, n),
			instantaneousErle:             make([]stErleInstantaneous, n),
			linearFiltersQualities:        make([]LinearFilterQuality, n),
		}
	}
	c := f.channels()
	for ch := range c.instantaneousErle {
		c.instantaneousErle[ch] = stErleInstantaneous{
			clampInstQualityToZero: config.Erle.ClampQualityEstimateToZero,
			clampInstQualityToOne:  config.Erle.ClampQualityEstimateToOne,
		}
		c.instantaneousErle[ch].Reset()
	}
	f.Reset()
}

func (f *FullBandErleEstimator) Reset() {
	c := f.channels()
	for ch := range c.instantaneousErle {
		c.instantaneousErle[ch].Reset()
	}
	f.updateQualityEstimates(c)
	for ch := range c.erleTimeDomainLog2 {
		c.erleTimeDomainLog2[ch] = f.minErleLog2
	}
	clear(c.holdCountersInstantaneousErle)
}

// Update updates the ERLE estimates.
func (f *FullBandErleEstimator) Update(X2 []float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	c := f.channels()
	for ch := range Y2 {
		if convergedFilters[ch] {
			// The three sums accumulate in bin order. The capture and
			// error sums are only used with enough render power; running
			// the chains side by side takes a third of the wait for them.
			var X2Sum, Y2Sum, E2Sum float32
			if len(X2) == FftLengthBy2Plus1 {
				x, y, e := (*[FftLengthBy2Plus1]float32)(X2), &Y2[ch], &E2[ch]
				for k := range x {
					X2Sum += x[k]
					Y2Sum += y[k]
					E2Sum += e[k]
				}
			} else {
				X2Sum = stAccumulate(X2)
				Y2Sum = stAccumulate(Y2[ch][:])
				E2Sum = stAccumulate(E2[ch][:])
			}
			if X2Sum > stX2BandEnergyThreshold*float32(len(X2)) {
				if c.instantaneousErle[ch].Update(Y2Sum, E2Sum) {
					c.holdCountersInstantaneousErle[ch] = stBlocksToHoldErle
					c.erleTimeDomainLog2[ch] += float32(0.05 * (c.instantaneousErle[ch].erleLog2.Value - c.erleTimeDomainLog2[ch]))
					c.erleTimeDomainLog2[ch] = stdMax(c.erleTimeDomainLog2[ch], f.minErleLog2)
				}
			}
		}
		c.holdCountersInstantaneousErle[ch]--
		if c.holdCountersInstantaneousErle[ch] == 0 {
			c.instantaneousErle[ch].ResetAccumulators()
		}
	}
	f.updateQualityEstimates(c)
}

// FullbandErleLog2 returns the minimum fullband ERLE over channels (log2).
func (f *FullBandErleEstimator) FullbandErleLog2() float32 {
	erle := f.channels().erleTimeDomainLog2
	minErle := erle[0]
	for ch := 1; ch < len(erle); ch++ {
		minErle = stdMin(minErle, erle[ch])
	}
	return minErle
}

// GetInstLinearQualityEstimates returns the per-channel linear filter quality.
func (f *FullBandErleEstimator) GetInstLinearQualityEstimates() []LinearFilterQuality {
	return f.channels().linearFiltersQualities
}

func (f *FullBandErleEstimator) updateQualityEstimates(c fbErleChannels) {
	for ch := range c.instantaneousErle {
		c.linearFiltersQualities[ch] = c.instantaneousErle[ch].GetQualityEstimate()
	}
}
