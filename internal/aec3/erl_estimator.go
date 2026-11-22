package aec3

const (
	stMinErl float32 = 0.01
	stMaxErl float32 = 1000
)

// ErlEstimator estimates the echo return loss (erl_estimator.h).
type ErlEstimator struct {
	startupPhaseLengthBlocks int
	erl                      [FftLengthBy2Plus1]float32
	// holdCounters[k] is the hold counter of bin k, hold_counters_[k - 1] in
	// the C++. holdCounters[0] is unused.
	holdCounters          [FftLengthBy2]int32
	erlTimeDomain         float32
	holdCounterTimeDomain int
	blocksSinceReset      int
}

func NewErlEstimator(startupPhaseLengthBlocks int) *ErlEstimator {
	e := &ErlEstimator{}
	e.init(startupPhaseLengthBlocks)
	return e
}

func (e *ErlEstimator) init(startupPhaseLengthBlocks int) {
	*e = ErlEstimator{startupPhaseLengthBlocks: startupPhaseLengthBlocks, erlTimeDomain: stMaxErl}
	for k := range e.erl {
		e.erl[k] = stMaxErl
	}
}

func (e *ErlEstimator) Reset() { e.blocksSinceReset = 0 }

func (e *ErlEstimator) Erl() *[FftLengthBy2Plus1]float32 { return &e.erl }
func (e *ErlEstimator) ErlTimeDomain() float32           { return e.erlTimeDomain }

// Update takes the spectra of one block.
func (e *ErlEstimator) Update(convergedFilters []bool, renderSpectra, captureSpectra [][FftLengthBy2Plus1]float32) {
	numCaptureChannels := len(convergedFilters)
	const x2Min = stErlX2Min
	firstConverged := -1
	for ch, c := range convergedFilters {
		if c {
			firstConverged = ch
			break
		}
	}
	anyFilterConverged := firstConverged >= 0
	e.blocksSinceReset++
	if e.blocksSinceReset < e.startupPhaseLengthBlocks || !anyFilterConverged {
		return
	}

	// With one channel the maximum over the channels is the channel itself.
	var maxCaptureSpectrum [FftLengthBy2Plus1]float32
	Y2 := &captureSpectra[0]
	if numCaptureChannels > 1 {
		maxCaptureSpectrum = captureSpectra[firstConverged]
		for ch := firstConverged + 1; ch < numCaptureChannels; ch++ {
			if !convergedFilters[ch] {
				continue
			}
			kernStdMax(maxCaptureSpectrum[:], captureSpectra[ch][:], maxCaptureSpectrum[:])
		}
		Y2 = &maxCaptureSpectrum
	}

	var maxRenderSpectrum [FftLengthBy2Plus1]float32
	X2 := &renderSpectra[0]
	if len(renderSpectra) > 1 {
		maxRenderSpectrum = renderSpectra[0]
		for ch := 1; ch < len(renderSpectra); ch++ {
			kernStdMax(maxRenderSpectrum[:], renderSpectra[ch][:], maxRenderSpectrum[:])
		}
		X2 = &maxRenderSpectrum
	}

	kernErlUpdate(X2, Y2, &e.erl, &e.holdCounters)
	e.erl[0] = e.erl[1]
	e.erl[FftLengthBy2] = e.erl[FftLengthBy2-1]

	// Both sums accumulate in bin order. They are independent, so the chains
	// run side by side.
	var x2Sum, y2Sum float32
	for k := range X2 {
		x2Sum += X2[k]
		y2Sum += Y2[k]
	}
	if x2Sum > x2Min*float32(len(X2)) {
		newErl := y2Sum / x2Sum
		if newErl < e.erlTimeDomain {
			e.holdCounterTimeDomain = 1000
			e.erlTimeDomain += float32(0.1 * (newErl - e.erlTimeDomain))
			e.erlTimeDomain = stdMax(e.erlTimeDomain, stMinErl)
		}
	}
	e.holdCounterTimeDomain--
	if e.holdCounterTimeDomain <= 0 {
		e.erlTimeDomain = stdMin(stMaxErl, 2*e.erlTimeDomain)
	}
}
