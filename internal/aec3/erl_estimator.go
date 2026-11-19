package aec3

const (
	stMinErl float32 = 0.01
	stMaxErl float32 = 1000
)

// ErlEstimator mirrors erl_estimator.h: estimates the echo return loss.
type ErlEstimator struct {
	startupPhaseLengthBlocks int
	erl                      [FftLengthBy2Plus1]float32
	holdCounters             [FftLengthBy2Minus1]int
	erlTimeDomain            float32
	holdCounterTimeDomain    int
	blocksSinceReset         int
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

// Update updates the ERL estimate.
func (e *ErlEstimator) Update(convergedFilters []bool, renderSpectra, captureSpectra [][FftLengthBy2Plus1]float32) {
	numCaptureChannels := len(convergedFilters)
	// Corresponds to WGN of power -46 dBFS.
	const x2Min float32 = 44015068.0
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

	var maxCaptureSpectrum [FftLengthBy2Plus1]float32
	if numCaptureChannels > 1 {
		maxCaptureSpectrum = captureSpectra[firstConverged]
		for ch := firstConverged + 1; ch < numCaptureChannels; ch++ {
			if !convergedFilters[ch] {
				continue
			}
			kernStdMax(maxCaptureSpectrum[:], captureSpectra[ch][:], maxCaptureSpectrum[:])
		}
	} else {
		maxCaptureSpectrum = captureSpectra[0]
	}

	var maxRenderSpectrum [FftLengthBy2Plus1]float32
	maxRenderSpectrum = renderSpectra[0]
	for ch := 1; ch < len(renderSpectra); ch++ {
		kernStdMax(maxRenderSpectrum[:], renderSpectra[ch][:], maxRenderSpectrum[:])
	}
	X2 := &maxRenderSpectrum
	Y2 := &maxCaptureSpectrum

	for k := 1; k < FftLengthBy2; k++ {
		if X2[k] > x2Min {
			newErl := Y2[k] / X2[k]
			if newErl < e.erl[k] {
				e.holdCounters[k-1] = 1000
				e.erl[k] += float32(0.1 * (newErl - e.erl[k]))
				e.erl[k] = stdMax(e.erl[k], stMinErl)
			}
		}
	}
	for i := range e.holdCounters {
		e.holdCounters[i]--
	}
	for i, a := range e.holdCounters {
		if a <= 0 {
			e.erl[i+1] = stdMin(stMaxErl, 2*e.erl[i+1])
		}
	}
	e.erl[0] = e.erl[1]
	e.erl[FftLengthBy2] = e.erl[FftLengthBy2-1]

	x2Sum := stAccumulate(X2[:])
	if x2Sum > x2Min*float32(len(X2)) {
		y2Sum := stAccumulate(Y2[:])
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
