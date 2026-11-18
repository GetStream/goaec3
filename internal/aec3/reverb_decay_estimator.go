package aec3

import "math"

const (
	stEarlyReverbMinSizeBlocks = 3
	stBlocksPerSection         = 6
)

// stEarlyReverbFirstPointAtLinearRegressors: the linear regression assumes
// a symmetric index around 0 (-0.5f * 6 * 64 + 0.5f, exact in float).
const stEarlyReverbFirstPointAtLinearRegressors float32 = -191.5

// stBlockAverage averages the FftLengthBy2 values of block blockIndex.
func stBlockAverage(v []float32, blockIndex int) float32 {
	const oneByFftLengthBy2 float32 = 1.0 / FftLengthBy2
	i := blockIndex * FftLengthBy2
	return stAccumulate(v[i:i+FftLengthBy2]) * oneByFftLengthBy2
}

func stAnalyzeBlockGain(h2 *[FftLengthBy2]float32, floorGain float32, previousGain *float32) (blockAdapting, decayingGain bool) {
	gain := stdMax(stBlockAverage(h2[:], 0), 1e-32)
	blockAdapting = *previousGain > 1.1*gain || *previousGain < 0.9*gain
	decayingGain = gain > floorGain
	*previousGain = gain
	return
}

// stSymmetricArithmetricSum computes N * (N*N - 1.0f) * (1.f/12.f) in float.
func stSymmetricArithmetricSum(n int) float32 {
	nn := float32(n*n) - 1
	return float32(n) * nn * float32(1.0/12.0)
}

// stBlockEnergyPeak returns the peak energy of block peakBlock.
func stBlockEnergyPeak(h []float32, peakBlock int) float32 {
	s := h[peakBlock*FftLengthBy2 : (peakBlock+1)*FftLengthBy2]
	peak := s[0]
	for _, v := range s[1:] {
		if peak*peak < v*v {
			peak = v
		}
	}
	return peak * peak
}

// stBlockEnergyAverage returns the average energy of block blockIndex.
func stBlockEnergyAverage(h []float32, blockIndex int) float32 {
	const oneByFftLengthBy2 float32 = 1.0 / FftLengthBy2
	var acc float32
	for _, b := range h[blockIndex*FftLengthBy2 : (blockIndex+1)*FftLengthBy2] {
		acc += float32(b * b)
	}
	return acc * oneByFftLengthBy2
}

// lateReverbLinearRegressor estimates the decay of the late reverb.
type lateReverbLinearRegressor struct {
	nz    float32
	nn    float32
	count float32
	N     int
	n     int
}

func (l *lateReverbLinearRegressor) Reset(numDataPoints int) {
	N := numDataPoints
	l.nz = 0
	l.nn = stSymmetricArithmetricSum(N)
	if N > 0 {
		l.count = float32(float32(-N)*0.5) + 0.5
	} else {
		l.count = 0
	}
	l.N = N
	l.n = 0
}

func (l *lateReverbLinearRegressor) Accumulate(z float32) {
	l.nz += float32(l.count * z)
	l.count++
	l.n++
}

func (l *lateReverbLinearRegressor) Estimate() float32 {
	if l.nn == 0 {
		return 0
	}
	return l.nz / l.nn
}

func (l *lateReverbLinearRegressor) EstimateAvailable() bool { return l.n == l.N && l.N != 0 }

// earlyReverbLengthEstimator identifies the length of the early reverb.
type earlyReverbLengthEstimator struct {
	numeratorsSmooth    []float32
	numerators          []float32
	coefficientsCounter int
	blockCounter        int
	nSections           int
}

func newEarlyReverbLengthEstimator(maxBlocks int) earlyReverbLengthEstimator {
	n := maxBlocks - stBlocksPerSection
	return earlyReverbLengthEstimator{numeratorsSmooth: make([]float32, n), numerators: make([]float32, n)}
}

func (e *earlyReverbLengthEstimator) Reset() {
	e.coefficientsCounter = 0
	clear(e.numerators)
	e.blockCounter = 0
}

func (e *earlyReverbLengthEstimator) Accumulate(value, smoothing float32) {
	firstSectionIndex := max(e.blockCounter-stBlocksPerSection+1, 0)
	lastSectionIndex := min(e.blockCounter, len(e.numerators)-1)
	xValue := float32(e.coefficientsCounter) + stEarlyReverbFirstPointAtLinearRegressors
	valueToInc := float32(FftLengthBy2 * value)
	valueToAdd := float32(xValue*value) + float32(float32(e.blockCounter-lastSectionIndex)*valueToInc)
	for section := lastSectionIndex; section >= firstSectionIndex; section, valueToAdd = section-1, valueToAdd+valueToInc {
		e.numerators[section] += valueToAdd
	}
	e.coefficientsCounter++
	if e.coefficientsCounter == FftLengthBy2 {
		if e.blockCounter >= stBlocksPerSection-1 {
			section := e.blockCounter - (stBlocksPerSection - 1)
			e.numeratorsSmooth[section] += float32(smoothing * (e.numerators[section] - e.numeratorsSmooth[section]))
			e.nSections = section + 1
		}
		e.blockCounter++
		e.coefficientsCounter = 0
	}
}

var (
	stEarlyNn          = stSymmetricArithmetricSum(stBlocksPerSection * FftLengthBy2)
	stEarlyNumerator11 = stMul(0.13750352374993502, stEarlyNn) / FftLengthBy2
	stEarlyNumerator08 = stMul(-0.32192809488736229, stEarlyNn) / FftLengthBy2
)

func (e *earlyReverbLengthEstimator) Estimate() int {
	const numSectionsToAnalyze = 9
	if e.nSections < numSectionsToAnalyze {
		return 0
	}
	var minNumeratorTail float32
	if e.nSections > numSectionsToAnalyze {
		tail := e.numeratorsSmooth[numSectionsToAnalyze:e.nSections]
		minNumeratorTail = tail[0]
		for _, v := range tail[1:] {
			if v < minNumeratorTail {
				minNumeratorTail = v
			}
		}
	} else if e.nSections < len(e.numeratorsSmooth) {
		// The C++ dereferences the end iterator of the empty range, which
		// reads the element at nSections.
		minNumeratorTail = e.numeratorsSmooth[e.nSections]
	}
	earlyReverbSizeMinus1 := 0
	for k := 0; k < numSectionsToAnalyze; k++ {
		if e.numeratorsSmooth[k] > stEarlyNumerator11 ||
			(e.numeratorsSmooth[k] < stEarlyNumerator08 && e.numeratorsSmooth[k] < 0.9*minNumeratorTail) {
			earlyReverbSizeMinus1 = k
		}
	}
	if earlyReverbSizeMinus1 == 0 {
		return 0
	}
	return earlyReverbSizeMinus1 + 1
}

// ReverbDecayEstimator mirrors reverb_decay_estimator.h: estimates the decay
// of the late reverb from the linear filter.
type ReverbDecayEstimator struct {
	filterLengthBlocks            int
	filterLengthCoefficients      int
	useAdaptiveEchoDecay          bool
	lateReverbDecayEstimator      lateReverbLinearRegressor
	earlyReverbEstimator          earlyReverbLengthEstimator
	lateReverbStart               int
	lateReverbEnd                 int
	blockToAnalyze                int
	estimationRegionCandidateSize int
	estimationRegionIdentified    bool
	previousGains                 []float32
	decay                         float32
	mildDecay                     float32
	tailGain                      float32
	smoothingConstant             float32
}

func NewReverbDecayEstimator(config *EchoCanceller3Config) *ReverbDecayEstimator {
	lb := config.Filter.Refined.LengthBlocks
	return &ReverbDecayEstimator{
		filterLengthBlocks:       lb,
		filterLengthCoefficients: GetTimeDomainLength(lb),
		useAdaptiveEchoDecay:     config.EpStrength.DefaultLen < 0,
		earlyReverbEstimator:     newEarlyReverbLengthEstimator(lb - stEarlyReverbMinSizeBlocks),
		lateReverbStart:          stEarlyReverbMinSizeBlocks,
		lateReverbEnd:            stEarlyReverbMinSizeBlocks,
		previousGains:            make([]float32, lb),
		decay:                    float32(math.Abs(float64(config.EpStrength.DefaultLen))),
		mildDecay:                float32(math.Abs(float64(config.EpStrength.NearendLen))),
	}
}

// Decay returns the decay of the exponential model; mild selects the milder
// decay used during nearend regions.
func (r *ReverbDecayEstimator) Decay(mild bool) float32 {
	if r.useAdaptiveEchoDecay {
		return r.decay
	}
	if mild {
		return r.mildDecay
	}
	return r.decay
}

// Update updates the decay estimate.
func (r *ReverbDecayEstimator) Update(filter []float32, filterQuality LinearFilterQuality, filterDelayBlocks int,
	usableLinearFilter, stationarySignal bool) {
	filterSize := len(filter)
	if stationarySignal {
		return
	}
	estimationFeasible := filterDelayBlocks <= r.filterLengthBlocks-stEarlyReverbMinSizeBlocks-1
	estimationFeasible = estimationFeasible && filterSize == r.filterLengthCoefficients
	estimationFeasible = estimationFeasible && filterDelayBlocks > 0
	estimationFeasible = estimationFeasible && usableLinearFilter
	if !estimationFeasible {
		r.resetDecayEstimation()
		return
	}
	if !r.useAdaptiveEchoDecay {
		return
	}
	var newSmoothing float32
	if filterQuality.Ok {
		newSmoothing = filterQuality.Value * 0.2
	}
	r.smoothingConstant = stdMax(newSmoothing, r.smoothingConstant)
	if r.smoothingConstant == 0 {
		return
	}
	if r.blockToAnalyze < r.filterLengthBlocks {
		r.analyzeFilter(filter)
		r.blockToAnalyze++
	} else {
		r.estimateDecay(filter, filterDelayBlocks)
	}
}

func (r *ReverbDecayEstimator) resetDecayEstimation() {
	r.earlyReverbEstimator.Reset()
	r.lateReverbDecayEstimator.Reset(0)
	r.blockToAnalyze = 0
	r.estimationRegionCandidateSize = 0
	r.estimationRegionIdentified = false
	r.smoothingConstant = 0
	r.lateReverbStart = 0
	r.lateReverbEnd = 0
}

func (r *ReverbDecayEstimator) estimateDecay(h []float32, peakBlock int) {
	r.blockToAnalyze = min(peakBlock+stEarlyReverbMinSizeBlocks, r.filterLengthBlocks)
	firstReverbGain := stBlockEnergyAverage(h, r.blockToAnalyze)
	hSizeBlocks := len(h) >> FftLengthBy2Log2
	r.tailGain = stBlockEnergyAverage(h, hSizeBlocks-1)
	peakEnergy := stBlockEnergyPeak(h, peakBlock)
	sufficientReverbDecay := firstReverbGain > float32(4*r.tailGain)
	validFilter := firstReverbGain > float32(2*r.tailGain) && peakEnergy < 100
	sizeEarlyReverb := r.earlyReverbEstimator.Estimate()
	sizeLateReverb := max(r.estimationRegionCandidateSize-sizeEarlyReverb, 0)
	if sizeLateReverb >= 5 {
		if validFilter && r.lateReverbDecayEstimator.EstimateAvailable() {
			decay := Exp2f(r.lateReverbDecayEstimator.Estimate() * FftLengthBy2)
			const maxDecay = 0.95
			const minDecay = 0.02
			decay = stdMax(0.97*r.decay, decay)
			decay = stdMin(decay, maxDecay)
			decay = stdMax(decay, minDecay)
			r.decay += float32(r.smoothingConstant * (decay - r.decay))
		}
		r.lateReverbDecayEstimator.Reset(sizeLateReverb * FftLengthBy2)
		r.lateReverbStart = peakBlock + stEarlyReverbMinSizeBlocks + sizeEarlyReverb
		r.lateReverbEnd = r.blockToAnalyze + r.estimationRegionCandidateSize - 1
	} else {
		r.lateReverbDecayEstimator.Reset(0)
		r.lateReverbStart = 0
		r.lateReverbEnd = 0
	}
	r.estimationRegionIdentified = !(validFilter && sufficientReverbDecay)
	r.estimationRegionCandidateSize = 0
	r.smoothingConstant = 0
	r.earlyReverbEstimator.Reset()
}

func (r *ReverbDecayEstimator) analyzeFilter(filter []float32) {
	h := filter[r.blockToAnalyze*FftLengthBy2 : (r.blockToAnalyze+1)*FftLengthBy2]
	var h2 [FftLengthBy2]float32
	for i, a := range h {
		h2[i] = a * a
	}
	adapting, aboveNoiseFloor := stAnalyzeBlockGain(&h2, r.tailGain, &r.previousGains[r.blockToAnalyze])
	r.estimationRegionIdentified = r.estimationRegionIdentified || adapting || !aboveNoiseFloor
	if !r.estimationRegionIdentified {
		r.estimationRegionCandidateSize++
	}
	if r.blockToAnalyze <= r.lateReverbEnd {
		if r.blockToAnalyze >= r.lateReverbStart {
			for _, h2k := range h2 {
				h2Log2 := FastApproxLog2f(float32(float64(h2k) + 1e-10))
				r.lateReverbDecayEstimator.Accumulate(h2Log2)
				r.earlyReverbEstimator.Accumulate(h2Log2, r.smoothingConstant)
			}
		} else {
			for _, h2k := range h2 {
				h2Log2 := FastApproxLog2f(float32(float64(h2k) + 1e-10))
				r.earlyReverbEstimator.Accumulate(h2Log2, r.smoothingConstant)
			}
		}
	}
}
