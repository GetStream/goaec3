package aec3

import "math"

// Port of suppression_gain.{h,cc}.
//
// The only optimization-dependent operation is VectorMath::Sqrt, which is
// correctly rounded on every lane (sqrtf, _mm_sqrt_ps, _mm256_sqrt_ps and
// vsqrtq_f32 on arm64), so the lanes share one Go implementation.

// sgLimitLowFrequencyGains limits the low frequency gains to avoid the impact
// of the high-pass filter on the lower-frequency gain influencing the overall
// achieved gain.
func sgLimitLowFrequencyGains(gain *[FftLengthBy2Plus1]float32) {
	v := min(gain[1], gain[2])
	gain[0] = v
	gain[1] = v
}

// sgLimitHighFrequencyGains limits the high frequency gains to avoid echo
// leakage due to an imperfect filter.
func sgLimitHighFrequencyGains(conservativeHfSuppression bool, gain *[FftLengthBy2Plus1]float32) {
	const firstBandToLimit = (64 * 2000) / 8000
	minUpperGain := gain[firstBandToLimit]
	for k := firstBandToLimit + 1; k < FftLengthBy2Plus1; k++ {
		gain[k] = min(gain[k], minUpperGain)
	}
	gain[FftLengthBy2] = gain[FftLengthBy2Minus1]

	if conservativeHfSuppression {
		// Limits the gain in the frequencies for which the adaptive filter has
		// not converged.
		const upperAccurateBandPlus1 = 29
		const oneByBandsInSum = float32(1) / float32(upperAccurateBandPlus1-20)
		var sum float32
		for _, g := range gain[20:upperAccurateBandPlus1] {
			sum += g
		}
		hfGainBound := sum * oneByBandsInSum
		for k := upperAccurateBandPlus1; k < FftLengthBy2Plus1; k++ {
			gain[k] = min(gain[k], hfGainBound)
		}
	}
}

func sgWeigh(threshold, normalizer float32, begin, end int, echo, weightedEcho *[FftLengthBy2Plus1]float32) {
	kernWeigh(threshold, normalizer, echo[begin:end], weightedEcho[begin:end])
}

// sgWeightEchoForAudibility scales the echo according to assessed audibility
// at the other end.
func sgWeightEchoForAudibility(config *EchoCanceller3Config, echo, weightedEcho *[FftLengthBy2Plus1]float32) {
	ea := &config.EchoAudibility
	threshold := float32(ea.FloorPower * ea.AudibilityThresholdLf)
	normalizer := 1 / (threshold - ea.FloorPower)
	sgWeigh(threshold, normalizer, 0, 3, echo, weightedEcho)

	threshold = float32(ea.FloorPower * ea.AudibilityThresholdMf)
	normalizer = 1 / (threshold - ea.FloorPower)
	sgWeigh(threshold, normalizer, 3, 7, echo, weightedEcho)

	threshold = float32(ea.FloorPower * ea.AudibilityThresholdHf)
	normalizer = 1 / (threshold - ea.FloorPower)
	sgWeigh(threshold, normalizer, 7, FftLengthBy2Plus1, echo, weightedEcho)
}

func sgLowFrequencyEnergy(spectrum *[FftLengthBy2Plus1]float32) float32 {
	var sum float32
	for _, v := range spectrum[1:16] {
		sum += v
	}
	return sum
}

func sgSumOfSquares(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += float32(v * v)
	}
	return sum
}

// sgLowNoiseRenderDetector mirrors SuppressionGain::LowNoiseRenderDetector.
type sgLowNoiseRenderDetector struct {
	averagePower float32
}

// Detect detects when the render signal can be considered to have low power
// and consist of stationary noise.
func (d *sgLowNoiseRenderDetector) Detect(render *Block) bool {
	var x2Sum, x2Max float32
	for ch := 0; ch < render.NumChannels(); ch++ {
		for _, x := range render.View(0, ch) {
			x2 := float32(x * x)
			x2Sum += x2
			x2Max = max(x2Max, x2)
		}
	}
	x2Sum = x2Sum / float32(render.NumChannels())

	const threshold = 50 * 50 * 64
	lowNoiseRender := d.averagePower < threshold && x2Max < 3*d.averagePower
	d.averagePower = float32(d.averagePower*0.9) + float32(x2Sum*0.1)
	return lowNoiseRender
}

// sgGainParameters mirrors SuppressionGain::GainParameters.
type sgGainParameters struct {
	maxIncFactor   float32
	maxDecFactorLf float32
	enrTransparent [FftLengthBy2Plus1]float32
	enrSuppress    [FftLengthBy2Plus1]float32
	emrTransparent [FftLengthBy2Plus1]float32
}

func newSgGainParameters(lastLfBand, firstHfBand int, tuning *Tuning) sgGainParameters {
	p := sgGainParameters{maxIncFactor: tuning.MaxIncFactor, maxDecFactorLf: tuning.MaxDecFactorLf}
	// Compute per-band masking thresholds.
	lf := &tuning.MaskLf
	hf := &tuning.MaskHf
	for k := 0; k < FftLengthBy2Plus1; k++ {
		var a float32
		if k <= lastLfBand {
			a = 0
		} else if k < firstHfBand {
			a = float32(k-lastLfBand) / float32(firstHfBand-lastLfBand)
		} else {
			a = 1
		}
		p.enrTransparent[k] = float32((1-a)*lf.EnrTransparent) + float32(a*hf.EnrTransparent)
		p.enrSuppress[k] = float32((1-a)*lf.EnrSuppress) + float32(a*hf.EnrSuppress)
		p.emrTransparent[k] = float32((1-a)*lf.EmrTransparent) + float32(a*hf.EmrTransparent)
	}
	return p
}

// SuppressionGain computes the echo suppression gains.
type SuppressionGain struct {
	optimization              Aec3Optimization
	config                    EchoCanceller3Config
	numCaptureChannels        int
	stateChangeDurationBlocks int
	lastGain                  [FftLengthBy2Plus1]float32
	lastNearend               [][FftLengthBy2Plus1]float32
	lastEcho                  [][FftLengthBy2Plus1]float32
	lowRenderDetector         sgLowNoiseRenderDetector
	initialState              bool
	initialStateChangeCounter int
	nearendSmoothers          []*MovingAverage
	nearendParams             sgGainParameters
	normalParams              sgGainParameters
	useUnboundedEchoSpectrum  bool
	dominantNearendDetector   NearendDetector
}

// NewSuppressionGain mirrors the C++ constructor. sampleRateHz is unused, as
// in the C++.
func NewSuppressionGain(config *EchoCanceller3Config, optimization Aec3Optimization, sampleRateHz int,
	numCaptureChannels int) *SuppressionGain {
	_ = sampleRateHz
	s := &SuppressionGain{
		optimization:              optimization,
		config:                    *config,
		numCaptureChannels:        numCaptureChannels,
		stateChangeDurationBlocks: config.Filter.ConfigChangeDurationBlocks,
		lastNearend:               make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		lastEcho:                  make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		lowRenderDetector:         sgLowNoiseRenderDetector{averagePower: 32768 * 32768},
		initialState:              true,
		nearendSmoothers:          make([]*MovingAverage, numCaptureChannels),
		nearendParams: newSgGainParameters(config.Suppressor.LastLfBand, config.Suppressor.FirstHfBand,
			&config.Suppressor.NearendTuning),
		normalParams: newSgGainParameters(config.Suppressor.LastLfBand, config.Suppressor.FirstHfBand,
			&config.Suppressor.NormalTuning),
		useUnboundedEchoSpectrum: config.Suppressor.DominantNearendDetection.UseUnboundedEchoSpectrum,
	}
	for ch := range s.nearendSmoothers {
		s.nearendSmoothers[ch] = NewMovingAverage(FftLengthBy2Plus1, config.Suppressor.NearendAverageBlocks)
	}
	for k := range s.lastGain {
		s.lastGain[k] = 1
	}
	if config.Suppressor.UseSubbandNearendDetection {
		s.dominantNearendDetector = NewSubbandNearendDetector(&config.Suppressor.SubbandNearendDetection,
			numCaptureChannels)
	} else {
		s.dominantNearendDetector = NewDominantNearendDetector(&config.Suppressor.DominantNearendDetection,
			numCaptureChannels)
	}
	return s
}

// IsDominantNearend returns whether the nearend detector is in the nearend
// state.
func (s *SuppressionGain) IsDominantNearend() bool {
	return s.dominantNearendDetector.IsNearendState()
}

// SetInitialState toggles the usage of the initial state.
func (s *SuppressionGain) SetInitialState(state bool) {
	s.initialState = state
	if state {
		s.initialStateChangeCounter = s.stateChangeDurationBlocks
	} else {
		s.initialStateChangeCounter = 0
	}
}

// GetGain computes the low band gain (amplitude domain) and the gain for the
// bands above the first band.
func (s *SuppressionGain) GetGain(nearendSpectrum, echoSpectrum, residualEchoSpectrum,
	residualEchoSpectrumUnbounded, comfortNoiseSpectrum [][FftLengthBy2Plus1]float32,
	renderSignalAnalyzer *RenderSignalAnalyzer, aecState *AecState, render *Block, clockDrift bool,
	highBandsGain *float32, lowBandGain *[FftLengthBy2Plus1]float32) {
	// Choose residual echo spectrum for dominant nearend detection.
	echo := residualEchoSpectrum
	if s.useUnboundedEchoSpectrum {
		echo = residualEchoSpectrumUnbounded
	}

	// Update the nearend state selection.
	s.dominantNearendDetector.Update(nearendSpectrum, echo, comfortNoiseSpectrum, s.initialState)

	// Compute gain for the lower band.
	lowNoiseRender := s.lowRenderDetector.Detect(render)
	s.LowerBandGain(lowNoiseRender, aecState, nearendSpectrum, residualEchoSpectrum, comfortNoiseSpectrum,
		clockDrift, lowBandGain)

	// Compute the gain for the upper bands.
	narrowPeakBand, hasNarrowPeakBand := renderSignalAnalyzer.NarrowPeakBand()

	*highBandsGain = s.UpperBandsGain(echoSpectrum, comfortNoiseSpectrum, narrowPeakBand, hasNarrowPeakBand,
		aecState.SaturatedEcho(), render, lowBandGain)
}

// UpperBandsGain computes the gain to apply for the bands beyond the first
// band.
func (s *SuppressionGain) UpperBandsGain(echoSpectrum, comfortNoiseSpectrum [][FftLengthBy2Plus1]float32,
	narrowPeakBand int, hasNarrowPeakBand bool, saturatedEcho bool, render *Block,
	lowBandGain *[FftLengthBy2Plus1]float32) float32 {
	if render.NumBands() == 1 {
		return 1
	}
	numRenderChannels := render.NumChannels()

	if hasNarrowPeakBand && narrowPeakBand > FftLengthBy2Plus1-10 {
		return 0.001
	}

	const lowBandGainLimit = FftLengthBy2 / 2
	gainBelow8Khz := lowBandGain[lowBandGainLimit]
	for _, g := range lowBandGain[lowBandGainLimit+1:] {
		if g < gainBelow8Khz {
			gainBelow8Khz = g
		}
	}

	// Always attenuate the upper bands when there is saturated echo.
	if saturatedEcho {
		return min(0.001, gainBelow8Khz)
	}

	// Compute the upper and lower band energies.
	var lowBandEnergy float32
	for ch := 0; ch < numRenderChannels; ch++ {
		lowBandEnergy = max(lowBandEnergy, sgSumOfSquares(render.View(0, ch)))
	}
	var highBandEnergy float32
	for k := 1; k < render.NumBands(); k++ {
		for ch := 0; ch < numRenderChannels; ch++ {
			highBandEnergy = max(highBandEnergy, sgSumOfSquares(render.View(k, ch)))
		}
	}

	// If there is more power in the lower frequencies than the upper
	// frequencies, or if the power in upper frequencies is low, do not bound
	// the gain in the upper bands.
	hb := &s.config.Suppressor.HighBandsSuppression
	var antiHowlingGain float32
	activationThreshold := BlockSize * hb.AntiHowlingActivationThreshold
	if highBandEnergy < max(lowBandEnergy, activationThreshold) {
		antiHowlingGain = 1
	} else {
		// In all other cases, bound the gain for upper frequencies.
		antiHowlingGain = hb.AntiHowlingGain * float32(math.Sqrt(float64(lowBandEnergy/highBandEnergy)))
	}

	gainBound := float32(1)
	if !s.dominantNearendDetector.IsNearendState() {
		// Bound the upper gain during significant echo activity.
		for ch := 0; ch < s.numCaptureChannels; ch++ {
			echoSum := sgLowFrequencyEnergy(&echoSpectrum[ch])
			noiseSum := sgLowFrequencyEnergy(&comfortNoiseSpectrum[ch])
			if echoSum > hb.EnrThreshold*noiseSum {
				gainBound = hb.MaxGainDuringEcho
				break
			}
		}
	}

	// Choose the gain as the minimum of the lower and upper gains.
	return min(min(gainBelow8Khz, antiHowlingGain), gainBound)
}

func (s *SuppressionGain) params() *sgGainParameters {
	if s.dominantNearendDetector.IsNearendState() {
		return &s.nearendParams
	}
	return &s.normalParams
}

// GainToNoAudibleEcho computes the gain to reduce the echo to a non audible
// level.
func (s *SuppressionGain) GainToNoAudibleEcho(nearend, echo, masker, gain *[FftLengthBy2Plus1]float32) {
	p := s.params()
	kernNoAudibleEcho(nearend, echo, masker, &p.enrTransparent, &p.enrSuppress, &p.emrTransparent, gain)
}

// GetMinGain computes the minimum gain as the attenuating gain to put the
// signal just above the zero sample values.
func (s *SuppressionGain) GetMinGain(weightedResidualEcho, lastNearend, lastEcho *[FftLengthBy2Plus1]float32,
	lowNoiseRender, saturatedEcho bool, minGain *[FftLengthBy2Plus1]float32) {
	if saturatedEcho {
		*minGain = [FftLengthBy2Plus1]float32{}
		return
	}
	minEchoPower := s.config.EchoAudibility.NormalRenderLimit
	if lowNoiseRender {
		minEchoPower = s.config.EchoAudibility.LowRenderLimit
	}
	kernMinGain(weightedResidualEcho, minEchoPower, minGain)

	sup := &s.config.Suppressor
	if !s.initialState || sup.LfSmoothingDuringInitialPhase {
		dec := s.params().maxDecFactorLf
		for k := 0; k <= sup.LastLfSmoothingBand; k++ {
			// Make sure the gains of the low frequencies do not decrease too
			// quickly after strong nearend.
			if lastNearend[k] > lastEcho[k] || k <= sup.LastPermanentLfSmoothingBand {
				minGain[k] = max(minGain[k], s.lastGain[k]*dec)
				minGain[k] = min(minGain[k], 1)
			}
		}
	}
}

// GetMaxGain computes the maximum gain by limiting the gain increase from the
// previous gain.
func (s *SuppressionGain) GetMaxGain(maxGain *[FftLengthBy2Plus1]float32) {
	inc := s.params().maxIncFactor
	floor := s.config.Suppressor.FloorFirstIncrease
	// maxGain = min(max(lastGain*inc, floor), 1).
	kernScale(s.lastGain[:], inc, maxGain[:])
	kernClamp(maxGain[:], floor, 1)
}

// LowerBandGain computes the gain for the first band.
func (s *SuppressionGain) LowerBandGain(lowNoiseRender bool, aecState *AecState,
	suppressorInput, residualEcho, comfortNoise [][FftLengthBy2Plus1]float32, clockDrift bool,
	gain *[FftLengthBy2Plus1]float32) {
	for k := range gain {
		gain[k] = 1
	}
	saturatedEcho := aecState.SaturatedEcho()
	var maxGain [FftLengthBy2Plus1]float32
	s.GetMaxGain(&maxGain)

	for ch := 0; ch < s.numCaptureChannels; ch++ {
		var G, nearend, weightedResidualEcho, minGain [FftLengthBy2Plus1]float32
		s.nearendSmoothers[ch].Average(suppressorInput[ch][:], nearend[:])

		// Weight echo power in terms of audibility.
		sgWeightEchoForAudibility(&s.config, &residualEcho[ch], &weightedResidualEcho)

		s.GetMinGain(&weightedResidualEcho, &s.lastNearend[ch], &s.lastEcho[ch], lowNoiseRender, saturatedEcho,
			&minGain)

		s.GainToNoAudibleEcho(&nearend, &weightedResidualEcho, &comfortNoise[0], &G)

		// Clamp gains.
		kernMinGo(G[:], maxGain[:], G[:])
		kernMaxGo(G[:], minGain[:], G[:])
		kernMinGo(gain[:], G[:], gain[:])

		// Store data required for the gain computation of the next block.
		s.lastNearend[ch] = nearend
		s.lastEcho[ch] = weightedResidualEcho
	}

	sgLimitLowFrequencyGains(gain)
	// Use conservative high-frequency gains during clock-drift or when not in
	// dominant nearend.
	if !s.dominantNearendDetector.IsNearendState() || clockDrift || s.config.Suppressor.ConservativeHfSuppression {
		sgLimitHighFrequencyGains(s.config.Suppressor.ConservativeHfSuppression, gain)
	}

	// Store computed gains.
	s.lastGain = *gain

	// Transform gains to amplitude domain.
	NewVectorMath(s.optimization).Sqrt(gain[:])
}
