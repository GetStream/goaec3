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
	v := stdMin(gain[1], gain[2])
	gain[0] = v
	gain[1] = v
}

// sgLimitHighFrequencyGains limits the high frequency gains to avoid echo
// leakage due to an imperfect filter.
func sgLimitHighFrequencyGains(conservativeHfSuppression bool, gain *[FftLengthBy2Plus1]float32) {
	const firstBandToLimit = (64 * 2000) / 8000
	minUpperGain := gain[firstBandToLimit]
	kernStdMinScalar(gain[firstBandToLimit+1:], minUpperGain)
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
		kernStdMinScalar(gain[upperAccurateBandPlus1:], hfGainBound)
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

// sgBandEnergies returns the largest sum of squares over the channels of
// band 0 of render, and over the channels of its upper bands (render has at
// least two bands). The C++ folds std::max(energy, sum of squares) from +0
// over the channels of band 0, and over the channels of band 1 and then of
// band 2. A sum of squares is NaN, which the fold skips, or at least +0, so
// each fold is the largest non-NaN sum (or +0) in any order of the sums; the
// sums of the bands of a channel are accumulated side by side.
func sgBandEnergies(render *Block) (lowBandEnergy, highBandEnergy float32) {
	for ch := 0; ch < render.NumChannels(); ch++ {
		b0 := (*[BlockSize]float32)(render.View(0, ch))
		b1 := (*[BlockSize]float32)(render.View(1, ch))
		b2 := b1
		if render.NumBands() > 2 {
			b2 = (*[BlockSize]float32)(render.View(2, ch))
		}
		e0, e1, e2 := kernSumSq3(b0, b1, b2)
		lowBandEnergy = stdMax(lowBandEnergy, e0)
		highBandEnergy = stdMax(stdMax(highBandEnergy, e1), e2)
	}
	return lowBandEnergy, highBandEnergy
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
		x2Sum, x2Max = kernSumMaxSq((*[BlockSize]float32)(render.View(0, ch)), x2Sum, x2Max)
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
	*suppressionGainState
	// The nearend detector: subbandNearendDetector with
	// Suppressor.UseSubbandNearendDetection, dominantNearendDetector (the
	// default, part of the SuppressionGain) otherwise.
	subbandNearendDetector  *SubbandNearendDetector
	more                    *sgMore // see sgMore
	dominantNearendDetector DominantNearendDetector
}

// suppressionGainState is the state of a SuppressionGain but its pointers.
type suppressionGainState struct {
	dominantNearendDetector dominantNearendState

	optimization              Aec3Optimization
	config                    EchoCanceller3Config
	numCaptureChannels        int
	nearendMemorySize         int // samples per channel
	stateChangeDurationBlocks int
	lastGain                  [FftLengthBy2Plus1]float32
	channels                  [inlineChannels]sgChannel
	nearendMemory             [inlineChannels][sgNearendMemoryInline]float32 // of the nearend smoothers
	lowRenderDetector         sgLowNoiseRenderDetector
	initialState              bool
	initialStateChangeCounter int
	nearendParams             sgGainParameters
	normalParams              sgGainParameters
	useUnboundedEchoSpectrum  bool
}

// sgNearendMemoryInline is the memory of the default nearend smoothing of
// one channel: FftLengthBy2Plus1 bins over 4 blocks.
const sgNearendMemoryInline = FftLengthBy2Plus1 * 3

// sgChannel is the state of one capture channel of a SuppressionGain.
type sgChannel struct {
	lastNearend     [FftLengthBy2Plus1]float32
	lastEcho        [FftLengthBy2Plus1]float32
	nearendSmoother movingAverageState
}

// sgMore is the state of a SuppressionGain that does not fit in it: the
// channel states with more than inlineChannels capture channels, and the
// memory of the nearend smoothers ([channel][nearendMemorySize]) with more
// channels or a longer smoothing than the default.
type sgMore struct {
	channels      []sgChannel
	nearendMemory []float32
}

// bind makes s the view of st.
func (s *SuppressionGain) bind(st *suppressionGainState, b *binder) {
	s.suppressionGainState = st
	if b != nil {
		bindPtr(b, &s.subbandNearendDetector, &b.x.sgSubbandNearendDetector)
		bindPtr(b, &s.more, &b.x.sgMore)
	}
	s.dominantNearendDetector.bind(&st.dominantNearendDetector, b)
}

// NewSuppressionGain mirrors the C++ constructor. sampleRateHz is unused, as
// in the C++.
func NewSuppressionGain(config *EchoCanceller3Config, optimization Aec3Optimization, sampleRateHz int,
	numCaptureChannels int) *SuppressionGain {
	s := &SuppressionGain{}
	s.bind(&suppressionGainState{}, nil)
	s.init(config, optimization, sampleRateHz, numCaptureChannels)
	return s
}

// init sets up s in place.
func (s *SuppressionGain) init(config *EchoCanceller3Config, optimization Aec3Optimization, sampleRateHz int,
	numCaptureChannels int) {
	_ = sampleRateHz
	st := s.suppressionGainState
	*st = suppressionGainState{}
	*s = SuppressionGain{}
	s.bind(st, nil)
	s.optimization = optimization
	s.config = *config
	s.numCaptureChannels = numCaptureChannels
	s.stateChangeDurationBlocks = config.Filter.ConfigChangeDurationBlocks
	s.lowRenderDetector = sgLowNoiseRenderDetector{averagePower: 32768 * 32768}
	s.initialState = true
	s.nearendParams = newSgGainParameters(config.Suppressor.LastLfBand, config.Suppressor.FirstHfBand,
		&config.Suppressor.NearendTuning)
	s.normalParams = newSgGainParameters(config.Suppressor.LastLfBand, config.Suppressor.FirstHfBand,
		&config.Suppressor.NormalTuning)
	s.useUnboundedEchoSpectrum = config.Suppressor.DominantNearendDetection.UseUnboundedEchoSpectrum
	var smoother movingAverageState
	memorySize := smoother.init(FftLengthBy2Plus1, config.Suppressor.NearendAverageBlocks)
	s.nearendMemorySize = memorySize
	if numCaptureChannels > inlineChannelLimit || memorySize > sgNearendMemoryInline {
		s.more = &sgMore{
			channels:      moreChannels[sgChannel](numCaptureChannels),
			nearendMemory: make([]float32, numCaptureChannels*memorySize),
		}
	}
	channels := s.channelStates()
	for ch := range channels {
		channels[ch].nearendSmoother = smoother
	}
	for k := range s.lastGain {
		s.lastGain[k] = 1
	}
	if config.Suppressor.UseSubbandNearendDetection {
		s.subbandNearendDetector = NewSubbandNearendDetector(&config.Suppressor.SubbandNearendDetection,
			numCaptureChannels)
	} else {
		s.dominantNearendDetector.init(&config.Suppressor.DominantNearendDetection, numCaptureChannels)
	}
}

// channelStates returns the state of every capture channel.
func (s *SuppressionGain) channelStates() []sgChannel {
	var more []sgChannel
	if s.more != nil {
		more = s.more.channels
	}
	return channelState(&s.channels, s.numCaptureChannels, more)
}

// smootherMemory returns the memory of the nearend smoother of a channel.
func (s *SuppressionGain) smootherMemory(ch int) []float32 {
	n := s.nearendMemorySize
	if s.more != nil {
		return s.more.nearendMemory[ch*n : (ch+1)*n]
	}
	return s.nearendMemory[ch][:n]
}

// IsDominantNearend returns whether the nearend detector is in the nearend
// state.
func (s *SuppressionGain) IsDominantNearend() bool {
	if s.subbandNearendDetector != nil {
		return s.subbandNearendDetector.IsNearendState()
	}
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
	if s.subbandNearendDetector != nil {
		s.subbandNearendDetector.Update(nearendSpectrum, echo, comfortNoiseSpectrum, s.initialState)
	} else {
		s.dominantNearendDetector.Update(nearendSpectrum, echo, comfortNoiseSpectrum, s.initialState)
	}

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
		return stdMin(0.001, gainBelow8Khz)
	}

	// Compute the upper and lower band energies.
	lowBandEnergy, highBandEnergy := sgBandEnergies(render)

	// If there is more power in the lower frequencies than the upper
	// frequencies, or if the power in upper frequencies is low, do not bound
	// the gain in the upper bands.
	hb := &s.config.Suppressor.HighBandsSuppression
	var antiHowlingGain float32
	activationThreshold := BlockSize * hb.AntiHowlingActivationThreshold
	if highBandEnergy < stdMax(lowBandEnergy, activationThreshold) {
		antiHowlingGain = 1
	} else {
		// In all other cases, bound the gain for upper frequencies.
		antiHowlingGain = hb.AntiHowlingGain * float32(math.Sqrt(float64(lowBandEnergy/highBandEnergy)))
	}

	gainBound := float32(1)
	if !s.IsDominantNearend() {
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
	return stdMin(stdMin(gainBelow8Khz, antiHowlingGain), gainBound)
}

func (s *SuppressionGain) params() *sgGainParameters {
	if s.IsDominantNearend() {
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
				minGain[k] = stdMax(minGain[k], s.lastGain[k]*dec)
				minGain[k] = stdMin(minGain[k], 1)
			}
		}
	}
}

// GetMaxGain computes the maximum gain by limiting the gain increase from the
// previous gain.
func (s *SuppressionGain) GetMaxGain(maxGain *[FftLengthBy2Plus1]float32) {
	inc := s.params().maxIncFactor
	floor := s.config.Suppressor.FloorFirstIncrease
	// maxGain = std::min(std::max(lastGain*inc, floor), 1.f).
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

	channels := s.channelStates()
	for ch := range channels {
		sc := &channels[ch]
		var G, nearend, weightedResidualEcho, minGain [FftLengthBy2Plus1]float32
		sc.nearendSmoother.average(s.smootherMemory(ch), suppressorInput[ch][:], nearend[:])

		// Weight echo power in terms of audibility.
		sgWeightEchoForAudibility(&s.config, &residualEcho[ch], &weightedResidualEcho)

		s.GetMinGain(&weightedResidualEcho, &sc.lastNearend, &sc.lastEcho, lowNoiseRender, saturatedEcho,
			&minGain)

		s.GainToNoAudibleEcho(&nearend, &weightedResidualEcho, &comfortNoise[0], &G)

		// Clamp gains.
		kernStdMin(G[:], maxGain[:], G[:])
		kernStdMax(G[:], minGain[:], G[:])
		kernStdMin(gain[:], G[:], gain[:])

		// Store data required for the gain computation of the next block.
		sc.lastNearend = nearend
		sc.lastEcho = weightedResidualEcho
	}

	sgLimitLowFrequencyGains(gain)
	// Use conservative high-frequency gains during clock-drift or when not in
	// dominant nearend.
	if !s.IsDominantNearend() || clockDrift || s.config.Suppressor.ConservativeHfSuppression {
		sgLimitHighFrequencyGains(s.config.Suppressor.ConservativeHfSuppression, gain)
	}

	// Store computed gains.
	s.lastGain = *gain

	// Transform gains to amplitude domain.
	NewVectorMath(s.optimization).Sqrt(gain[:])
}
