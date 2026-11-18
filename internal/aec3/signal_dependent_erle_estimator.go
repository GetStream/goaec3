package aec3

// SignalDependentErleEstimatorSubbands is SignalDependentErleEstimator::kSubbands.
const SignalDependentErleEstimatorSubbands = 6

var stBandBoundaries = [SignalDependentErleEstimatorSubbands + 1]int{1, 8, 16, 24, 32, 48, FftLengthBy2Plus1}

func stFormSubbandMap() [FftLengthBy2Plus1]int {
	var m [FftLengthBy2Plus1]int
	subband := 1
	for k := range m {
		if k >= stBandBoundaries[subband] {
			subband++
		}
		m[k] = subband - 1
	}
	return m
}

// stDefineFilterSectionSizes splits the filter in sections whose sizes grow
// with the distance to the direct path.
func stDefineFilterSectionSizes(delayHeadroomBlocks, numBlocks, numSections int) []int {
	filterLengthBlocks := numBlocks - delayHeadroomBlocks
	sectionSizes := make([]int, numSections)
	remainingBlocks := filterLengthBlocks
	remainingSections := numSections
	estimatorSize := 2
	idx := 0
	for remainingSections > 1 && remainingBlocks > estimatorSize*remainingSections {
		sectionSizes[idx] = estimatorSize
		remainingBlocks -= estimatorSize
		remainingSections--
		estimatorSize *= 2
		idx++
	}
	lastGroupsSize := remainingBlocks / remainingSections
	for ; idx < numSections; idx++ {
		sectionSizes[idx] = lastGroupsSize
	}
	sectionSizes[numSections-1] += remainingBlocks - lastGroupsSize*remainingSections
	return sectionSizes
}

func stSetSectionsBoundaries(delayHeadroomBlocks, numBlocks, numSections int) []int {
	b := make([]int, numSections+1)
	if len(b) == 2 {
		b[0] = 0
		b[1] = numBlocks
		return b
	}
	sectionSizes := stDefineFilterSectionSizes(delayHeadroomBlocks, numBlocks, len(b)-1)
	idx := 0
	currentSizeBlock := 0
	b[0] = delayHeadroomBlocks
	for k := delayHeadroomBlocks; k < numBlocks; k++ {
		currentSizeBlock++
		if currentSizeBlock >= sectionSizes[idx] {
			idx = idx + 1
			if idx == len(sectionSizes) {
				break
			}
			b[idx] = k + 1
			currentSizeBlock = 0
		}
	}
	b[len(sectionSizes)] = numBlocks
	return b
}

// SignalDependentErleEstimator mirrors signal_dependent_erle_estimator.h:
// refines the ERLE with a correction factor that depends on how many filter
// sections carry the echo energy.
type SignalDependentErleEstimator struct {
	minErle                 float32
	numSections             int
	numBlocks               int
	delayHeadroomBlocks     int
	bandToSubband           [FftLengthBy2Plus1]int
	maxErle                 [SignalDependentErleEstimatorSubbands]float32
	sectionBoundariesBlocks []int
	useOnsetDetection       bool
	erle                    [][FftLengthBy2Plus1]float32
	erleOnsetCompensated    [][FftLengthBy2Plus1]float32
	S2SectionAccum          [][][FftLengthBy2Plus1]float32
	erleEstimators          [][][SignalDependentErleEstimatorSubbands]float32
	erleRef                 [][SignalDependentErleEstimatorSubbands]float32
	correctionFactors       [][][SignalDependentErleEstimatorSubbands]float32
	numUpdates              [][SignalDependentErleEstimatorSubbands]int
	nActiveSections         [][FftLengthBy2Plus1]int
}

func NewSignalDependentErleEstimator(config *EchoCanceller3Config, numCaptureChannels int) *SignalDependentErleEstimator {
	s := &SignalDependentErleEstimator{
		minErle:             config.Erle.Min,
		numSections:         config.Erle.NumSections,
		numBlocks:           config.Filter.Refined.LengthBlocks,
		delayHeadroomBlocks: config.Delay.DelayHeadroomSamples / BlockSize,
		bandToSubband:       stFormSubbandMap(),
		useOnsetDetection:   config.Erle.OnsetDetection,
	}
	limitSubbandL := s.bandToSubband[FftLengthBy2/2]
	for i := range s.maxErle {
		if i < limitSubbandL {
			s.maxErle[i] = config.Erle.MaxL
		} else {
			s.maxErle[i] = config.Erle.MaxH
		}
	}
	s.sectionBoundariesBlocks = stSetSectionsBoundaries(s.delayHeadroomBlocks, s.numBlocks, s.numSections)
	s.erle = make([][FftLengthBy2Plus1]float32, numCaptureChannels)
	s.erleOnsetCompensated = make([][FftLengthBy2Plus1]float32, numCaptureChannels)
	s.S2SectionAccum = make([][][FftLengthBy2Plus1]float32, numCaptureChannels)
	s.erleEstimators = make([][][SignalDependentErleEstimatorSubbands]float32, numCaptureChannels)
	s.erleRef = make([][SignalDependentErleEstimatorSubbands]float32, numCaptureChannels)
	s.correctionFactors = make([][][SignalDependentErleEstimatorSubbands]float32, numCaptureChannels)
	s.numUpdates = make([][SignalDependentErleEstimatorSubbands]int, numCaptureChannels)
	s.nActiveSections = make([][FftLengthBy2Plus1]int, numCaptureChannels)
	for ch := 0; ch < numCaptureChannels; ch++ {
		s.S2SectionAccum[ch] = make([][FftLengthBy2Plus1]float32, s.numSections)
		s.erleEstimators[ch] = make([][SignalDependentErleEstimatorSubbands]float32, s.numSections)
		s.correctionFactors[ch] = make([][SignalDependentErleEstimatorSubbands]float32, s.numSections)
	}
	s.Reset()
	return s
}

func (s *SignalDependentErleEstimator) Reset() {
	for ch := range s.erle {
		for k := range s.erle[ch] {
			s.erle[ch][k] = s.minErle
			s.erleOnsetCompensated[ch][k] = s.minErle
		}
		for i := range s.erleEstimators[ch] {
			for j := range s.erleEstimators[ch][i] {
				s.erleEstimators[ch][i][j] = s.minErle
			}
		}
		for j := range s.erleRef[ch] {
			s.erleRef[ch][j] = s.minErle
		}
		for i := range s.correctionFactors[ch] {
			for j := range s.correctionFactors[ch][i] {
				s.correctionFactors[ch][i][j] = 1
			}
		}
		s.numUpdates[ch] = [SignalDependentErleEstimatorSubbands]int{}
		s.nActiveSections[ch] = [FftLengthBy2Plus1]int{}
	}
}

// Erle returns the ERLE, onset compensated when requested and enabled.
func (s *SignalDependentErleEstimator) Erle(onsetCompensated bool) [][FftLengthBy2Plus1]float32 {
	if onsetCompensated && s.useOnsetDetection {
		return s.erleOnsetCompensated
	}
	return s.erle
}

// Update refines averageErle and averageErleOnsetCompensated with the
// signal dependent correction factors.
func (s *SignalDependentErleEstimator) Update(renderBuffer *RenderBuffer, filterFrequencyResponses [][][FftLengthBy2Plus1]float32,
	X2 *[FftLengthBy2Plus1]float32, Y2, E2, averageErle, averageErleOnsetCompensated [][FftLengthBy2Plus1]float32,
	convergedFilters []bool) {
	s.computeEchoEstimatePerFilterSection(renderBuffer, filterFrequencyResponses)
	s.computeActiveFilterSections()
	s.updateCorrectionFactors(X2, Y2, E2, convergedFilters)
	for ch := range s.erle {
		for k := 0; k < FftLengthBy2; k++ {
			correctionFactor := s.correctionFactors[ch][s.nActiveSections[ch][k]][s.bandToSubband[k]]
			s.erle[ch][k] = stSafeClamp(averageErle[ch][k]*correctionFactor, s.minErle, s.maxErle[s.bandToSubband[k]])
			if s.useOnsetDetection {
				s.erleOnsetCompensated[ch][k] = stSafeClamp(averageErleOnsetCompensated[ch][k]*correctionFactor,
					s.minErle, s.maxErle[s.bandToSubband[k]])
			}
		}
	}
}

func stSubbandPowers(powerSpectrum []float32, out *[SignalDependentErleEstimatorSubbands]float32) {
	for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
		out[subband] = stAccumulate(powerSpectrum[stBandBoundaries[subband]:stBandBoundaries[subband+1]])
	}
}

func (s *SignalDependentErleEstimator) updateCorrectionFactors(X2 *[FftLengthBy2Plus1]float32, Y2, E2 [][FftLengthBy2Plus1]float32, convergedFilters []bool) {
	for ch := range convergedFilters {
		if !convergedFilters[ch] {
			continue
		}
		const smthConstantDecreases float32 = 0.1
		const smthConstantIncreases float32 = smthConstantDecreases / 2
		var X2Subbands, E2Subbands, Y2Subbands [SignalDependentErleEstimatorSubbands]float32
		stSubbandPowers(X2[:], &X2Subbands)
		stSubbandPowers(E2[ch][:], &E2Subbands)
		stSubbandPowers(Y2[ch][:], &Y2Subbands)
		var idxSubbands [SignalDependentErleEstimatorSubbands]int
		for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
			sec := s.nActiveSections[ch][stBandBoundaries[subband]:stBandBoundaries[subband+1]]
			m := sec[0]
			for _, v := range sec[1:] {
				m = min(m, v)
			}
			idxSubbands[subband] = m
		}
		var newErle [SignalDependentErleEstimatorSubbands]float32
		var isErleUpdated [SignalDependentErleEstimatorSubbands]bool
		for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
			if X2Subbands[subband] > stX2BandEnergyThreshold && E2Subbands[subband] > 0 {
				newErle[subband] = Y2Subbands[subband] / E2Subbands[subband]
				isErleUpdated[subband] = true
				s.numUpdates[ch][subband]++
			}
		}
		for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
			idx := idxSubbands[subband]
			est := &s.erleEstimators[ch][idx][subband]
			alpha := smthConstantDecreases
			if newErle[subband] > *est {
				alpha = smthConstantIncreases
			}
			alpha = stBoolToFloat(isErleUpdated[subband]) * alpha
			*est += float32(alpha * (newErle[subband] - *est))
			*est = stSafeClamp(*est, s.minErle, s.maxErle[subband])
		}
		for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
			ref := &s.erleRef[ch][subband]
			alpha := smthConstantDecreases
			if newErle[subband] > *ref {
				alpha = smthConstantIncreases
			}
			alpha = stBoolToFloat(isErleUpdated[subband]) * alpha
			*ref += float32(alpha * (newErle[subband] - *ref))
			*ref = stSafeClamp(*ref, s.minErle, s.maxErle[subband])
		}
		for subband := 0; subband < SignalDependentErleEstimatorSubbands; subband++ {
			const numUpdateThr = 50
			if isErleUpdated[subband] && s.numUpdates[ch][subband] > numUpdateThr {
				idx := idxSubbands[subband]
				newCorrectionFactor := s.erleEstimators[ch][idx][subband] / s.erleRef[ch][subband]
				cf := &s.correctionFactors[ch][idx][subband]
				*cf += float32(0.1 * (newCorrectionFactor - *cf))
			}
		}
	}
}

func (s *SignalDependentErleEstimator) computeEchoEstimatePerFilterSection(renderBuffer *RenderBuffer,
	filterFrequencyResponses [][][FftLengthBy2Plus1]float32) {
	spectrumRenderBuffer := renderBuffer.GetSpectrumBuffer()
	numRenderChannels := len(spectrumRenderBuffer.At(0))
	oneByNumRenderChannels := 1 / float32(numRenderChannels)
	for captureCh := range s.S2SectionAccum {
		idxRender := renderBuffer.Position()
		idxRender = spectrumRenderBuffer.OffsetIndex(idxRender, s.sectionBoundariesBlocks[0])
		for section := 0; section < s.numSections; section++ {
			var X2Section, H2Section [FftLengthBy2Plus1]float32
			blockLimit := min(s.sectionBoundariesBlocks[section+1], len(filterFrequencyResponses[captureCh]))
			for block := s.sectionBoundariesBlocks[section]; block < blockLimit; block++ {
				for renderCh := range spectrumRenderBuffer.At(idxRender) {
					x := &spectrumRenderBuffer.At(idxRender)[renderCh]
					for k := range X2Section {
						X2Section[k] += float32(x[k] * oneByNumRenderChannels)
					}
				}
				h := &filterFrequencyResponses[captureCh][block]
				for k := range H2Section {
					H2Section[k] = H2Section[k] + h[k]
				}
				idxRender = spectrumRenderBuffer.IncIndex(idxRender)
			}
			acc := &s.S2SectionAccum[captureCh][section]
			for k := range acc {
				acc[k] = X2Section[k] * H2Section[k]
			}
		}
		for section := 1; section < s.numSections; section++ {
			prev := &s.S2SectionAccum[captureCh][section-1]
			cur := &s.S2SectionAccum[captureCh][section]
			for k := range cur {
				cur[k] = prev[k] + cur[k]
			}
		}
	}
}

func (s *SignalDependentErleEstimator) computeActiveFilterSections() {
	for ch := range s.nActiveSections {
		s.nActiveSections[ch] = [FftLengthBy2Plus1]int{}
		for k := 0; k < FftLengthBy2Plus1; k++ {
			section := s.numSections
			target := 0.9 * s.S2SectionAccum[ch][s.numSections-1][k]
			for section > 0 && s.S2SectionAccum[ch][section-1][k] >= target {
				section--
				s.nActiveSections[ch][k] = section
			}
		}
	}
}
