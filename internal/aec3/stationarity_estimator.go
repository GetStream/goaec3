package aec3

const (
	stMinNoisePower           float32 = 10
	stHangoverBlocks                  = NumBlocksPerSecond / 20
	stNBlocksAverageInitPhase         = 20
	stNBlocksInitialPhase             = NumBlocksPerSecond * 2
	stWindowLength                    = 13
)

// stNoiseSpectrum mirrors StationarityEstimator::NoiseSpectrum.
type stNoiseSpectrum struct {
	noiseSpectrum [FftLengthBy2Plus1]float32
	blockCounter  int
}

func (n *stNoiseSpectrum) Reset() {
	n.blockCounter = 0
	for k := range n.noiseSpectrum {
		n.noiseSpectrum[k] = stMinNoisePower
	}
}

func (n *stNoiseSpectrum) Power(band int) float32 { return n.noiseSpectrum[band] }

func (n *stNoiseSpectrum) Update(spectrum [][FftLengthBy2Plus1]float32) {
	numRenderChannels := len(spectrum)
	var avgSpectrumData [FftLengthBy2Plus1]float32
	var avgSpectrum *[FftLengthBy2Plus1]float32
	if numRenderChannels == 1 {
		avgSpectrum = &spectrum[0]
	} else {
		avgSpectrum = &avgSpectrumData
		avgSpectrumData = spectrum[0]
		for ch := 1; ch < numRenderChannels; ch++ {
			for k := 1; k < FftLengthBy2Plus1; k++ {
				avgSpectrumData[k] += spectrum[ch][k]
			}
		}
		oneByNumChannels := 1 / float32(numRenderChannels)
		for k := 1; k < FftLengthBy2Plus1; k++ {
			avgSpectrumData[k] *= oneByNumChannels
		}
	}
	n.blockCounter++
	alpha := n.getAlpha()
	for k := 0; k < FftLengthBy2Plus1; k++ {
		if n.blockCounter <= stNBlocksAverageInitPhase {
			n.noiseSpectrum[k] += float32(float32(1.0/stNBlocksAverageInitPhase) * avgSpectrum[k])
		} else {
			n.noiseSpectrum[k] = n.updateBandBySmoothing(avgSpectrum[k], n.noiseSpectrum[k], alpha)
		}
	}
}

// stTiltAlpha is (kAlphaInit - kAlpha) / kNBlocksInitialPhase in float.
var stTiltAlpha = stDiv(stSub(0.04, 0.004), stNBlocksInitialPhase)

func (n *stNoiseSpectrum) getAlpha() float32 {
	const alpha float32 = 0.004
	const alphaInit float32 = 0.04
	if n.blockCounter > stNBlocksInitialPhase+stNBlocksAverageInitPhase {
		return alpha
	}
	// The C++ subtracts in size_t, which wraps for the first blocks; the
	// value is unused there.
	d := float32(uint64(n.blockCounter) - stNBlocksAverageInitPhase)
	return alphaInit - float32(stTiltAlpha*d)
}

func (n *stNoiseSpectrum) updateBandBySmoothing(powerBand, powerBandNoise, alpha float32) float32 {
	powerBandNoiseUpdated := powerBandNoise
	if powerBandNoise < powerBand {
		alphaInc := alpha * (powerBandNoise / powerBand)
		if n.blockCounter > stNBlocksInitialPhase {
			if 10*powerBandNoise < powerBand {
				alphaInc *= 0.1
			}
		}
		powerBandNoiseUpdated += float32(alphaInc * (powerBand - powerBandNoise))
	} else {
		powerBandNoiseUpdated += float32(alpha * (powerBand - powerBandNoise))
		powerBandNoiseUpdated = stdMax(powerBandNoiseUpdated, stMinNoisePower)
	}
	return powerBandNoiseUpdated
}

// StationarityEstimator mirrors stationarity_estimator.h: estimates the
// stationarity of the render signal per band.
type StationarityEstimator struct {
	noise             stNoiseSpectrum
	hangovers         [FftLengthBy2Plus1]int
	stationarityFlags [FftLengthBy2Plus1]bool
}

func NewStationarityEstimator() *StationarityEstimator {
	s := &StationarityEstimator{}
	s.Reset()
	return s
}

func (s *StationarityEstimator) Reset() {
	s.noise.Reset()
	s.hangovers = [FftLengthBy2Plus1]int{}
	s.stationarityFlags = [FftLengthBy2Plus1]bool{}
}

// UpdateNoiseEstimator updates only the noise estimator.
func (s *StationarityEstimator) UpdateNoiseEstimator(spectrum [][FftLengthBy2Plus1]float32) {
	s.noise.Update(spectrum)
}

// UpdateStationarityFlags updates the per-band stationarity flags.
func (s *StationarityEstimator) UpdateStationarityFlags(spectrumBuffer *SpectrumBuffer,
	renderReverbContributionSpectrum []float32, idxCurrent, numLookahead int) {
	var indexes [stWindowLength]int
	numLookaheadBounded := min(numLookahead, stWindowLength-1)
	idx := idxCurrent
	if numLookaheadBounded < stWindowLength-1 {
		numLookback := (stWindowLength - 1) - numLookaheadBounded
		idx = spectrumBuffer.OffsetIndex(idxCurrent, numLookback)
	}
	indexes[0] = idx
	for k := 1; k < len(indexes); k++ {
		indexes[k] = spectrumBuffer.DecIndex(indexes[k-1])
	}
	for k := range s.stationarityFlags {
		s.stationarityFlags[k] = s.estimateBandStationarity(spectrumBuffer, renderReverbContributionSpectrum, &indexes, k)
	}
	s.updateHangover()
	s.smoothStationaryPerFreq()
}

// IsBandStationary reports whether band is stationary.
func (s *StationarityEstimator) IsBandStationary(band int) bool {
	return s.stationarityFlags[band] && s.hangovers[band] == 0
}

// IsBlockStationary reports whether most bands are stationary.
func (s *StationarityEstimator) IsBlockStationary() bool {
	var acumStationarity float32
	for band := range s.stationarityFlags {
		if s.IsBandStationary(band) {
			acumStationarity += 1
		}
	}
	return acumStationarity*float32(1.0/FftLengthBy2Plus1) > 0.75
}

func (s *StationarityEstimator) estimateBandStationarity(spectrumBuffer *SpectrumBuffer, averageReverb []float32,
	indexes *[stWindowLength]int, band int) bool {
	const thrStationarity float32 = 10
	var acumPower float32
	numRenderChannels := len(spectrumBuffer.At(0))
	oneByNumChannels := 1 / float32(numRenderChannels)
	for _, idx := range indexes {
		for ch := 0; ch < numRenderChannels; ch++ {
			acumPower += float32(spectrumBuffer.At(idx)[ch][band] * oneByNumChannels)
		}
	}
	acumPower += averageReverb[band]
	noise := stWindowLength * s.noise.Power(band)
	return acumPower < thrStationarity*noise
}

func (s *StationarityEstimator) areAllBandsStationary() bool {
	for _, b := range s.stationarityFlags {
		if !b {
			return false
		}
	}
	return true
}

func (s *StationarityEstimator) updateHangover() {
	reduceHangover := s.areAllBandsStationary()
	for k := range s.stationarityFlags {
		if !s.stationarityFlags[k] {
			s.hangovers[k] = stHangoverBlocks
		} else if reduceHangover {
			s.hangovers[k] = max(s.hangovers[k]-1, 0)
		}
	}
}

func (s *StationarityEstimator) smoothStationaryPerFreq() {
	var smooth [FftLengthBy2Plus1]bool
	for k := 1; k < FftLengthBy2Plus1-1; k++ {
		smooth[k] = s.stationarityFlags[k-1] && s.stationarityFlags[k] && s.stationarityFlags[k+1]
	}
	smooth[0] = smooth[1]
	smooth[FftLengthBy2Plus1-1] = smooth[FftLengthBy2Plus1-2]
	s.stationarityFlags = smooth
}
