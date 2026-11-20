package aec3

// AlignmentMixerMixingVariant mirrors AlignmentMixer::MixingVariant.
type AlignmentMixerMixingVariant int

const (
	MixingVariantDownmix AlignmentMixerMixingVariant = iota
	MixingVariantAdaptive
	MixingVariantFixed
)

func dlyChooseMixingVariant(downmix, adaptiveSelection bool, numChannels int) AlignmentMixerMixingVariant {
	if numChannels == 1 {
		return MixingVariantFixed
	}
	if downmix {
		return MixingVariantDownmix
	}
	if adaptiveSelection {
		return MixingVariantAdaptive
	}
	return MixingVariantFixed
}

// AlignmentMixer converts the channels to mono to provide a decent input for
// the delay estimation (alignment_mixer.h).
type AlignmentMixer struct {
	moreEnergies              []float32 // cumulativeEnergies of more than inlineChannels channels
	numChannels               int
	oneByNumChannels          float32
	excitationEnergyThreshold float32
	preferFirstTwoChannels    bool
	selectionVariant          AlignmentMixerMixingVariant
	strongBlockCounters       [2]int
	inlineEnergies            [inlineChannels]float32 // cumulativeEnergies of up to inlineChannels channels
	selectedChannel           int
	blockCounter              int
}

// NewAlignmentMixer mirrors AlignmentMixer(num_channels, config).
func NewAlignmentMixer(numChannels int, config AlignmentMixing) *AlignmentMixer {
	return NewAlignmentMixerWithParams(numChannels, config.Downmix, config.AdaptiveSelection,
		config.ActivityPowerThreshold, config.PreferFirstTwoChannels)
}

// NewAlignmentMixerWithParams mirrors the five-argument C++ constructor.
func NewAlignmentMixerWithParams(numChannels int, downmix, adaptiveSelection bool, activityPowerThreshold float32,
	preferFirstTwoChannels bool) *AlignmentMixer {
	m := &AlignmentMixer{}
	m.init(numChannels, downmix, adaptiveSelection, activityPowerThreshold, preferFirstTwoChannels)
	return m
}

// init sets up m in place.
func (m *AlignmentMixer) init(numChannels int, downmix, adaptiveSelection bool, activityPowerThreshold float32,
	preferFirstTwoChannels bool) {
	*m = AlignmentMixer{
		numChannels:               numChannels,
		oneByNumChannels:          1 / float32(numChannels),
		excitationEnergyThreshold: float32(BlockSize) * activityPowerThreshold,
		preferFirstTwoChannels:    preferFirstTwoChannels,
		selectionVariant:          dlyChooseMixingVariant(downmix, adaptiveSelection, numChannels),
	}
	if m.selectionVariant == MixingVariantAdaptive {
		m.moreEnergies = moreChannels[float32](numChannels)
	}
}

// ProduceOutput writes the mixed band-0 signal of x into y (BlockSize
// samples).
func (m *AlignmentMixer) ProduceOutput(x *Block, y []float32) {
	y = y[:BlockSize]
	if m.selectionVariant == MixingVariantDownmix {
		m.downmix(x, y)
		return
	}
	ch := 0
	if m.selectionVariant != MixingVariantFixed {
		ch = m.selectChannel(x)
	}
	copy(y, x.View(0, ch))
}

func (m *AlignmentMixer) downmix(x *Block, y []float32) {
	copy(y, x.View(0, 0))
	for ch := 1; ch < m.numChannels; ch++ {
		xCh := x.View(0, ch)
		for i := range BlockSize {
			y[i] += xCh[i]
		}
	}
	for i := range BlockSize {
		y[i] *= m.oneByNumChannels
	}
}

func (m *AlignmentMixer) selectChannel(x *Block) int {
	energies := channelState(&m.inlineEnergies, m.numChannels, m.moreEnergies)
	const blocksToChooseLeftOrRight = int(0.5 * NumBlocksPerSecond)
	goodSignalInLeftOrRight := m.preferFirstTwoChannels &&
		(m.strongBlockCounters[0] > blocksToChooseLeftOrRight || m.strongBlockCounters[1] > blocksToChooseLeftOrRight)

	numChToAnalyze := m.numChannels
	if goodSignalInLeftOrRight {
		numChToAnalyze = 2
	}

	const numBlocksBeforeEnergySmoothing = 60 * NumBlocksPerSecond
	m.blockCounter++

	for ch := 0; ch < numChToAnalyze; ch++ {
		var x2Sum float32
		xCh := x.View(0, ch)
		for i := range BlockSize {
			x2Sum += float32(xCh[i] * xCh[i])
		}

		if ch < 2 && x2Sum > m.excitationEnergyThreshold {
			m.strongBlockCounters[ch]++
		}

		if m.blockCounter <= numBlocksBeforeEnergySmoothing {
			energies[ch] += x2Sum
		} else {
			const smoothing = float32(1.0 / (10 * NumBlocksPerSecond))
			energies[ch] += float32(smoothing * (x2Sum - energies[ch]))
		}
	}

	// Normalize the energies to allow the energy computations to from now be
	// based on smoothing.
	if m.blockCounter == numBlocksBeforeEnergySmoothing {
		const oneByNumBlocksBeforeEnergySmoothing = float32(1.0 / numBlocksBeforeEnergySmoothing)
		for ch := 0; ch < numChToAnalyze; ch++ {
			energies[ch] *= oneByNumBlocksBeforeEnergySmoothing
		}
	}

	strongestCh := 0
	for ch := 0; ch < numChToAnalyze; ch++ {
		if energies[ch] > energies[strongestCh] {
			strongestCh = ch
		}
	}

	if (goodSignalInLeftOrRight && m.selectedChannel > 1) ||
		energies[strongestCh] > 2*energies[m.selectedChannel] {
		m.selectedChannel = strongestCh
	}
	return m.selectedChannel
}
