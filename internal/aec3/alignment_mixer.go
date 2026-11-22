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

// AlignmentMixer mixes the channels to mono for the delay estimation
// (alignment_mixer.h).
type AlignmentMixer struct {
	*alignmentMixerState
	moreEnergies []float32 // cumulativeEnergies of more than inlineChannels channels
}

// alignmentMixerState is the state of an AlignmentMixer except its slice.
type alignmentMixerState struct {
	moreEnergiesSpan          span
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

// NewAlignmentMixerWithParams mirrors the constructor with five arguments.
func NewAlignmentMixerWithParams(numChannels int, downmix, adaptiveSelection bool, activityPowerThreshold float32,
	preferFirstTwoChannels bool) *AlignmentMixer {
	m := &AlignmentMixer{}
	m.bind(&alignmentMixerState{}, bindAlone)
	m.init(numChannels, downmix, adaptiveSelection, activityPowerThreshold, preferFirstTwoChannels)
	return m
}

func (m *AlignmentMixer) bind(st *alignmentMixerState, b *binder) {
	m.alignmentMixerState = st
	b.f32(&m.moreEnergies, &st.moreEnergiesSpan)
}

func (m *AlignmentMixer) init(numChannels int, downmix, adaptiveSelection bool, activityPowerThreshold float32,
	preferFirstTwoChannels bool) {
	st := m.alignmentMixerState
	*st = alignmentMixerState{
		numChannels:               numChannels,
		oneByNumChannels:          1 / float32(numChannels),
		excitationEnergyThreshold: float32(BlockSize) * activityPowerThreshold,
		preferFirstTwoChannels:    preferFirstTwoChannels,
		selectionVariant:          dlyChooseMixingVariant(downmix, adaptiveSelection, numChannels),
	}
	*m = AlignmentMixer{}
	m.bind(st, bindAlone)
	if m.selectionVariant == MixingVariantAdaptive {
		m.moreEnergies = moreChannels[float32](numChannels)
	}
}

// ProduceOutput writes the mix of band 0 of x to the BlockSize samples of y.
func (m AlignmentMixer) ProduceOutput(x *Block, y []float32) {
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

func (m AlignmentMixer) downmix(x *Block, y []float32) {
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

func (m AlignmentMixer) selectChannel(x *Block) int {
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

	// Normalize the energies: from here on they are smoothed.
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
