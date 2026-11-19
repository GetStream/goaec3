package bands

const (
	bsSamplesPerBand               = 160
	bsTwoBandFilterSamplesPerFrame = 320
)

// TwoBandsStates holds the QMF filter states of one channel.
type TwoBandsStates struct {
	AnalysisState1  [6]int32
	AnalysisState2  [6]int32
	SynthesisState1 [6]int32
	SynthesisState2 [6]int32
}

// SplittingFilter splits a 32 kHz or 48 kHz signal into 2 or 3 bands of
// 160 samples (modules/audio_processing/splitting_filter.cc). Data is laid
// out as data[channel][sample] and bands[band][channel][sample].
type SplittingFilter struct {
	numBands             int
	twoBandsStates       []TwoBandsStates
	threeBandFilterBanks []ThreeBandFilterBank

	// Scratch.
	bands16    [2][bsSamplesPerBand]int16
	fullBand16 [bsTwoBandFilterSamplesPerFrame]int16
}

func NewSplittingFilter(numChannels, numBands, _ int) *SplittingFilter {
	f := &SplittingFilter{}
	f.Init(numChannels, numBands)
	return f
}

// Init sets f up for numChannels channels of 2 or 3 bands, with zero filter
// states. The per-channel states are one pointer-free allocation.
func (f *SplittingFilter) Init(numChannels, numBands int) {
	if numBands != 2 && numBands != 3 {
		panic("bands: SplittingFilter supports 2 or 3 bands")
	}
	*f = SplittingFilter{numBands: numBands}
	if numBands == 2 {
		f.twoBandsStates = make([]TwoBandsStates, numChannels)
	} else {
		f.threeBandFilterBanks = make([]ThreeBandFilterBank, numChannels)
	}
}

// Analysis splits data[channel][numBands*160] into bands[band][channel][160].
func (f *SplittingFilter) Analysis(data [][]float32, bands [][][]float32) {
	switch len(bands) {
	case 2:
		f.TwoBandsAnalysis(data, bands)
	case 3:
		f.ThreeBandsAnalysis(data, bands)
	}
}

// Synthesis merges bands[band][channel][160] into data[channel][numBands*160].
func (f *SplittingFilter) Synthesis(bands [][][]float32, data [][]float32) {
	switch len(bands) {
	case 2:
		f.TwoBandsSynthesis(bands, data)
	case 3:
		f.ThreeBandsSynthesis(bands, data)
	}
}

// AnalysisChannel splits data (numBands*160 samples of channel ch) into the
// bands b0, b1 and, with three bands, b2, of 160 samples each.
func (f *SplittingFilter) AnalysisChannel(ch int, data, b0, b1, b2 []float32) {
	if f.numBands == 2 {
		s := &f.twoBandsStates[ch]
		FloatS16ToS16Slice(data[:bsTwoBandFilterSamplesPerFrame], f.fullBand16[:])
		AnalysisQMF(f.fullBand16[:], f.bands16[0][:], f.bands16[1][:], &s.AnalysisState1, &s.AnalysisState2)
		S16ToFloatS16Slice(f.bands16[0][:], b0)
		S16ToFloatS16Slice(f.bands16[1][:], b1)
		return
	}
	out := [NumBands][]float32{b0, b1, b2}
	f.threeBandFilterBanks[ch].Analysis((*[FullBandSize]float32)(data), &out)
}

// SynthesisChannel merges the bands b0, b1 and, with three bands, b2 (160
// samples each) of channel ch into data (numBands*160 samples).
func (f *SplittingFilter) SynthesisChannel(ch int, b0, b1, b2, data []float32) {
	if f.numBands == 2 {
		s := &f.twoBandsStates[ch]
		FloatS16ToS16Slice(b0[:bsSamplesPerBand], f.bands16[0][:])
		FloatS16ToS16Slice(b1[:bsSamplesPerBand], f.bands16[1][:])
		SynthesisQMF(f.bands16[0][:], f.bands16[1][:], f.fullBand16[:], &s.SynthesisState1, &s.SynthesisState2)
		S16ToFloatS16Slice(f.fullBand16[:], data)
		return
	}
	in := [NumBands][]float32{b0, b1, b2}
	f.threeBandFilterBanks[ch].Synthesis(&in, (*[FullBandSize]float32)(data))
}

func (f *SplittingFilter) TwoBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.twoBandsStates {
		f.AnalysisChannel(i, data[i], bands[0][i], bands[1][i], nil)
	}
}

func (f *SplittingFilter) TwoBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		f.SynthesisChannel(i, bands[0][i], bands[1][i], nil, data[i])
	}
}

func (f *SplittingFilter) ThreeBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.threeBandFilterBanks {
		f.AnalysisChannel(i, data[i], bands[0][i], bands[1][i], bands[2][i])
	}
}

func (f *SplittingFilter) ThreeBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		f.SynthesisChannel(i, bands[0][i], bands[1][i], bands[2][i], data[i])
	}
}
