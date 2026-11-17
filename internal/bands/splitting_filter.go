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
	if numBands != 2 && numBands != 3 {
		panic("bands: SplittingFilter supports 2 or 3 bands")
	}
	f := &SplittingFilter{numBands: numBands}
	if numBands == 2 {
		f.twoBandsStates = make([]TwoBandsStates, numChannels)
	} else {
		f.threeBandFilterBanks = make([]ThreeBandFilterBank, numChannels)
	}
	return f
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

func (f *SplittingFilter) TwoBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.twoBandsStates {
		s := &f.twoBandsStates[i]
		FloatS16ToS16Slice(data[i][:bsTwoBandFilterSamplesPerFrame], f.fullBand16[:])
		AnalysisQMF(f.fullBand16[:], f.bands16[0][:], f.bands16[1][:], &s.AnalysisState1, &s.AnalysisState2)
		S16ToFloatS16Slice(f.bands16[0][:], bands[0][i])
		S16ToFloatS16Slice(f.bands16[1][:], bands[1][i])
	}
}

func (f *SplittingFilter) TwoBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		s := &f.twoBandsStates[i]
		FloatS16ToS16Slice(bands[0][i][:bsSamplesPerBand], f.bands16[0][:])
		FloatS16ToS16Slice(bands[1][i][:bsSamplesPerBand], f.bands16[1][:])
		SynthesisQMF(f.bands16[0][:], f.bands16[1][:], f.fullBand16[:], &s.SynthesisState1, &s.SynthesisState2)
		S16ToFloatS16Slice(f.fullBand16[:], data[i])
	}
}

func (f *SplittingFilter) ThreeBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.threeBandFilterBanks {
		out := [NumBands][]float32{bands[0][i], bands[1][i], bands[2][i]}
		f.threeBandFilterBanks[i].Analysis((*[FullBandSize]float32)(data[i]), &out)
	}
}

func (f *SplittingFilter) ThreeBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		in := [NumBands][]float32{bands[0][i], bands[1][i], bands[2][i]}
		f.threeBandFilterBanks[i].Synthesis(&in, (*[FullBandSize]float32)(data[i]))
	}
}
