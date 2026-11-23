package bands

const (
	bsSamplesPerBand               = 160
	bsTwoBandFilterSamplesPerFrame = 320
)

// bsTwoBandsStates holds the QMF filter states of one channel
// (TwoBandsStates in splitting_filter.h).
type bsTwoBandsStates struct {
	AnalysisState1  [6]int32
	AnalysisState2  [6]int32
	SynthesisState1 [6]int32
	SynthesisState2 [6]int32
}

// SplittingFilter splits a 32 kHz or 48 kHz signal into 2 or 3 bands of
// 160 samples and merges them back
// (modules/audio_processing/splitting_filter.cc). Data is FloatS16, laid
// out as data[channel][sample] and bands[band][channel][sample]. The two-band
// filter is the fixed-point QMF, which rounds and saturates its input to
// int16; the three-band filter is a [ThreeBandFilterBank] per channel.
type SplittingFilter struct {
	numBands             int
	twoBandsStates       []bsTwoBandsStates
	threeBandFilterBanks []ThreeBandFilterBank

	// Scratch.
	bands16    [2][bsSamplesPerBand]int16
	fullBand16 [bsTwoBandFilterSamplesPerFrame]int16
}

// NewSplittingFilter returns a SplittingFilter for numChannels channels of
// numBands bands, which is 2 or 3. The third argument is the frame length
// of the C++ constructor, which follows from numBands.
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
		f.twoBandsStates = make([]bsTwoBandsStates, numChannels)
	} else {
		f.threeBandFilterBanks = make([]ThreeBandFilterBank, numChannels)
	}
}

// Analysis splits data[channel][numBands*160] into bands[band][channel][160].
func (f *SplittingFilter) Analysis(data [][]float32, bands [][][]float32) {
	switch len(bands) {
	case 2:
		f.twoBandsAnalysis(data, bands)
	case 3:
		f.threeBandsAnalysis(data, bands)
	}
}

// Synthesis merges bands[band][channel][160] into data[channel][numBands*160].
func (f *SplittingFilter) Synthesis(bands [][][]float32, data [][]float32) {
	switch len(bands) {
	case 2:
		f.twoBandsSynthesis(bands, data)
	case 3:
		f.threeBandsSynthesis(bands, data)
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

// twoBandsAnalysis is SplittingFilter::TwoBandsAnalysis.
func (f *SplittingFilter) twoBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.twoBandsStates {
		f.AnalysisChannel(i, data[i], bands[0][i], bands[1][i], nil)
	}
}

// twoBandsSynthesis is SplittingFilter::TwoBandsSynthesis.
func (f *SplittingFilter) twoBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		f.SynthesisChannel(i, bands[0][i], bands[1][i], nil, data[i])
	}
}

// threeBandsAnalysis is SplittingFilter::ThreeBandsAnalysis.
func (f *SplittingFilter) threeBandsAnalysis(data [][]float32, bands [][][]float32) {
	for i := range f.threeBandFilterBanks {
		f.AnalysisChannel(i, data[i], bands[0][i], bands[1][i], bands[2][i])
	}
}

// threeBandsSynthesis is SplittingFilter::ThreeBandsSynthesis.
func (f *SplittingFilter) threeBandsSynthesis(bands [][][]float32, data [][]float32) {
	for i := range data {
		f.SynthesisChannel(i, bands[0][i], bands[1][i], bands[2][i], data[i])
	}
}

// ChannelSplitter is the splitting filter of one channel: a pointer-free
// value whose zero value is ready, for callers that keep the channels in one
// array. A ChannelSplitter is used with one number of bands; its results are
// those of a [SplittingFilter] channel.
type ChannelSplitter struct {
	two   bsTwoBandsStates
	three ThreeBandFilterBank

	// Scratch.
	bands16    [2][bsSamplesPerBand]int16
	fullBand16 [bsTwoBandFilterSamplesPerFrame]int16
}

// Analysis splits data (numBands*160 samples, 2 or 3 bands) into the bands
// b0, b1 and, with three bands, b2, of 160 samples each
// (SplittingFilter.AnalysisChannel).
func (c *ChannelSplitter) Analysis(numBands int, data, b0, b1, b2 []float32) {
	if numBands == 2 {
		s := &c.two
		FloatS16ToS16Slice(data[:bsTwoBandFilterSamplesPerFrame], c.fullBand16[:])
		AnalysisQMF(c.fullBand16[:], c.bands16[0][:], c.bands16[1][:], &s.AnalysisState1, &s.AnalysisState2)
		S16ToFloatS16Slice(c.bands16[0][:], b0)
		S16ToFloatS16Slice(c.bands16[1][:], b1)
		return
	}
	out := [NumBands][]float32{b0, b1, b2}
	c.three.Analysis((*[FullBandSize]float32)(data), &out)
}

// Synthesis merges the bands b0, b1 and, with three bands, b2 (160 samples
// each) into data (numBands*160 samples) (SplittingFilter.SynthesisChannel).
func (c *ChannelSplitter) Synthesis(numBands int, b0, b1, b2, data []float32) {
	if numBands == 2 {
		s := &c.two
		FloatS16ToS16Slice(b0[:bsSamplesPerBand], c.bands16[0][:])
		FloatS16ToS16Slice(b1[:bsSamplesPerBand], c.bands16[1][:])
		SynthesisQMF(c.bands16[0][:], c.bands16[1][:], c.fullBand16[:], &s.SynthesisState1, &s.SynthesisState2)
		S16ToFloatS16Slice(c.fullBand16[:], data)
		return
	}
	in := [NumBands][]float32{b0, b1, b2}
	c.three.Synthesis(&in, (*[FullBandSize]float32)(data))
}
