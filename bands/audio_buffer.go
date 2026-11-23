package bands

// BandSplitter is the band splitting of webrtc::AudioBuffer
// (SplitIntoFrequencyBands and MergeFrequencyBands in audio_buffer.cc) on
// one 10 ms frame, for a buffer without resampling.
//
// Full-band data is data[channel][sampleRateHz/100] and split data is
// bands[band][channel][160], both FloatS16 and owned by the caller. At
// 16 kHz there is one band, and splitting and merging copy between data and
// bands[0].
type BandSplitter struct {
	numChannels      int
	numBands         int
	numFramesPerBand int
	splittingFilter  *SplittingFilter
}

// NewBandSplitter returns a BandSplitter for numChannels channels at
// sampleRateHz, which is 16000, 32000 or 48000.
func NewBandSplitter(sampleRateHz, numChannels int) *BandSplitter {
	numFrames := sampleRateHz / 100
	numBands := 1
	switch numFrames {
	case 320:
		numBands = 2
	case 480:
		numBands = 3
	}
	s := &BandSplitter{
		numChannels:      numChannels,
		numBands:         numBands,
		numFramesPerBand: numFrames / numBands,
	}
	if numBands > 1 {
		s.splittingFilter = NewSplittingFilter(numChannels, numBands, numFrames)
	}
	return s
}

// NumBands returns the number of bands: 1 at 16 kHz, 2 at 32 kHz and 3 at
// 48 kHz.
func (s *BandSplitter) NumBands() int { return s.numBands }

// NumFramesPerBand returns the number of samples of a band in a frame, 160.
func (s *BandSplitter) NumFramesPerBand() int { return s.numFramesPerBand }

// NumChannels returns the number of channels.
func (s *BandSplitter) NumChannels() int { return s.numChannels }

// SplitIntoFrequencyBands splits data into bands.
func (s *BandSplitter) SplitIntoFrequencyBands(data [][]float32, bands [][][]float32) {
	if s.splittingFilter == nil {
		for ch := 0; ch < s.numChannels; ch++ {
			copy(bands[0][ch][:s.numFramesPerBand], data[ch])
		}
		return
	}
	s.splittingFilter.Analysis(data[:s.numChannels], bands[:s.numBands])
}

// MergeFrequencyBands merges bands into data.
func (s *BandSplitter) MergeFrequencyBands(bands [][][]float32, data [][]float32) {
	if s.splittingFilter == nil {
		for ch := 0; ch < s.numChannels; ch++ {
			copy(data[ch][:s.numFramesPerBand], bands[0][ch])
		}
		return
	}
	s.splittingFilter.Synthesis(bands[:s.numBands], data[:s.numChannels])
}
