package bands

// BandSplitter performs what webrtc::AudioBuffer::SplitIntoFrequencyBands and
// MergeFrequencyBands do on one 10 ms frame (audio_buffer.cc), for a buffer
// without resampling.
//
// Full-band data is data[channel][sampleRateHz/100] and split data is
// bands[band][channel][160], both in FloatS16 scale and owned by the caller.
// AudioBuffer::CopyFrom on float input applies FloatToFloatS16 to each
// sample (see FloatToFloatS16Slice) before splitting, and CopyTo applies
// FloatS16ToFloat after merging.
//
// At 16 kHz AudioBuffer has a single band whose split view aliases the
// full-band data; Split and Merge then copy between data and bands[0].
type BandSplitter struct {
	numChannels      int
	numBands         int
	numFramesPerBand int
	splittingFilter  *SplittingFilter
}

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

func (s *BandSplitter) NumBands() int         { return s.numBands }
func (s *BandSplitter) NumFramesPerBand() int { return s.numFramesPerBand }
func (s *BandSplitter) NumChannels() int      { return s.numChannels }

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
