package aec3

// [B,A] = butter(2,100/8000,'high')
var ec3HighPassFilterCoefficients16kHz = [1]BiQuadCoefficients{
	{B: [3]float32{0.972613898, -1.945227797, 0.972613898}, A: [2]float32{-1.944477658, 0.945977936}},
}

// [B,A] = butter(2,100/16000,'high')
var ec3HighPassFilterCoefficients32kHz = [1]BiQuadCoefficients{
	{B: [3]float32{0.986211925, -1.972423849, 0.986211925}, A: [2]float32{-1.972233729, 0.972613969}},
}

// [B,A] = butter(2,100/24000,'high')
var ec3HighPassFilterCoefficients48kHz = [1]BiQuadCoefficients{
	{B: [3]float32{0.990786698, -1.981573396, 0.990786698}, A: [2]float32{-1.981488509, 0.981658283}},
}

func ec3ChooseHighPassCoefficients(sampleRateHz int) []BiQuadCoefficients {
	switch sampleRateHz {
	case 16000:
		return ec3HighPassFilterCoefficients16kHz[:]
	case 32000:
		return ec3HighPassFilterCoefficients32kHz[:]
	case 48000:
		return ec3HighPassFilterCoefficients48kHz[:]
	}
	panic("aec3: invalid sample rate for HighPassFilter")
}

// HighPassFilter is the 100 Hz second order high-pass filter of
// modules/audio_processing/high_pass_filter.h, one CascadedBiQuadFilter per
// channel. The AudioBuffer overload of Process is not ported; Process takes
// the channels directly.
type HighPassFilter struct {
	sampleRateHz int
	filters      []*CascadedBiQuadFilter
}

// NewHighPassFilter returns a filter for numChannels channels at
// sampleRateHz (16000, 32000 or 48000).
func NewHighPassFilter(sampleRateHz, numChannels int) *HighPassFilter {
	f := &HighPassFilter{sampleRateHz: sampleRateHz, filters: make([]*CascadedBiQuadFilter, numChannels)}
	coefficients := ec3ChooseHighPassCoefficients(sampleRateHz)
	for k := range f.filters {
		f.filters[k] = NewCascadedBiQuadFilter(coefficients)
	}
	return f
}

// Process filters audio[channel] in place (the C++
// Process(std::vector<std::vector<float>>*) overload).
func (f *HighPassFilter) Process(audio [][]float32) {
	if len(audio) != len(f.filters) {
		panic("aec3: HighPassFilter channel count mismatch")
	}
	for k := range audio {
		f.filters[k].ProcessInPlace(audio[k])
	}
}

// Reset resets the filter states.
func (f *HighPassFilter) Reset() {
	for k := range f.filters {
		f.filters[k].Reset()
	}
}

// ResetChannels resets the filter states and changes the number of channels
// (the C++ Reset(size_t num_channels) overload).
func (f *HighPassFilter) ResetChannels(numChannels int) {
	oldNumChannels := len(f.filters)
	if numChannels <= oldNumChannels {
		f.filters = f.filters[:numChannels]
		f.Reset()
		return
	}
	for k := 0; k < oldNumChannels; k++ {
		f.filters[k].Reset()
	}
	coefficients := ec3ChooseHighPassCoefficients(f.sampleRateHz)
	for k := oldNumChannels; k < numChannels; k++ {
		f.filters = append(f.filters, NewCascadedBiQuadFilter(coefficients))
	}
}

// SampleRateHz returns the sample rate the filter was designed for.
func (f *HighPassFilter) SampleRateHz() int { return f.sampleRateHz }

// NumChannels returns the number of channels.
func (f *HighPassFilter) NumChannels() int { return len(f.filters) }
