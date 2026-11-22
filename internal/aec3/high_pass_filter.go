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
// modules/audio_processing/high_pass_filter.h, one cascaded biquad filter
// (of one section) per channel. The sections of up to inlineChannels
// channels are part of the filter; those of more channels share one
// pointer-free backing array. The AudioBuffer overload of Process is not
// ported; Process takes the channels directly.
type HighPassFilter struct {
	more         []BiQuad // [channel], with more than inlineChannels channels
	sampleRateHz int
	numChannels  int
	inline       [inlineChannels]BiQuad
}

// sections returns the section of every channel.
func (f *HighPassFilter) sections() []BiQuad {
	if f.more != nil {
		return f.more
	}
	return f.inline[:f.numChannels]
}

// setChannels gives f numChannels channels, keeping the sections of the
// channels it has and adding new ones.
func (f *HighPassFilter) setChannels(numChannels int) {
	old := f.sections()
	sections := f.inline[:min(numChannels, len(f.inline))]
	if numChannels > len(f.inline) {
		sections = make([]BiQuad, numChannels)
	}
	copy(sections, old) // onto itself when the sections stay inline
	c := ec3ChooseHighPassCoefficients(f.sampleRateHz)[0]
	for k := len(old); k < numChannels; k++ {
		sections[k] = BiQuad{Coefficients: c}
	}
	f.more = nil
	if numChannels > len(f.inline) {
		f.more = sections
	}
	f.numChannels = numChannels
}

// NewHighPassFilter returns a filter for numChannels channels at
// sampleRateHz (16000, 32000 or 48000).
func NewHighPassFilter(sampleRateHz, numChannels int) *HighPassFilter {
	f := &HighPassFilter{}
	f.init(sampleRateHz, numChannels)
	return f
}

func (f *HighPassFilter) init(sampleRateHz, numChannels int) {
	*f = HighPassFilter{sampleRateHz: sampleRateHz}
	f.setChannels(numChannels)
}

// Process filters audio[channel] in place.
func (f *HighPassFilter) Process(audio [][]float32) {
	if len(audio) != f.NumChannels() {
		panic("aec3: HighPassFilter channel count mismatch")
	}
	for k := range audio {
		f.ProcessChannel(k, audio[k])
	}
}

// ProcessChannel filters the samples of one channel in place.
func (f *HighPassFilter) ProcessChannel(channel int, audio []float32) {
	dlyApplyBiQuadCascade(audio, audio, f.sections()[channel:channel+1])
}

// Reset resets the filter states.
func (f *HighPassFilter) Reset() {
	sections := f.sections()
	for k := range sections {
		sections[k].Reset()
	}
}

// ResetChannels resets the filter states and changes the number of channels.
func (f *HighPassFilter) ResetChannels(numChannels int) {
	f.Reset()
	f.setChannels(numChannels)
}

// NumChannels returns the number of channels.
func (f *HighPassFilter) NumChannels() int { return f.numChannels }
