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
// modules/audio_processing/high_pass_filter.h, one cascaded biquad filter per
// channel. The sections of every channel share one pointer-free backing
// array. The AudioBuffer overload of Process is not ported; Process takes
// the channels directly.
type HighPassFilter struct {
	sampleRateHz int
	numSections  int      // per channel
	sections     []BiQuad // [channel][numSections]
}

// newHighPassSections returns the sections of numChannels filters.
func newHighPassSections(coefficients []BiQuadCoefficients, numChannels int) []BiQuad {
	n := len(coefficients)
	sections := make([]BiQuad, numChannels*n)
	for k := range sections {
		sections[k] = BiQuad{Coefficients: coefficients[k%n]}
	}
	return sections
}

// NewHighPassFilter returns a filter for numChannels channels at
// sampleRateHz (16000, 32000 or 48000).
func NewHighPassFilter(sampleRateHz, numChannels int) *HighPassFilter {
	f := &HighPassFilter{}
	f.init(sampleRateHz, numChannels)
	return f
}

func (f *HighPassFilter) init(sampleRateHz, numChannels int) {
	coefficients := ec3ChooseHighPassCoefficients(sampleRateHz)
	*f = HighPassFilter{sampleRateHz: sampleRateHz, numSections: len(coefficients),
		sections: newHighPassSections(coefficients, numChannels)}
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
	n := f.numSections
	dlyApplyBiQuadCascade(audio, audio, f.sections[channel*n:(channel+1)*n])
}

// Reset resets the filter states.
func (f *HighPassFilter) Reset() {
	for k := range f.sections {
		f.sections[k].Reset()
	}
}

// ResetChannels resets the filter states and changes the number of channels.
func (f *HighPassFilter) ResetChannels(numChannels int) {
	oldNumChannels := f.NumChannels()
	if numChannels <= oldNumChannels {
		f.sections = f.sections[:numChannels*f.numSections]
		f.Reset()
		return
	}
	f.Reset()
	f.sections = append(f.sections, newHighPassSections(ec3ChooseHighPassCoefficients(f.sampleRateHz),
		numChannels-oldNumChannels)...)
}

// SampleRateHz returns the sample rate the filter was created for.
func (f *HighPassFilter) SampleRateHz() int { return f.sampleRateHz }

// NumChannels returns the number of channels.
func (f *HighPassFilter) NumChannels() int { return len(f.sections) / f.numSections }
