package aec3

// ReverbModel is an exponential reverberation model over power spectra
// (reverb_model.h).
type ReverbModel struct {
	reverb [FftLengthBy2Plus1]float32
}

func NewReverbModel() *ReverbModel { return &ReverbModel{} }

func (r *ReverbModel) Reset() { r.reverb = [FftLengthBy2Plus1]float32{} }

// Reverb returns the power spectrum of the reverb.
func (r *ReverbModel) Reverb() *[FftLengthBy2Plus1]float32 { return &r.reverb }

// UpdateReverbNoFreqShaping updates the reverb with one scaling for all
// frequencies.
func (r *ReverbModel) UpdateReverbNoFreqShaping(powerSpectrum []float32, powerSpectrumScaling, reverbDecay float32) {
	if reverbDecay > 0 {
		kernReverbScalar(r.reverb[:len(powerSpectrum)], powerSpectrum, powerSpectrumScaling, reverbDecay)
	}
}

// UpdateReverb updates the reverb with a scaling per frequency.
func (r *ReverbModel) UpdateReverb(powerSpectrum, powerSpectrumScaling []float32, reverbDecay float32) {
	if reverbDecay > 0 {
		kernReverb(r.reverb[:len(powerSpectrum)], powerSpectrum, powerSpectrumScaling, reverbDecay)
	}
}
