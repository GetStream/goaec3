package aec3

// ReverbModel mirrors reverb_model.h: an exponential reverberation model
// applied over power spectra.
type ReverbModel struct {
	reverb [FftLengthBy2Plus1]float32
}

func NewReverbModel() *ReverbModel { return &ReverbModel{} }

// Reset clears the state.
func (r *ReverbModel) Reset() { r.reverb = [FftLengthBy2Plus1]float32{} }

// Reverb returns the reverb power spectrum.
func (r *ReverbModel) Reverb() *[FftLengthBy2Plus1]float32 { return &r.reverb }

// UpdateReverbNoFreqShaping updates the reverb estimate with the same
// scaling for every frequency.
func (r *ReverbModel) UpdateReverbNoFreqShaping(powerSpectrum []float32, powerSpectrumScaling, reverbDecay float32) {
	if reverbDecay > 0 {
		for k := range powerSpectrum {
			r.reverb[k] = (r.reverb[k] + float32(powerSpectrum[k]*powerSpectrumScaling)) * reverbDecay
		}
	}
}

// UpdateReverb updates the reverb estimate with a per-frequency scaling.
func (r *ReverbModel) UpdateReverb(powerSpectrum, powerSpectrumScaling []float32, reverbDecay float32) {
	if reverbDecay > 0 {
		kernReverb(r.reverb[:len(powerSpectrum)], powerSpectrum, powerSpectrumScaling, reverbDecay)
	}
}
